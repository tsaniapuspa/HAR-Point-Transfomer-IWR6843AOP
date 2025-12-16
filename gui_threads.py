# General Library Imports
import numpy as np
import pandas as pd
import time
# PyQt imports
from PySide2.QtCore import QThread, Signal
import pyqtgraph as pg
from collections import deque
import tensorflow as tf
# from PyQt5.QtGui import QPixmap
from PySide2.QtGui import QPixmap

# Local Imports
from gui_parser import UARTParser
from gui_common import *
from graph_utilities import *
from queue import Queue
from threading import Thread

# Logger
import logging
log = logging.getLogger(__name__)

# Classifier Configurables
MAX_NUM_TRACKS = 20  # This could vary depending on the configuration file. Use 20 here as a safe likely maximum to ensure there's enough memory for the classifier

# Expected minimums and maximums to bound the range of colors used for coloring points
SNR_EXPECTED_MIN = 5
SNR_EXPECTED_MAX = 40
SNR_EXPECTED_RANGE = SNR_EXPECTED_MAX - SNR_EXPECTED_MIN
DOPPLER_EXPECTED_MIN = -30
DOPPLER_EXPECTED_MAX = 30
DOPPLER_EXPECTED_RANGE = DOPPLER_EXPECTED_MAX - DOPPLER_EXPECTED_MIN

# Different methods to color the points
COLOR_MODE_SNR = 'SNR'
COLOR_MODE_HEIGHT = 'Height'
COLOR_MODE_DOPPLER = 'Doppler'
COLOR_MODE_TRACK = 'Associated Track'

# Magic Numbers for Target Index TLV
TRACK_INDEX_WEAK_SNR = 253  # Point not associated, SNR too weak
TRACK_INDEX_BOUNDS = 254  # Point not associated, located outside boundary of interest
TRACK_INDEX_NOISE = 255  # Point not associated, considered as noise


class parseUartThread(QThread):
    fin = Signal(dict)

    def __init__(self, uParser, window_size=30, stride=1):
        QThread.__init__(self)
        self.parser = uParser

        # Tambahan dari pipeline real-time---------------------
        # self.queue = Queue()
        # self.predThread = Thread(target=self.prediction)
        
        self.queue_pred = Queue(maxsize=5)   # khusus prediksi
        self.queue_gui  = Queue(maxsize=5)   # khusus GUI

        self.predThread = Thread(target=self.prediction)

        self.predThread.daemon = True
        self.predThread.start()

        # ✅ Import keras + patch InputLayer global
        from keras.models import load_model
        import keras.layers as keras_layers

        class PatchedInputLayer(keras_layers.InputLayer):
            def __init__(self, *args, **kwargs):
                kwargs.pop("batch_shape", None)  # buang argumen batch_shape
                super().__init__(*args, **kwargs)

        # Override InputLayer di keras.layers
        keras_layers.InputLayer = PatchedInputLayer

        self.saved_model = tf.saved_model.load(
            r"E:\Real-Time Radar Point Transformer_frame_id\frame_id\best_pointtransformer_32_NoDBSCAN_SAVEDMODEL"
        )
        self.model = self.saved_model.signatures["serve"]   # signature inferensi

        # === Load StandardScaler untuk normalisasi fitur 8D ===
        import joblib
        self.scaler = joblib.load(r"E:\Real-Time Radar Point Transformer_frame_id\frame_id\E_RADAR TRANSFORMERscaler_standard_32_NoDBSCAN.pkl")

        # ✅ update class names sesuai dataset baru
        self.class_names = ['Berdiri', 'Duduk', 'Jalan', 'Jatuh']
        
        # parameter voxel lama (kalau masih dipakai di fungsi lain)
        self.x, self.y, self.z = 10, 32, 32
        self.x_min, self.x_max = -1.5, 1.5
        self.y_min, self.y_max = 0, 4
        self.z_min, self.z_max = 0, 2

        self.x_res = (self.x_max - self.x_min) / self.x
        self.y_res = (self.y_max - self.y_min) / self.y
        self.z_res = (self.z_max - self.z_min) / self.z

        # buffer untuk windowing prediksi
        self.frameBuffer = deque(maxlen=window_size)
        self.window_size = window_size
        self.stride = stride
        self.counter = 0
        # ----------------------------------------------
        
        self.vote_buffer = []
        self.vote_window = 3   # panjang jendela vote
        self.vote_threshold = 3  # minimal jumlah label yang sama

        self.timestamp = time.strftime("%m%d%Y%H%M%S")
        self.outputDir = f'./dataset/{self.timestamp}'
        # Ensure the directory is created only once
        os.makedirs(self.outputDir, exist_ok=True)
        fin = Signal(dict)
        predSignal = Signal(str) 
        

    def run(self):
        if self.parser.parserType == "SingleCOMPort":
            outputDict = self.parser.readAndParseUartSingleCOMPort()
        else:
            outputDict = self.parser.readAndParseUartDoubleCOMPort()

        # --- push ke GUI queue, non-blocking ---
        try:
            self.queue_gui.put_nowait(outputDict)
        except:
            pass

        # --- DEBUG RAW POINTCLOUD TO CSV ----------------------------------------------
        pc = outputDict["pointCloud"]    # NxC array
        frame_num = outputDict.get("frameNum", -1)

        # buat dataframe dengan nama kolom dinamis
        num_cols = pc.shape[1]
        colnames = [f"col_{i}" for i in range(num_cols)]

        df = pd.DataFrame(pc, columns=colnames)
        df["frameNum"] = frame_num   # biar tau dari frame mana

        # append ke file CSV
        df.to_csv("debug_awal_dataset.csv", mode="a", index=False, header=False)
        # -------------------------------------------------------------------------------

        self.fin.emit(outputDict)

        # ==== SAVE RAW DATASET ====
        frameNum = outputDict.get("frameNum", None)
        self.save_raw_dataset_csv(outputDict, frameNum)
        # ==========================
        
        
        # Akses saveBinary melalui self.parser
        if self.parser.saveBinary == 1:
            # Simpan data ke JSON dan CSV
            frameJSON = {'frameData': outputDict,
                         'timestamp': time.time() * 1000}
            
            # ==== Tambahan baru (wajib agar format sama real-time) ====
            df_frame = self.parser.convertFrameDataToDataFrame_pengambilan(frameJSON)
            frameJSON["csvFrame"] = df_frame
            # ==========================================================
            
            # === DEBUG: SAVE FRAME YANG UDAH DIPROSES OLEH convertFrameDataToDataFrame ===
            df_frame.to_csv(
                "debug_after_convert_pengambilandataset.csv",
                mode='a',
                header=not os.path.exists("debug_after_convert_realtime.csv"),
                index=False
            )
            # ========================================================================
                    
            self.parser.frames.append(frameJSON)
            csvFilePath = f'{self.outputDir}/dataset.csv'

            # Simpan CSV
            self.parser.saveDataToCsv(csvFilePath, frameJSON)
            
            # === Tahap 2 + 3: pc_128 masuk buffer untuk windowing ===
            pc_128 = self.extract_features(frameJSON)  # (128, 8)

            # pastikan tidak kosong
            if pc_128 is not None and pc_128.shape == (128, 8):
                self.frameBuffer.append(pc_128)
            else:
                print("[BUF] pc_128 INVALID → SKIP")

            # === Sliding: proses kalau buffer sudah penuh ===
            if len(self.frameBuffer) >= self.window_size:
                if self.counter % self.stride == 0:
                    window = np.array(list(self.frameBuffer), dtype=np.float32)  # should be (30,3)
                    self.process_window(window)
                self.counter += 1
            else:
                print(f"[SLIDE] Skip frame, counter={self.counter}")
            
    def save_raw_dataset_csv(self, outputDict, frameNum):
        import csv, os
        import numpy as np

        path = "RAW_Dataset2.csv"
        write_header = not os.path.exists(path)

        # Ambil point cloud
        pc = outputDict.get("pointCloud", None)
        if pc is None:
            pc = outputDict.get("detectedPoints", None)
        if pc is None:
            pc = []

        if len(pc) == 0:
            return

        # fungsi aman untuk dict / numpy array
        def extract_value(p, idx):
            key_map = ["range", "doppler", "snr", "azimuth", "elevation"]

            if isinstance(p, dict):
                return p.get(key_map[idx], "")

            try:
                return p[idx]
            except:
                return ""

        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow(["frameNum", "pointIdx",
                                "range", "doppler", "snr",
                                "azimuth", "elevation"])

            for i, p in enumerate(pc):
                writer.writerow([
                    frameNum,
                    i,
                    extract_value(p, 0),
                    extract_value(p, 1),
                    extract_value(p, 2),
                    extract_value(p, 3),
                    extract_value(p, 4),
                ])

    def process_window(self, window):
        """
        window shape: (30,128,8)
        Tahap normalisasi + kirim ke thread prediksi
        """
        # 1. Validasi shape
        if window is None or window.shape != (self.window_size, 128, 8):
            print(f"[PROC] INVALID SHAPE → {window.shape}, expected ({self.window_size},128,8)")
            return

        # 2. Flatten untuk normalisasi
        flat = window.reshape(-1, 8)   # (30*128, 8)
        
        # 3. Apply scaler
        flat_norm = self.scaler.transform(flat)

        # 4. Bersihkan NaN / inf
        flat_norm = np.nan_to_num(flat_norm, nan=0.0, posinf=1.0, neginf=-1.0)

        # 5. Kembalikan bentuk ke (30,128,8)
        window_norm = flat_norm.reshape(self.window_size, 128, 8)

        # 6. Kirim ke prediction thread
        try:
            self.queue_pred.put_nowait(window_norm.astype(np.float32))
        except:
            pass


    def extract_features(self, frameJSON):
        # Ambil dataframe HASIL convertFrameDataToDataFrame
        df = frameJSON.get("csvFrame", None)

        if df is None or len(df) == 0:
            return np.zeros((128, 8), dtype=np.float32)

        # Ambil kolom fitur EXACT seperti pipeline CSV
        try:
            pc = df[["x","y","z","doppler","SNR","Range","Azimuth","Elevation"]].values.astype(np.float32)
        except:
            print("[FEAT] DF missing required columns!")
            return np.zeros((128, 8), dtype=np.float32)

        N = pc.shape[0]

        # Sampling 128 → SAMA dengan pipeline CSV
        if N >= 128:
            idx = np.random.choice(N, 128, replace=False)
        else:
            idx = np.random.choice(N, 128, replace=True)

        pc_128 = pc[idx]
        return pc_128
     
    def stop(self):
        self.terminate()

    def stop(self):
        self.terminate()

    def prediction(self):
        while True:
            if not self.queue_pred.empty():
                window = self.queue_pred.get()  # (30,128,8)
                print(f"[PRED] Got window: {window.shape}")
            
                # Validasi shape window
                if window.shape != (self.window_size, 128, 8):
                    log.warning(f"[Prediction] Skip window shape {window.shape}, expected ({self.window_size},128,8)")
                    continue

                # Tambahkan batch dim → (1,30,128,8)
                input_tensor = np.expand_dims(window, axis=0).astype(np.float32)

                try:
                    # === SavedModel inferensi ===
                    output = self.model(tf.constant(input_tensor))   # panggil signature
                    probs = output["output_0"].numpy()[0]            # ambil output

                    pred_idx = np.argmax(probs)
                    label = self.class_names[pred_idx]
                    confidence = probs[pred_idx]

                    print(f"[PRED] Result: {label} ({confidence:.3f})")
                    
                    self.last_pred = (label, probs)

                    # --- Majority Vote buffer ---
                    self.vote_buffer.append(label)
                    if len(self.vote_buffer) > self.vote_window:
                        self.vote_buffer.pop(0)

                    # --- Hitung majority ---
                    unique, counts = np.unique(self.vote_buffer, return_counts=True)
                    top_label = unique[np.argmax(counts)]
                    top_count = counts.max()

                    # Jika belum mencapai mayoritas → skip
                    if top_count < self.vote_threshold:
                        continue

                    # --- Tambahan syarat confidence minimal 0.9 ---
                    if confidence < 0.96:
                        continue

                    # Lolos semua syarat → stable
                    stable_label = top_label

                    # Update GUI
                    if hasattr(self, 'guiWindow'):
                        self.guiWindow.predictionLabel.setText(
                            f"Aktivitas: {stable_label} ({confidence*100:.1f}%)"
                        )
                        self.guiWindow.updatePredictionImage(stable_label)

                except Exception as e:
                    log.error(f"[Prediction Error] {e}")
                    self.last_pred = ("error", np.zeros(len(self.class_names)))

            else:
                time.sleep(0.05)


class sendCommandThread(QThread):
    done = Signal()

    def __init__(self, uParser, command):
        QThread.__init__(self)
        self.parser = uParser
        self.command = command

    def run(self):
        self.parser.sendLine(self.command)
        self.done.emit()

class updateQTTargetThread3D(QThread):
    done = Signal()

    def __init__(self, pointCloud, targets, scatter, pcplot, numTargets, ellipsoids, coords, colorGradient=None, classifierOut=[], zRange=[-3, 3], pointColorMode="", drawTracks=True, trackColorMap=None, pointBounds={'enabled': False}):
        QThread.__init__(self)
        self.pointCloud = pointCloud
        self.targets = targets
        self.scatter = scatter
        self.pcplot = pcplot
        self.colorArray = ('r', 'g', 'b', 'w')
        self.numTargets = numTargets
        self.ellipsoids = ellipsoids
        self.coordStr = coords
        self.classifierOut = classifierOut
        self.zRange = zRange
        self.colorGradient = colorGradient
        self.pointColorMode = pointColorMode
        self.drawTracks = drawTracks
        self.trackColorMap = trackColorMap
        self.pointBounds = pointBounds
        # This ignores divide by 0 errors when calculating the log2
        np.seterr(divide='ignore')

    def drawTrack(self, track, trackColor):
        # Get necessary track data
        tid = int(track[0])
        x = track[1]
        y = track[2]
        z = track[3]

        track = self.ellipsoids[tid]
        mesh = getBoxLinesCoords(x, y, z)
        track.setData(pos=mesh, color=trackColor, width=2,
                      antialias=True, mode='lines')
        track.setVisible(True)

    # Return transparent color if pointBounds is enabled and point is outside pointBounds
    # Otherwise, color the point depending on which color mode we are in
    def getPointColors(self, i):
        if (self.pointBounds['enabled']):
            xyz_coords = self.pointCloud[i, 0:3]
            if (xyz_coords[0] < self.pointBounds['minX']
                        or xyz_coords[0] > self.pointBounds['maxX']
                        or xyz_coords[1] < self.pointBounds['minY']
                        or xyz_coords[1] > self.pointBounds['maxY']
                        or xyz_coords[2] < self.pointBounds['minZ']
                        or xyz_coords[2] > self.pointBounds['maxZ']
                    ) :
                return pg.glColor((0, 0, 0, 0))

        # Color the points by their SNR
        if (self.pointColorMode == COLOR_MODE_SNR):
            snr = self.pointCloud[i, 4]
            # SNR value is out of expected bounds, make it white
            if (snr < SNR_EXPECTED_MIN) or (snr > SNR_EXPECTED_MAX):
                return pg.glColor('w')
            else:
                return pg.glColor(self.colorGradient.getColor((snr-SNR_EXPECTED_MIN)/SNR_EXPECTED_RANGE))

        # Color the points by their Height
        elif (self.pointColorMode == COLOR_MODE_HEIGHT):
            zs = self.pointCloud[i, 2]

            # Points outside expected z range, make it white
            if (zs < self.zRange[0]) or (zs > self.zRange[1]):
                return pg.glColor('w')
            else:
                colorRange = self.zRange[1]+abs(self.zRange[0])
                zs = self.zRange[1] - zs
                return pg.glColor(self.colorGradient.getColor(abs(zs/colorRange)))

        # Color Points by their doppler
        elif (self.pointColorMode == COLOR_MODE_DOPPLER):
            doppler = self.pointCloud[i, 3]
            # Doppler value is out of expected bounds, make it white
            if (doppler < DOPPLER_EXPECTED_MIN) or (doppler > DOPPLER_EXPECTED_MAX):
                return pg.glColor('w')
            else:
                return pg.glColor(self.colorGradient.getColor((doppler-DOPPLER_EXPECTED_MIN)/DOPPLER_EXPECTED_RANGE))

        # Color the points by their associate track
        elif (self.pointColorMode == COLOR_MODE_TRACK):
            trackIndex = int(self.pointCloud[i, 6])
            # trackIndex of 253, 254, or 255 indicates a point isn't associated to a track, so check for those magic numbers here
            if (trackIndex == TRACK_INDEX_WEAK_SNR or trackIndex == TRACK_INDEX_BOUNDS or trackIndex == TRACK_INDEX_NOISE):
                return pg.glColor('w')
            else:
                # Catch any errors that may occur if track or point index go out of bounds
                try:
                    return self.trackColorMap[trackIndex]
                except Exception as e:
                    log.error(e)
                    return pg.glColor('w')

        # Unknown Color Option, make all points green
        else:
            return pg.glColor('g')

    def run(self):

        # if self.pointCloud is None or len(self.pointCloud) == 0:
        #     print("Point Cloud is empty or None.")
        # else:
        #     print("Point Cloud Shape:", self.pointCloud.shape)

        # Clear all previous targets
        for e in self.ellipsoids:
            if (e.visible()):
                e.hide()
        try:
            # Create a list of just X, Y, Z values to be plotted
            if (self.pointCloud is not None):
                toPlot = self.pointCloud[:, 0:3]
                # print("Data for Visualization:", toPlot)

                # Determine the size of each point based on its SNR
                with np.errstate(divide='ignore'):
                    size = np.log2(self.pointCloud[:, 4])

                # Each color is an array of 4 values, so we need an numPoints*4 size 2d array to hold these values
                pointColors = np.zeros((self.pointCloud.shape[0], 4))

                # Set the color of each point
                for i in range(self.pointCloud.shape[0]):
                    pointColors[i] = self.getPointColors(i)

                # Plot the points
                self.scatter.setData(pos=toPlot, color=pointColors, size=size)
                # Debugging
                # print("Pos Data for Visualization:", toPlot)
                # print("Color Data for Visualization:", pointColors)
                # print("Size Data for Visualization:", size)

                # Make the points visible
                self.scatter.setVisible(True)
            else:
                # Make the points invisible if none are detected.
                self.scatter.setVisible(False)
        except Exception as e:
            log.error(
                "Unable to draw point cloud, ignoring and continuing execution...")
            print("Unable to draw point cloud, ignoring and continuing execution...")
            print(f"Error in point cloud visualization: {e}")

        # Graph the targets
        try:
            if (self.drawTracks):
                if (self.targets is not None):
                    for track in self.targets:
                        trackID = int(track[0])
                        trackColor = self.trackColorMap[trackID]
                        self.drawTrack(track, trackColor)
        except:
            log.error(
                "Unable to draw all tracks, ignoring and continuing execution...")
            print("Unable to draw point cloud, ignoring and continuing execution...")
            print(f"Error in point cloud visualization: {e}")
        self.done.emit()

    def stop(self):
        self.terminate()
