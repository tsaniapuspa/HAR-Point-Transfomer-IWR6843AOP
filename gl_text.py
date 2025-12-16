import numpy as np
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.Qt import QtGui, QtCore
from OpenGL.GL import glColor4f


class GLTextItem(GLGraphicsItem):
    def __init__(self, X=0, Y=0, Z=0, text=""):
        super().__init__()
        self.X = X
        self.Y = Y
        self.Z = Z
        self.text = text
        self.GLViewWidget = None

    def setGLViewWidget(self, GLViewWidget):
        self.GLViewWidget = GLViewWidget
        self.update()

    # Compatibility
    def setX(self, X): self.X = float(X); self.update()
    def setY(self, Y): self.Y = float(Y); self.update()
    def setZ(self, Z): self.Z = float(Z); self.update()
    def setText(self, text): self.text = str(text); self.update()

    def setPosition(self, X, Y, Z):
        # self.X = float(X)
        # self.Y = float(Y)
        # self.Z = float(Z)
        # self.update()
        
        return

    def paint(self):
        # if self.GLViewWidget is None:
        #     return

        # glColor4f(1.0, 1.0, 1.0, 1.0)
        # # gunakan fungsi OpenGL bawaan (pasti supported)
        # self.GLViewWidget.renderText(
        #     float(self.X),
        #     float(self.Y),
        #     float(self.Z),
        #     self.text,
        #     QtGui.QFont("Arial", 12)
        # )
        return
