import cv2
import numpy as np

from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtGui import QImage, QPainter, QPen, QBrush, QPainterPath
from PySide6.QtCore import Qt, QRectF, QSize

from gui.theme import (
    PANEL_BG, PANEL_FILL, PANEL_BORDER,
    CORNER_RADIUS, PANEL_INSET,
)


class CameraWidget(QWidget):
    """Single camera view — 16:10 container with letterboxed content."""

    ASPECT_W = 16
    ASPECT_H = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self._qimage = None
        self.setMinimumSize(160, 100)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

    def update_frame(self, bgr_frame: np.ndarray):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        self._qimage = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        painter.fillRect(self.rect(), PANEL_BG)

        r = CORNER_RADIUS
        inset = PANEL_INSET
        panel = QRectF(inset, inset, self.width() - 2 * inset, self.height() - 2 * inset)

        path = QPainterPath()
        path.addRoundedRect(panel, r, r)
        painter.setClipPath(path)

        painter.fillPath(path, QBrush(PANEL_FILL))

        if self._qimage is not None:
            pw = panel.width()
            ph = panel.height()
            img_w = self._qimage.width()
            img_h = self._qimage.height()

            scale = min(pw / img_w, ph / img_h)
            draw_w = int(img_w * scale)
            draw_h = int(img_h * scale)
            x = panel.x() + (pw - draw_w) / 2
            y = panel.y() + (ph - draw_h) / 2

            scaled = self._qimage.scaled(
                draw_w, draw_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawImage(int(x), int(y), scaled)

        painter.setClipping(False)
        painter.setPen(QPen(PANEL_BORDER, 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(panel, r, r)

        painter.end()

    def heightForWidth(self, w):
        return int(w * self.ASPECT_H / self.ASPECT_W)

    def hasHeightForWidth(self):
        return True

    def sizeHint(self):
        return QSize(640, 400)
