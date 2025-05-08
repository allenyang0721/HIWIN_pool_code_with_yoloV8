import sys
import socket
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QSpinBox, QMessageBox
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("機械手臂控制介面")

        # 視訊顯示區
        self.video_label = QLabel(alignment=Qt.AlignCenter)
        self.video_label.setScaledContents(True)
        self.cap = None

        # 座標微調區
        self.spinboxes = []
        grid = QGridLayout()
        axis_names = ["X", "Y", "Z", "A", "B", "C"]
        for i, name in enumerate(axis_names):
            lbl = QLabel(name)
            sb = QSpinBox()
            sb.setRange(-100000, 100000)
            sb.setValue(0)
            self.spinboxes.append(sb)
            grid.addWidget(lbl, i // 3, (i % 3) * 2)
            grid.addWidget(sb, i // 3, (i % 3) * 2 + 1)

        # 按鈕區
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_command)
        self.open_cam_btn = QPushButton("Open Camera")
        self.open_cam_btn.clicked.connect(self.toggle_camera)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.send_btn)
        btn_layout.addWidget(self.open_cam_btn)

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addLayout(grid)
        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)

        # 定時器更新影像
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        # 注意：預填當前座標功能需由手臂系統端提供對應支援，此處暫時移除自動抓取。

    def toggle_camera(self):
        if not self.cap:
            self.cap = cv2.VideoCapture(2)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not self.cap.isOpened():
                QMessageBox.warning(self, "Error", "無法開啟攝影機")
                self.cap = None
                return
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_label.setFixedSize(width, height)
            self.timer.start(30)
            self.open_cam_btn.setText("Close Camera")
        else:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.video_label.clear()
            self.open_cam_btn.setText("Open Camera")

    def update_frame(self):
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        h, w = frame.shape[:2]
        length = 20
        cx, cy = w // 2, h // 2
        # 繪製中心 + 準星
        cv2.line(frame, (cx - length, cy), (cx + length, cy), (0, 255, 0), 2)
        cv2.line(frame, (cx, cy - length), (cx, cy + length), (0, 255, 0), 2)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img))

    def send_command(self):
        coords = [sb.value() for sb in self.spinboxes]
        data = ",".join(str(v) for v in coords) + "\n"
        try:
            with socket.create_connection(("127.0.0.1", 9000), timeout=2) as sock:
                sock.sendall(data.encode())
        except Exception as e:
            QMessageBox.critical(self, "Send Error", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
