import sys
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPainter
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem
)

# 图片查看器类，用于显示放大的图片
class ImageViewer(QDialog):
    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        self.setWindowTitle("图片查看器")
        self.setWindowFlags(Qt.Dialog | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.setMinimumSize(800, 600)
        
        # 设置深色背景
        self.setStyleSheet("""
            QDialog {
                background-color: #121826;
                border: 1px solid #2c3e50;
            }
        """)
        
        # 创建图像显示区域
        self.view = QGraphicsView(self)
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # 添加图像
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        
        # 设置缩放级别按钮
        zoom_in_button = QPushButton("放大", self)
        zoom_in_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        zoom_in_button.clicked.connect(self.zoom_in)
        
        zoom_out_button = QPushButton("缩小", self)
        zoom_out_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        zoom_out_button.clicked.connect(self.zoom_out)
        
        reset_button = QPushButton("重置大小", self)
        reset_button.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
        """)
        reset_button.clicked.connect(self.reset_zoom)
        
        # 添加布局
        layout = QVBoxLayout(self)
        button_layout = QHBoxLayout()
        button_layout.addWidget(zoom_in_button)
        button_layout.addWidget(zoom_out_button)
        button_layout.addWidget(reset_button)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        layout.addWidget(self.view)
        
        # 设置图像显示
        self.view.setSceneRect(self.pixmap_item.boundingRect())
        self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        
        # 显示关闭提示
        close_hint = QLabel("按ESC键或点击关闭按钮退出", self)
        close_hint.setStyleSheet("color: #ecf0f1; font-size: 12px;")
        close_hint.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(close_hint)
        
        # 添加状态栏显示缩放比例
        self.zoom_level = 1.0
        self.status_label = QLabel(f"缩放比例: 100%", self)
        self.status_label.setStyleSheet("color: #ecf0f1; font-size: 12px;")
        layout.addWidget(self.status_label)
        
        # 设置窗口尺寸和位置
        self.resize(900, 700)
        
    def resizeEvent(self, event):
        self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        super().resizeEvent(event)
        
    def zoom_in(self):
        self.zoom_level *= 1.25
        self.view.scale(1.25, 1.25)
        self.update_status()
        
    def zoom_out(self):
        self.zoom_level *= 0.8
        self.view.scale(0.8, 0.8)
        self.update_status()
        
    def reset_zoom(self):
        self.view.resetTransform()
        self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self.zoom_level = 1.0
        self.update_status()
        
    def update_status(self):
        self.status_label.setText(f"缩放比例: {int(self.zoom_level * 100)}%")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.accept()
        super().keyPressEvent(event)