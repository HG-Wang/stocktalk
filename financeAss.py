import sys
import threading
import datetime
import os

import requests
import yfinance as yf
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
from PySide6.QtCore import Qt, QAbstractListModel, QModelIndex, QSize, QRectF, Signal, QTimer, QPoint, QRect
from PySide6.QtGui import QColor, QPainter, QFont, QPixmap, QAction, QPalette, QTextDocument
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QTextEdit,
    QPushButton,
    QListView,
    QLabel,
    QHBoxLayout,
    QFrame,
    QFileDialog,
    QToolButton,
    QInputDialog
)

from interface.Message import MessageDelegate,MessageItem,MessageModel

# 设置字体为支持中文的字体，例如 SimSun（宋体）、SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False    # 解决坐标轴负号显示问题

API_KEY = '95391dfcc5194055a072f0662c7e69ab'

def plot_stock_prediction(hist, lstm_pred_rescaled, actual, conf_interval, stock_code):
    """
    绘制股票价格预测图。

    参数:
    - hist: 股票历史数据
    - lstm_pred_rescaled: LSTM预测结果（反标准化）
    - actual: 实际的股票收盘价
    - conf_interval: 预测结果的99%置信区间
    - stock_code: 股票代码，用于图表标题和保存图像
    """
    plt.figure(figsize=(12, 6))
    plt.plot(hist.index[-len(lstm_pred_rescaled):], actual, label='实际价格', color='blue')
    plt.plot(hist.index[-len(lstm_pred_rescaled):], lstm_pred_rescaled, label='预测价格', color='red', linestyle='--')
    plt.fill_between(hist.index[-len(lstm_pred_rescaled):],
                     lstm_pred_rescaled - conf_interval,
                     lstm_pred_rescaled + conf_interval,
                     alpha=0.2, color='gray', label='99%置信区间')
    plt.title(f'{stock_code} 股票价格预测')
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.legend()
    plt.tight_layout()
    # 保存图像到本地
    plt.savefig(f'{stock_code}_prediction.png')
    plt.close()


class TelegramLikeChat(QWidget):
    responseReady = Signal(str)

    def __init__(self):
        super().__init__()

        self.user_avatar_path = 'user_avatar.jpg'
        self.assistant_avatar_path = 'assistant_avatar.jpg'

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setFixedSize(450, 800)

        self.background_image = None
        self.setStyleSheet("background-color: #f5f5f5;")

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        title_bar = QFrame()
        title_bar.setFixedHeight(40)
        title_bar.setStyleSheet("background-color: #007bff;")
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(10, 0, 10, 0)

        title = QLabel("实时交易信号智能对话助手")
        title_font = QFont("Segoe UI", 16, QFont.Bold)
        title.setFont(title_font)
        title.setStyleSheet("color: white;")
        title.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        title_layout.addWidget(title)

        title_layout.addStretch()

        settings_button = QToolButton()
        settings_button.setText("⚙️")
        settings_button.setStyleSheet("""
            QToolButton {
                color: white;
                border: none;
                font-size: 16px;
            }
            QToolButton:hover {
                color: #dddddd;
            }
        """)
        settings_button.clicked.connect(self.open_settings)
        title_layout.addWidget(settings_button)

        self.layout.addWidget(title_bar)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(10)

        self.model = MessageModel()
        self.list_view = QListView()
        self.list_view.setModel(self.model)
        self.list_view.setItemDelegate(MessageDelegate(self.list_view))
        self.list_view.setWordWrap(True)

        self.list_view.setStyleSheet("""
            QListView {
                background-color: rgba(255, 255, 255, 200);
                border: 1px solid #dedede;
                border-radius: 10px;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 10px;
                margin: 0px 0px 0px 0px;
            }
            QScrollBar::handle:vertical {
                background: #c1c1c1;
                min-height: 30px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        self.list_view.setVerticalScrollMode(QListView.ScrollPerPixel)
        self.list_view.setHorizontalScrollMode(QListView.ScrollPerPixel)
        content_layout.addWidget(self.list_view)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        content_layout.addWidget(separator)

        input_layout = QHBoxLayout()
        self.input_box = QTextEdit()
        self.input_box.setPlaceholderText("输入消息或股票代码...")
        self.input_box.setFixedHeight(60)
        self.input_box.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #dedede;
                border-radius: 10px;
                padding: 5px;
                font-size: 14px;
            }
        """)
        input_layout.addWidget(self.input_box)

        self.send_button = QPushButton("发送")
        self.send_button.setFixedWidth(80)
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004080;
            }
        """)
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)

        self.analyze_button = QPushButton("分析股票")
        self.analyze_button.setFixedWidth(100)
        self.analyze_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
        """)
        self.analyze_button.clicked.connect(self.analyze_stock)
        input_layout.addWidget(self.analyze_button)

        content_layout.addLayout(input_layout)

        self.layout.addWidget(content_widget)

        self.conversation = [{"role": "system",
                              "content": "你是一位金融专家，专门解答与金融相关的问题。请提供专业、准确的金融建议和信息，避免提及任何开发背景、技术细节或平台信息，以确保专注于解决用户的金融问题。"}]
        self.responseReady.connect(self.display_response)

        self.loading_message_index = None

        self.loading_texts = ["加载中", "加载中.", "加载中..", "加载中..."]
        self.current_loading_index = 0
        self.loading_timer = QTimer(self)
        self.loading_timer.setInterval(500)
        self.loading_timer.timeout.connect(self.update_loading_message)

        self.dragging = False
        self.drag_position = QPoint()

    def open_settings(self):
        avatar_choices = ["用户头像", "助手头像", "背景图片"]
        selected_avatar_type, ok = QInputDialog.getItem(self, "选择类型", "请选择您要更改的类型:", avatar_choices, 0, False)

        if ok and selected_avatar_type:
            selected_file, _ = QFileDialog.getOpenFileName(
                self, "选择图片", "", "Images (*.png *.xpm *.jpg *.bmp *.gif)")
            if selected_file:
                if selected_avatar_type == "用户头像":
                    self.user_avatar_path = selected_file
                    self.update_avatar(True)
                elif selected_avatar_type == "助手头像":
                    self.assistant_avatar_path = selected_file
                    self.update_avatar(False)
                elif selected_avatar_type == "背景图片":
                    self.set_background_image(selected_file)

    def update_avatar(self, is_user):
        for message in self.model.messages:
            if message.is_sent == is_user:
                message.avatar = QPixmap(self.user_avatar_path if is_user else self.assistant_avatar_path)
        self.model.layoutChanged.emit()

    def set_background_image(self, image_path):
        self.background_image = image_path
        self.setStyleSheet(f"""
            QWidget {{
                background-image: url("{self.background_image}");
                background-repeat: no-repeat;
                background-position: center;
                background-color: rgba(0, 0, 0, 100);
            }}
        """)

    def send_message(self):
        message = self.input_box.toPlainText().strip()

        if message:
            new_message = MessageItem(message, sender="我", is_sent=True)
            new_message.avatar = QPixmap(self.user_avatar_path)
            self.model.add_message(new_message)
            self.input_box.clear()
            self.list_view.scrollToBottom()

            self.conversation.append({"role": "user", "content": message})

            self.send_to_llm(message)
        else:
            print("消息为空，无法发送")

    def analyze_stock(self):
        stock_code, ok = QInputDialog.getText(self, "股票分析", "请输入股票代码（例如：AAPL）：")
        if ok and stock_code:
            threading.Thread(target=self.stock_analysis, args=(stock_code.strip().upper(),), daemon=True).start()
            
    def stock_analysis(self,stock_code):
        from analyze.analyze_stock import analyze_stock,fetch_stock
        hist=fetch_stock(stock_code)
        # 如果没有找到股票数据，返回提示
        if hist.empty:
            self.responseReady.emit(f"未找到股票代码：{stock_code}")
            return
        self.responseReady.emit("正在训练模型中，请稍候...")
        report=analyze_stock(stock_code,hist)

        if not report:
            self.responseReady.emit("分析出错，请稍后重试。")
            return
        image = QPixmap()
        if not image.load(f'{os.path.dirname(__file__)}\\{stock_code}_prediction.png'):
            self.responseReady.emit("图像加载失败。")


        # 显示图像
        image_message = MessageItem(sender='助手', is_sent=False, image=image)
        image_message.avatar = QPixmap(self.assistant_avatar_path)
        self.model.add_message(image_message)
        self.list_view.scrollToBottom()

        self.responseReady.emit(report)
        self.responseReady.emit("分析中，请稍候...")
        self.conversation.append({"role": "user", "content": report})
        self.get_response(report)

    def send_to_llm(self, message):
        self.responseReady.emit("正在获取回复，请稍候...")

        self.loading_message_index = self.model.rowCount() - 1

        self.current_loading_index = 0

        self.loading_timer.start()

        threading.Thread(target=self.get_response, args=(message,), daemon=True).start()

    def get_response(self, message=None):
        try:
            url = "https://api.lingyiwanwu.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}"
            }
            payload = {
                "model": "yi-lightning",
                "messages": self.conversation,
                "temperature": 0.9
            }

            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()

            result = response.json()
            assistant_message = result['choices'][0]['message']['content'].strip()
            self.conversation.append({"role": "assistant", "content": assistant_message})
            self.responseReady.emit(assistant_message)
        except Exception as e:
            print(f"获取回复时发生错误：{e}")
            self.responseReady.emit("抱歉，获取回复时出错。")

    def display_response(self, response):
        self.loading_timer.stop()

        new_message = MessageItem(response, sender="助手", is_sent=False)
        new_message.avatar = QPixmap(self.assistant_avatar_path)

        if self.loading_message_index is not None:
            self.model.update_message(self.loading_message_index, new_message)
            self.loading_message_index = None
        else:
            self.model.add_message(new_message)

        self.list_view.scrollToBottom()

    def update_loading_message(self):
        if self.loading_message_index is not None:
            new_text = self.loading_texts[self.current_loading_index]
            loading_message = MessageItem(new_text, sender="助手", is_sent=False)
            self.model.update_message(self.loading_message_index, loading_message)
            self.current_loading_index = (self.current_loading_index + 1) % len(self.loading_texts)
            self.list_view.scrollToBottom()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self.dragging:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        self.dragging = False

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = TelegramLikeChat()
    window.show()
    sys.exit(app.exec())