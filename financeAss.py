import sys
import threading
import time
import datetime
import os

import requests
import yfinance as yf
import matplotlib
matplotlib.use('Agg')

import numpy as np
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
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
    QInputDialog,
    QDialog,
    QScrollArea,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem
)

from interface.Message import MessageDelegate,MessageItem,MessageModel
from utils import ImageViewer  # 从utils模块导入ImageViewer类

# 设置字体为支持中文的字体，例如 SimSun（宋体）、SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False    # 解决坐标轴负号显示问题
plt.rcParams['figure.facecolor'] = '#f8f9fa'  # 设置图表背景色
plt.rcParams['axes.facecolor'] = '#ffffff'    # 设置坐标区背景色
plt.rcParams['axes.grid'] = True              # 默认显示网格
plt.rcParams['grid.alpha'] = 0.3              # 网格透明度
plt.rcParams['grid.linestyle'] = '--'         # 网格线样式

API_KEY = '95391dfcc5194055a072f0662c7e69ab'

def plot_stock_prediction(hist, forecast, actual, stock_code, analysis_mode='deep'):
    """
    绘制增强版股票价格预测图(AutoTS版本)。

    参数:
    - hist: 股票历史数据
    - forecast: AutoTS预测结果DataFrame，包含'Close'、'Close_lower'、'Close_upper'列
    - actual: 实际的股票收盘价
    - stock_code: 股票代码，用于图表标题和保存图像
    - analysis_mode: 分析模式，'deep'为深度分析，'fast'为快速分析
    """
    # 创建子图布局
    fig = plt.figure(figsize=(16, 12), dpi=100)
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])
    
    # 使用数字索引而不是日期索引，避免时区问题
    # 从forecast中提取预测值和置信区间
    predicted_prices = forecast.get('Close', pd.Series([])).values if analysis_mode == 'deep' else []
    
    # 动态匹配置信区间列名
    upper_cols = [col for col in forecast.columns if 'upper' in col.lower() or '0.95' in col] if analysis_mode == 'deep' and not forecast.empty else []
    lower_cols = [col for col in forecast.columns if 'lower' in col.lower() or '0.05' in col] if analysis_mode == 'deep' and not forecast.empty else []
    
    if upper_cols and lower_cols:
        conf_interval = (forecast[upper_cols[0]].values - forecast[lower_cols[0]].values) / 2
    else:
        # 使用历史波动率作为默认置信区间
        rolling_std = hist['Close'].rolling(30).std().iloc[-len(forecast):].values if analysis_mode == 'deep' else []
        conf_interval = 1.96 * rolling_std if analysis_mode == 'deep' else []
    
    num_points = len(predicted_prices) if analysis_mode == 'deep' else 0
    date_range = forecast.index if analysis_mode == 'deep' else []
    pred_error = actual - predicted_prices if analysis_mode == 'deep' and len(actual) == len(predicted_prices) else np.array([])
    mape = np.mean(np.abs(pred_error / actual)) * 100 if analysis_mode == 'deep' and len(actual) > 0 else 0.0
    rmse = np.sqrt(np.mean(pred_error**2)) if analysis_mode == 'deep' and len(pred_error) > 0 else 0.0
    
    # 计算预测趋势和精度
    trend_actual = np.mean(actual[-5:]) - np.mean(actual[-10:-5]) if analysis_mode == 'deep' and len(actual) >= 10 else 0
    trend_pred = np.mean(predicted_prices[-5:]) - np.mean(predicted_prices[-10:-5]) if analysis_mode == 'deep' and len(predicted_prices) >= 10 else 0
    trend_correct = (trend_actual * trend_pred) > 0 if analysis_mode == 'deep' else False  # 趋势方向是否一致
    
    # 计算预测准确度评级
    if analysis_mode == 'deep':
        if mape < 1:
            accuracy_rating = "极高"
        elif mape < 3:
            accuracy_rating = "高"
        elif mape < 5:
            accuracy_rating = "中等"
        elif mape < 10:
            accuracy_rating = "低"
        else:
            accuracy_rating = "极低"
    else:
        accuracy_rating = "N/A"
        
    # 准备将来的日期用于未来预测显示
    last_date = date_range[-1] if analysis_mode == 'deep' and len(date_range) > 0 else hist.index[-1]
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 31)]
    
    # 计算未来预测值（这里简单使用当前趋势延伸）
    last_pred = predicted_prices[-1] if analysis_mode == 'deep' and len(predicted_prices) > 0 else hist['Close'].iloc[-1]
    future_trend = predicted_prices[-1] - predicted_prices[-5] if analysis_mode == 'deep' and len(predicted_prices) >= 5 else 0
    daily_change = future_trend / 5 if analysis_mode == 'deep' else 0
    future_preds = [last_pred + daily_change * i for i in range(1, 31)] if analysis_mode == 'deep' else []
    
    # ---------------------------------------------
    # 价格和预测子图
    # ---------------------------------------------
    ax1 = plt.subplot(gs[0])
    
    # 绘制实际价格 - 显示全部历史数据，并添加日志以验证数据
    print(f"绘图函数接收到的历史数据长度: {len(hist.index)}")
    print(f"绘图函数接收到的实际价格数据长度: {len(actual)}")
    ax1.plot(hist.index, hist['Close'].values, label='实际价格', color='#1f77b4', linewidth=2, marker='o',
             markersize=3, markerfacecolor='white')
    
    if analysis_mode == 'deep' and len(predicted_prices) > 0:
        # 绘制预测价格
        ax1.plot(forecast.index, predicted_prices, label='预测价格', color='#ff7f0e',
                 linestyle='--', linewidth=2)
        
        # 绘制未来预测（虚线延伸）
        ax1.plot(future_dates, future_preds, label='未来预测', color='#2ca02c',
                 linestyle='-.', linewidth=2)
        
        # 绘制置信区间（如果存在）
        upper_cols = [col for col in forecast.columns if 'upper' in col.lower() or '0.95' in col]
        lower_cols = [col for col in forecast.columns if 'lower' in col.lower() or '0.05' in col]
        
        if upper_cols and lower_cols:
            ax1.fill_between(forecast.index,
                            forecast[lower_cols[0]].values,
                            forecast[upper_cols[0]].values,
                            alpha=0.15, color='#ff7f0e', label='95%置信区间')
        else:
            # 如果没有置信区间列，使用滚动标准差作为替代
            rolling_std = hist['Close'].rolling(30).std().iloc[-len(forecast):].values
            conf_interval = 1.96 * rolling_std
            ax1.fill_between(forecast.index,
                            predicted_prices - conf_interval,
                            predicted_prices + conf_interval,
                            alpha=0.15, color='#ff7f0e', label='估计置信区间')
    
    # 计算并绘制关键支撑位和阻力位
    recent_high = np.max(hist['Close'].values[-30:]) if len(hist['Close'].values) >= 30 else np.max(hist['Close'].values)
    recent_low = np.min(hist['Close'].values[-30:]) if len(hist['Close'].values) >= 30 else np.min(hist['Close'].values)
    
    # 支撑位
    support = recent_low - recent_low * 0.01
    ax1.axhline(y=support, color='green', linestyle='-', alpha=0.7, linewidth=1)
    ax1.text(hist.index[0], support, f'支撑位: {support:.2f}',
             fontsize=10, color='green', verticalalignment='bottom')
    
    # 阻力位
    resistance = recent_high + recent_high * 0.01
    ax1.axhline(y=resistance, color='red', linestyle='-', alpha=0.7, linewidth=1)
    ax1.text(hist.index[0], resistance, f'阻力位: {resistance:.2f}',
             fontsize=10, color='red', verticalalignment='bottom')
    
    # 标记最大误差点（如果实际数据长度足够）
    if analysis_mode == 'deep' and len(actual) > 0 and len(pred_error) > 0:
        max_error_idx = np.argmax(np.abs(pred_error))
        if max_error_idx < len(forecast.index):
            ax1.scatter(forecast.index[max_error_idx], actual[max_error_idx],
                       s=100, color='red', marker='x', zorder=5)
    
    # 格式化x轴日期
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    
    # 设置标签和标题
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    title = f'{stock_code} 股票价格预测分析 (生成日期: {current_date})' if analysis_mode == 'deep' else f'{stock_code} 股票价格分析 (生成日期: {current_date})'
    ax1.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('')
    ax1.set_ylabel('价格', fontsize=12)
    
    # 添加网格
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 添加图例
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    if analysis_mode == 'deep':
        # 添加预测准确度文本框
        performance_text = (
            f"预测性能指标:\n"
            f"MAPE: {mape:.2f}%\n"
            f"RMSE: {rmse:.2f}\n"
            f"趋势准确性: {'✓' if trend_correct else '✗'}\n"
            f"整体评级: {accuracy_rating}"
        )
        
        # 为文本框创建一个透明的矩形
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
        ax1.text(0.02, 0.05, performance_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='bottom', bbox=props)
    
    # ---------------------------------------------
    # 移动平均线子图
    # ---------------------------------------------
    ax2 = plt.subplot(gs[1], sharex=ax1)
    
    # 绘制移动平均线
    ax2.plot(hist.index, hist['Close'].rolling(window=20).mean(), label='MA20', color='orange', linewidth=1.5)
    ax2.plot(hist.index, hist['Close'].rolling(window=50).mean(), label='MA50', color='green', linewidth=1.5)
    ax2.plot(hist.index, hist['Close'].rolling(window=200).mean(), label='MA200', color='red', linewidth=1.5)
    
    # 设置标签
    ax2.set_xlabel('')
    ax2.set_ylabel('移动平均线', fontsize=10)
    ax2.set_title('移动平均线分析', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # ---------------------------------------------
    # 成交量子图
    # ---------------------------------------------
    ax3 = plt.subplot(gs[2], sharex=ax1)
    
    # 提取成交量数据，使用完整历史数据
    volume_values = hist['Volume'].values
    
    # 准备颜色数据
    colors = []
    for i in range(len(hist.index)):
        if hist['Close'].iloc[i] >= hist['Open'].iloc[i]:
            colors.append('green')
        else:
            colors.append('red')
    
    # 绘制成交量柱状图
    ax3.bar(hist.index, volume_values, alpha=0.6, color=colors)
    
    # 计算并绘制成交量移动平均线 (使用numpy)
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w
    
    if len(volume_values) > 5:
        volume_ma = moving_average(volume_values, 5)
        # 由于移动平均会减少数据点，需要调整x轴
        ma_dates = hist.index[4:]
        ax3.plot(ma_dates, volume_ma, color='blue', linewidth=1.5, label='5日均量')
    
    # 设置标签
    ax3.set_xlabel('日期', fontsize=12)
    ax3.set_ylabel('成交量', fontsize=10)
    ax3.set_title('成交量分析', fontsize=12)
    ax3.legend(loc='upper left')
    
    # 旋转x轴标签
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # 添加水印
    fig.text(0.99, 0.01, 'StockTalk AI', fontsize=12, color='gray',
             ha='right', va='bottom', alpha=0.5)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    # 保存图像到本地，提高DPI以获得更清晰的图像
    save_path = os.path.join(os.path.dirname(__file__), f'{stock_code}_prediction.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


class TelegramLikeChat(QWidget):
    responseReady = Signal(str)

    def __init__(self):
        super().__init__()

        # 使用绝对路径获取头像文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.user_avatar_path = os.path.join(current_dir, 'user_avatar.png')
        self.assistant_avatar_path = os.path.join(current_dir, 'AI.png')  # 使用AI.png作为助手头像

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setMinimumSize(400, 650)  # 增加默认窗口大小

        self.background_image = None
        self.setObjectName("main_widget")
        self.setStyleSheet("""
            QWidget#main_widget {
                background-color: #121826;
                border-radius: 14px;
                border: 1px solid #2c3e50;
            }
            QFrame#title_bar {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2980b9, stop:1 #3498db);
                border-top-left-radius: 14px;
                border-top-right-radius: 14px;
            }
            QLabel#title {
                color: #ffffff;
                font-size: 17px;
                font-weight: bold;
                font-family: "Microsoft YaHei";
            }
            QToolButton {
                background-color: transparent;
                color: #ffffff;
                font-size: 18px;
            }
            QToolButton:hover {
                color: #ecf0f1;
            }
            QScrollBar:vertical {
                background: #2c3e50;
                width: 10px;
                margin: 0px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #3498db;
                min-height: 30px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #2980b9;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        title_bar = QFrame()
        title_bar.setObjectName("title_bar")
        title_bar.setFixedHeight(50)  # 增加标题栏高度
        title_bar.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2980b9, stop:1 #3498db);
            border-top-left-radius: 14px;
            border-top-right-radius: 14px;
        """)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(15, 0, 15, 0)

        title = QLabel("实时交易信号智能对话助手")
        title.setObjectName("title")
        title_font = QFont("Microsoft YaHei", 17, QFont.Bold)
        title.setFont(title_font)
        title.setStyleSheet("color: white; letter-spacing: 1px;")
        title.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        title_layout.addWidget(title)

        title_layout.addStretch()

        settings_button = QToolButton()
        settings_button.setText("⚙️")
        settings_button.setStyleSheet("""
            QToolButton {
                color: white;
                border: none;
                font-size: 18px;
                padding: 5px;
            }
            QToolButton:hover {
                background-color: rgba(255, 255, 255, 0.15);
                border-radius: 5px;
            }
        """)
        settings_button.clicked.connect(self.open_settings)
        title_layout.addWidget(settings_button)

        self.layout.addWidget(title_bar)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(15, 15, 15, 15)
        content_layout.setSpacing(15)

        self.model = MessageModel()
        self.list_view = QListView()
        self.list_view.setModel(self.model)
        self.list_view.setItemDelegate(MessageDelegate(self.list_view))
        self.list_view.setWordWrap(True)

        self.list_view.setStyleSheet("""
            QListView {
                background-color: #1a2635;
                border: 1px solid #2c3e50;
                border-radius: 10px;
                padding: 8px;
            }
            QListView::item {
                border: none;
                padding: 2px;
                margin-bottom: 4px;
            }
            QListView::item:hover {
                background: transparent;
            }
        """)
        self.list_view.setVerticalScrollMode(QListView.ScrollPerPixel)
        self.list_view.setHorizontalScrollMode(QListView.ScrollPerPixel)
        content_layout.addWidget(self.list_view)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #2c3e50; min-height: 1px;")
        content_layout.addWidget(separator)

        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)  # 增加间距
        
        self.input_box = QTextEdit()
        self.input_box.setPlaceholderText("输入消息或股票代码...")
        self.input_box.setFixedHeight(60)
        self.input_box.setStyleSheet("""
            QTextEdit {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #34495e;
                border-radius: 10px;
                padding: 10px;
                font-size: 14px;
                font-family: "Microsoft YaHei";
            }
            QTextEdit:focus {
                border: 1px solid #3498db;
            }
        """)
        input_layout.addWidget(self.input_box)

        self.send_button = QPushButton("发送")
        self.send_button.setFixedWidth(80)
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 15px;
                font-weight: bold;
                padding: 8px 12px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1f6dad;
            }
        """)
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
    
        self.analyze_button = QPushButton("分析股票")
        self.analyze_button.setFixedWidth(100)
        self.analyze_button.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 15px;
                font-weight: bold;
                padding: 8px 12px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
            QPushButton:pressed {
                background-color: #713688;
            }
        """)
        self.analyze_button.clicked.connect(self.analyze_stock)
        input_layout.addWidget(self.analyze_button)
    
        # 添加分析模式选择下拉菜单
        from PySide6.QtWidgets import QComboBox
        self.mode_selector = QComboBox()
        self.mode_selector.addItem("深度分析", "deep")
        self.mode_selector.addItem("快速分析", "fast")
        self.mode_selector.setFixedWidth(120)
        self.mode_selector.setStyleSheet("""
            QComboBox {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #34495e;
                border-radius: 10px;
                padding: 5px;
                font-size: 14px;
                font-family: "Microsoft YaHei";
            }
            QComboBox:hover {
                border: 1px solid #3498db;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 25px;
                border-left-width: 1px;
                border-left-color: #34495e;
                border-left-style: solid;
                border-top-right-radius: 10px;
                border-bottom-right-radius: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #2c3e50;
                color: #ecf0f1;
                selection-background-color: #3498db;
                selection-color: white;
                border: 1px solid #34495e;
            }
        """)
        input_layout.addWidget(self.mode_selector)

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
        
        # 添加欢迎消息
        welcome_message = MessageItem("欢迎使用金融智能助手！我可以帮您解答金融问题或分析股票。请输入问题或点击分析股票按钮开始。", 
                                    sender="助手", is_sent=False)
        welcome_message.avatar = QPixmap(self.assistant_avatar_path)
        self.model.add_message(welcome_message)

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
            analysis_mode = self.mode_selector.currentData()
            threading.Thread(target=self.stock_analysis, args=(stock_code.strip().upper(), analysis_mode), daemon=True).start()
            
    def stock_analysis(self, stock_code, analysis_mode='deep'):
        from analyze.analyze_stock import analyze_stock, fetch_stock
        hist = fetch_stock(stock_code)
        # 如果没有找到股票数据，返回提示
        if hist.empty:
            self.responseReady.emit(f"未找到股票代码：{stock_code}")
            return
        self.responseReady.emit("正在训练模型中，请稍候...")
        report = analyze_stock(stock_code, hist, analysis_mode)
    
        if not report:
            self.responseReady.emit("分析出错，请稍后重试。")
            return
        image_path = os.path.join(os.path.dirname(__file__), f'{stock_code}_prediction.png')
        print(f"尝试加载图片路径: {image_path}")
        
        # 等待图片文件生成(最多等待5秒)
        max_attempts = 10
        for _ in range(max_attempts):
            if os.path.exists(image_path):
                break
            time.sleep(0.5)
        else:
            print(f"图片文件未生成: {image_path}")
            self.responseReady.emit("图像生成失败。")
            return
            
        print(f"文件存在: {os.path.exists(image_path)}")
        
        image = QPixmap()
        if not image.load(image_path):
            print(f"图片加载失败，路径: {image_path}")
            # 尝试重新加载一次
            if not image.load(image_path):
                self.responseReady.emit("图像加载失败。")
                return
    
    
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
            loading_message.avatar = QPixmap(self.assistant_avatar_path)  # 设置加载消息的头像
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