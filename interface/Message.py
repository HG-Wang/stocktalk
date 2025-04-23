from PySide6.QtCore import Qt, QAbstractListModel, QModelIndex, QSize, QRectF, QEvent
from PySide6.QtGui import QColor, QPainter, QFont, QPixmap, QTextDocument, QPen, QBrush, QLinearGradient, QPainterPath, QCursor
from PySide6.QtWidgets import QStyledItemDelegate, QApplication

# 添加 Markdown 支持
try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
    print("警告: Python-Markdown 库未安装，Markdown 格式将不可用")

# 修正导入语句
import sys
import os

# 添加父目录到sys.path，确保可以找到utils模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 直接导入utils模块
from utils import ImageViewer

class MessageItem:
    def __init__(self, message='', sender="我", is_sent=True, image=None):
        self.message = message
        self.sender = sender
        self.is_sent = is_sent
        self.avatar = QPixmap('user_avatar.png') if is_sent else QPixmap('AI.png')
        self.image = image

class MessageDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.padding = 12  # 内边距
        self.bubble_max_width = 300  # 减小最大宽度以确保不超出
        self.avatar_size = 45  # 头像尺寸
        self.avatar_margin = 12  # 增加头像与气泡间距
        self.parent_view = parent
        self.image_rect = None  # 存储当前图片区域
        self.current_image = None  # 存储当前显示的图片
        
        # 安装事件过滤器以支持鼠标点击
        if self.parent_view:
            self.parent_view.viewport().installEventFilter(self)
            self.parent_view.viewport().setCursor(Qt.ArrowCursor)

    def paint(self, painter, option, index):
        message_item = index.model().data(index, Qt.DisplayRole)

        if message_item:
            painter.save()
            painter.setRenderHint(QPainter.Antialiasing)

            # 确定显示位置和颜色
            if message_item.sender == "我":
                # 用户发送的消息使用渐变蓝色
                gradient = QLinearGradient(0, 0, 1, 0)
                gradient.setCoordinateMode(QLinearGradient.ObjectBoundingMode)
                gradient.setColorAt(0, QColor(41, 128, 185))  # 蓝色起始
                gradient.setColorAt(1, QColor(52, 152, 219))  # 蓝色结束
                bubble_brush = QBrush(gradient)
                text_color = QColor(255, 255, 255)  # 白色文本
                align = Qt.AlignRight
                avatar = message_item.avatar.scaled(
                    self.avatar_size, self.avatar_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                avatar_x = option.rect.right() - self.avatar_size - 10
            else:
                # 助手发送的消息使用深灰背景
                bubble_brush = QBrush(QColor(44, 62, 80))  # 深灰色背景
                text_color = QColor(255, 255, 255)  # 白色文本
                align = Qt.AlignLeft
                avatar = message_item.avatar.scaled(
                    self.avatar_size, self.avatar_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                avatar_x = option.rect.left() + 10

            # 计算内容尺寸
            if message_item.image:
                self.current_image = message_item.image  # 保存当前图片
                image = message_item.image.scaledToWidth(self.bubble_max_width - 2 * self.padding, Qt.SmoothTransformation)
                image_width = image.width()
                image_height = image.height()

                bubble_width = image_width + 2 * self.padding
                bubble_height = image_height + 2 * self.padding
            else:
                self.current_image = None
                font = QFont("Microsoft YaHei", 10)
                text_document = QTextDocument()
                text_document.setDefaultFont(font)
                
                # 设置文本颜色为白色
                text_document.setDefaultStyleSheet(f"body {{ color: white; }}")
                
                # Markdown 渲染支持
                if MARKDOWN_AVAILABLE:
                    # 使用自定义样式包装 Markdown 内容，强制使用白色文本
                    styled_content = f"""
                    <style>
                    body, p, h1, h2, h3, h4, h5, h6, ul, ol, li, blockquote, pre, code {{
                        color: white !important;
                    }}
                    code {{
                        background-color: #2c3e50 !important;
                        padding: 2px 4px;
                        border-radius: 3px;
                    }}
                    pre {{
                        background-color: #2c3e50 !important;
                        padding: 8px;
                        border-radius: 5px;
                        overflow-x: auto;
                    }}
                    blockquote {{
                        border-left: 4px solid #3498db;
                        padding-left: 10px;
                        margin-left: 0;
                        color: #ecf0f1 !important;
                    }}
                    a {{
                        color: #3498db !important;
                        text-decoration: none;
                    }}
                    a:hover {{
                        text-decoration: underline;
                    }}
                    table {{
                        border-collapse: collapse;
                    }}
                    th, td {{
                        border: 2px solid white;
                        padding: 6px;
                    }}
                    th {{
                        background-color: #34495e;
                    }}
                    </style>
                    """ + markdown.markdown(message_item.message)
                    text_document.setHtml(styled_content)
                else:
                    text_document.setHtml(f"<div style='color: white;'>{message_item.message}</div>")
                
                # 限制文本宽度，确保不会超出气泡
                available_width = self.bubble_max_width - 2 * self.padding
                text_document.setTextWidth(available_width)

                text_width = text_document.idealWidth()
                text_height = text_document.size().height()

                # 确保气泡宽度不超过最大宽度
                bubble_width = min(max(text_width + 2 * self.padding, 80), self.bubble_max_width)
                bubble_height = text_height + 2 * self.padding

            # 计算气泡位置，确保有足够空间
            if align == Qt.AlignRight:
                x = option.rect.right() - bubble_width - self.padding - self.avatar_size - self.avatar_margin
            else:
                x = option.rect.left() + self.avatar_size + self.avatar_margin + 5

            y = option.rect.top() + self.padding
            bubble_rect = QRectF(x, y, bubble_width, bubble_height)

            # 绘制阴影效果
            shadow_color = QColor(0, 0, 0, 50)  # 半透明黑色阴影
            shadow_offset = 3  # 阴影偏移
            shadow_radius = 12  # 阴影圆角
            shadow_rect = QRectF(
                bubble_rect.x() + shadow_offset / 2,
                bubble_rect.y() + shadow_offset / 2,
                bubble_rect.width(),
                bubble_rect.height()
            )
            painter.setBrush(shadow_color)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(shadow_rect, shadow_radius, shadow_radius)

            # 绘制气泡
            painter.setBrush(bubble_brush)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(bubble_rect, 12, 12)
            
            # 绘制内容
            content_rect = QRectF(
                bubble_rect.left() + self.padding,
                bubble_rect.top() + self.padding,
                bubble_width - 2 * self.padding,
                bubble_height - 2 * self.padding
            )

            if message_item.image:
                # 绘制图片
                painter.drawPixmap(content_rect.toRect(), image)
                
                # 保存图片区域信息，用于点击检测
                self.image_rect = content_rect.toRect()
                
                # 添加固定的图片点击提示，不再使用悬停效果
                zoom_hint = "点击查看大图"
                font = QFont("Microsoft YaHei", 9)
                painter.setFont(font)
                
                # 创建半透明背景
                hint_rect = QRectF(content_rect.left(), content_rect.bottom() - 25, content_rect.width(), 25)
                painter.setBrush(QColor(0, 0, 0, 120))
                painter.setPen(Qt.NoPen)
                painter.drawRect(hint_rect)
                
                painter.setPen(Qt.white)
                painter.drawText(hint_rect, Qt.AlignCenter, zoom_hint)
                
            else:
                # 绘制文本，确保文本在气泡内部
                painter.translate(content_rect.topLeft())
                text_document.drawContents(painter)
                painter.translate(-content_rect.topLeft())

            # 绘制头像，使用圆形蒙版
            if align == Qt.AlignRight:
                avatar_y = y  # 头像位于气泡顶部
            else:
                avatar_y = y  # 头像位于气泡顶部
                
            # 绘制带边框的圆形头像
            avatar_rect = QRectF(avatar_x, avatar_y, self.avatar_size, self.avatar_size)
            painter.setClipRect(option.rect)
            
            # 创建圆形剪裁区域
            path = QPainterPath()
            path.addEllipse(avatar_rect)
            painter.setClipPath(path)
            
            # 绘制头像
            painter.drawPixmap(avatar_rect.toRect(), avatar)
            
            # 添加细边框
            painter.setClipping(False)
            pen = QPen(QColor(22, 36, 50), 1.5)  # 深色边框
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(avatar_rect)

            painter.restore()

    def sizeHint(self, option, index):
        message_item = index.model().data(index, Qt.DisplayRole)

        if message_item:
            if message_item.image:
                image = message_item.image.scaledToWidth(self.bubble_max_width - 2 * self.padding, Qt.SmoothTransformation)
                bubble_height = image.height() + 2 * self.padding
                total_height = max(bubble_height, self.avatar_size) + 2 * self.padding
                return QSize(option.rect.width(), total_height)
            else:
                font = QFont("Microsoft YaHei", 10)
                text_document = QTextDocument()
                text_document.setDefaultFont(font)
                # 使用白色文本
                text_document.setDefaultStyleSheet("body { color: white; }")
                
                # 检查是否支持 Markdown
                if MARKDOWN_AVAILABLE:
                    # 使用自定义样式包装 Markdown 内容，确保文本为白色
                    styled_content = f"""
                    <style>
                    body, p, h1, h2, h3, h4, h5, h6, ul, ol, li, blockquote, pre, code {{
                        color: white !important;
                    }}
                    code {{
                        background-color: #2c3e50 !important;
                        padding: 2px 4px;
                        border-radius: 3px;
                    }}
                    pre {{
                        background-color: #2c3e50 !important;
                        padding: 8px;
                        border-radius: 5px;
                        overflow-x: auto;
                    }}
                    blockquote {{
                        border-left: 4px solid #3498db;
                        padding-left: 10px;
                        margin-left: 0;
                        color: #ecf0f1 !important;
                    }}
                    a {{
                        color: #3498db !important;
                        text-decoration: none;
                    }}
                    a:hover {{
                        text-decoration: underline;
                    }}
                    table {{
                        border-collapse: collapse;
                    }}
                    th, td {{
                        border: 1px solid #34495e;
                        padding: 6px;
                    }}
                    th {{
                        background-color: #34495e;
                    }}
                    </style>
                    """ + markdown.markdown(message_item.message)
                    html_content = styled_content
                else:
                    html_content = f"<div style='color: white;'>{message_item.message}</div>"
                
                text_document.setHtml(html_content)
                text_document.setTextWidth(self.bubble_max_width - 2 * self.padding)

                text_height = text_document.size().height()
                total_height = max(text_height + 2 * self.padding, self.avatar_size) + 2 * self.padding
                return QSize(option.rect.width(), total_height)

        return super().sizeHint(option, index)
        
    def eventFilter(self, obj, event):
        """处理鼠标事件，简化后只处理点击事件"""
        if obj == self.parent_view.viewport():
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                # 获取鼠标点击位置
                pos = event.pos()
                
                # 检查是否点击在图片区域内
                if self.image_rect and self.image_rect.contains(pos) and self.current_image:
                    print("图片被点击 - 准备打开查看器")
                    # 直接创建并显示图片查看器
                    try:
                        from utils import ImageViewer
                        viewer = ImageViewer(self.current_image, self.parent_view)
                        viewer.exec()
                        print("查看器关闭")
                    except Exception as e:
                        print(f"显示图片查看器时出错: {e}")
                    return True
                    
        return super().eventFilter(obj, event)

class MessageModel(QAbstractListModel):
    def __init__(self, messages=None):
        super().__init__()
        self.messages = messages if messages else []

    def rowCount(self, parent=QModelIndex()):
        return len(self.messages)

    def data(self, index, role):
        if role == Qt.DisplayRole:
            return self.messages[index.row()]
        return None

    def add_message(self, message_item):
        self.beginInsertRows(QModelIndex(), len(self.messages), len(self.messages))
        self.messages.append(message_item)
        self.endInsertRows()

    def update_message(self, index, new_message_item):
        if 0 <= index < len(self.messages):
            self.messages[index] = new_message_item
            self.dataChanged.emit(self.index(index), self.index(index))