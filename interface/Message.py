from PySide6.QtCore import Qt, QAbstractListModel, QModelIndex, QSize, QRectF
from PySide6.QtGui import QColor, QPainter, QFont, QPixmap, QTextDocument
from PySide6.QtWidgets import QStyledItemDelegate
class MessageItem:
    def __init__(self, message='', sender="我", is_sent=True, image=None):
        self.message = message
        self.sender = sender
        self.is_sent = is_sent
        self.avatar = QPixmap('user_avatar.jpg') if is_sent else QPixmap('assistant_avatar.jpg')
        self.image = image

class MessageDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.padding = 10
        self.bubble_max_width = 300
        self.avatar_size = 40

    def paint(self, painter, option, index):
        message_item = index.model().data(index, Qt.DisplayRole)

        if message_item:
            painter.save()
            painter.setRenderHint(QPainter.Antialiasing)

            if message_item.sender == "我":
                bubble_color = QColor(0, 122, 204)
                text_color = QColor(0, 0, 0)
                align = Qt.AlignRight
                avatar = message_item.avatar.scaled(
                    self.avatar_size, self.avatar_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                avatar_x = option.rect.right() - self.avatar_size - 10
            else:
                bubble_color = QColor(240, 240, 240)
                text_color = QColor(0, 0, 0)
                align = Qt.AlignLeft
                avatar = message_item.avatar.scaled(
                    self.avatar_size, self.avatar_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                avatar_x = option.rect.left() + 10

            if message_item.image:
                image = message_item.image.scaledToWidth(self.bubble_max_width, Qt.SmoothTransformation)
                image_width = image.width()
                image_height = image.height()

                bubble_width = image_width + 2 * self.padding
                bubble_height = image_height + 2 * self.padding
            else:
                font = QFont("Segoe UI", 10)
                text_document = QTextDocument()
                text_document.setDefaultFont(font)
                text_document.setDefaultStyleSheet(f"body {{ color: {text_color.name()}; }}")
                text_document.setMarkdown(message_item.message)
                text_document.setTextWidth(self.bubble_max_width)

                text_width = text_document.idealWidth()
                text_height = text_document.size().height()

                bubble_width = text_width + 2 * self.padding
                bubble_height = text_height + 2 * self.padding

            if align == Qt.AlignRight:
                x = option.rect.right() - bubble_width - self.padding - self.avatar_size - 15
            else:
                x = option.rect.left() + self.avatar_size + 20

            y = option.rect.top() + self.padding
            bubble_rect = QRectF(x, y, bubble_width, bubble_height)

            shadow_color = QColor(0, 0, 0, 30)
            shadow_offset = 2
            shadow_rect = QRectF(
                bubble_rect.x() + shadow_offset,
                bubble_rect.y() + shadow_offset,
                bubble_rect.width(),
                bubble_rect.height()
            )
            painter.setBrush(shadow_color)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(shadow_rect, 10, 10)

            painter.setBrush(bubble_color)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(bubble_rect, 10, 10)

            content_rect = QRectF(
                bubble_rect.left() + self.padding,
                bubble_rect.top() + self.padding,
                bubble_width - 2 * self.padding,
                bubble_height - 2 * self.padding
            )

            if message_item.image:
                # 绘制图片
                painter.drawPixmap(content_rect.toRect(), message_item.image)
            else:
                # 绘制文本
                painter.translate(content_rect.topLeft())
                text_document.drawContents(painter)
                painter.translate(-content_rect.topLeft())

            if align == Qt.AlignRight:
                avatar_y = y + bubble_height - self.avatar_size
            else:
                avatar_y = y

            painter.drawPixmap(avatar_x, avatar_y, self.avatar_size, self.avatar_size, avatar)

            painter.restore()

def sizeHint(self, option, index):
    message_item = index.model().data(index, Qt.DisplayRole)

    if message_item:
        if message_item.image:
            image = message_item.image.scaledToWidth(self.bubble_max_width, Qt.SmoothTransformation)
            bubble_height = image.height() + 2 * self.padding
            total_height = max(bubble_height, self.avatar_size) + self.padding
            return QSize(option.rect.width(), total_height)
        else:
            font = QFont("Segoe UI", 10)
            text_document = QTextDocument()
            text_document.setDefaultFont(font)
            text_document.setDefaultStyleSheet("")  # 使用默认样式
            text_document.setMarkdown(message_item.message)
            text_document.setTextWidth(self.bubble_max_width)

            text_height = text_document.size().height()
            total_height = max(text_height + 2 * self.padding, self.avatar_size) + self.padding
            return QSize(option.rect.width(), total_height)

    return super().sizeHint(option, index)
