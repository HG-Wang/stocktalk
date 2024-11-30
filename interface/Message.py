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