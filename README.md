
## 项目概述

该项目是一个基于PySide6的桌面应用程序，模拟了一个类似Telegram的聊天界面，具备实时对话和股票分析功能。用户可以通过输入消息与智能助手进行对话，并可以请求对特定股票的分析。该应用程序集成了自然语言处理（通过外部API）和机器学习模型（LSTM）来进行股票价格预测和指标分析。

## 主要功能

1. **实时聊天功能**：
    - 用户可以与智能助手进行对话。
    - 对话内容通过HTTP请求发送到外部API（`yi-lightning`模型），并接收回复。
    - 支持显示加载中动画，增强用户体验。
2. **股票分析功能**：
    - 用户可以输入股票代码，应用程序会获取该股票的历史数据并进行分析。
    - 使用LSTM模型对股票价格进行预测，并生成技术指标分析报告。
    - 预测结果以图表形式展示，并通过QPixmap在聊天界面中显示。
3. **用户界面**：
    - 采用PySide6构建图形用户界面，支持自定义头像和背景图片。
    - 聊天消息以气泡形式展示，支持文本和图片消息。
    - 界面支持拖动，增强用户交互体验。

![image](https://github.com/user-attachments/assets/7badb6e2-a0f0-4fc6-8b3c-2a1f02e09223)

![image](https://github.com/user-attachments/assets/69aa643b-6c8b-46bf-a77a-17ea34a6bc63)

![image](https://github.com/user-attachments/assets/b7cf1e1b-6fae-4290-82dd-7361b229639b)

![image](https://github.com/user-attachments/assets/e62935d4-5a9c-4566-b2b4-74c02e4127ec)

---

## 技术细节

### 1. 聊天功能实现

- **MessageItem类**：
    - 用于存储单条消息的内容、发送者、是否发送、头像和图片信息。
- **MessageModel类**：
    - 继承自QAbstractListModel，用于存储和管理聊天消息列表。
    - 提供添加消息、更新消息等方法。
- **MessageDelegate类**：
    - 继承自QStyledItemDelegate，用于自定义消息气泡的绘制。
    - 支持绘制文本消息和图片消息，并根据发送者显示不同的气泡颜色和头像。
- **TelegramLikeChat类**：
    - 主窗口类，包含聊天界面和输入框。
    - 通过QTimer实现加载中动画，通过QThread实现多线程消息发送和股票分析。

### 2. 股票分析功能实现

- **数据获取**：
    - 使用yfinance库获取股票历史数据。
    - 对缺失数据进行处理，确保数据完整性。
- **技术指标计算**：
    - 计算移动平均线（MA）、相对强弱指数（RSI）、MACD等技术指标。
    - 使用pandas和numpy进行数据处理和分析。
- **LSTM模型**：
    - 使用Keras构建LSTM模型，对股票价格进行预测。
    - 数据标准化处理，使用MinMaxScaler对特征进行缩放。
    - 创建序列数据，训练LSTM模型，并进行价格预测。
- **结果展示**：
    - 使用matplotlib绘制实际价格和预测价格的折线图。
    - 生成技术分析报告，并通过QPixmap在聊天界面中显示。

### 3. 用户界面实现

- **自定义头像和背景**：
    - 支持用户自定义头像和背景图片，通过QFileDialog选择图片文件。
    - 更新头像和背景图片时，实时刷新界面。
- **界面样式**：
    - 使用QSS样式表自定义窗口和控件的样式，包括颜色、边框、背景等。

## 代码结构

```
.
├── main.py
    ├── MessageItem类
    ├── MessageModel类
    ├── MessageDelegate类
    ├── TelegramLikeChat类
    ├── 主程序入口

```

### 主要类和方法

- **MessageItem类**：
    - `__init__`：初始化消息内容、发送者、是否发送、头像和图片信息。
- **MessageModel类**：
    - `__init__`：初始化消息列表。
    - `rowCount`：返回消息数量。
    - `data`：返回指定索引的消息。
    - `add_message`：添加新消息。
    - `update_message`：更新现有消息。
- **MessageDelegate类**：
    - `paint`：绘制单条消息气泡。
    - `sizeHint`：返回消息气泡的大小。
- **TelegramLikeChat类**：
    - `__init__`：初始化主窗口，设置界面布局和控件。
    - `open_settings`：打开设置对话框，选择并更新头像和背景图片。
    - `send_message`：发送用户消息，并请求智能助手回复。
    - `analyze_stock`：分析用户输入的股票代码，获取历史数据并进行分析和预测。
    - `fetch_and_analyze_stock`：获取股票数据，计算技术指标，训练LSTM模型，并生成预测结果和图表。
    - `send_to_llm`：发送消息到外部API，获取智能助手回复。
    - `get_response`：处理API回复，更新聊天界面。
    - `display_response`：显示智能助手回复。
    - `update_loading_message`：更新加载中动画。
    - `mousePressEvent`、`mouseMoveEvent`、`mouseReleaseEvent`：实现窗口拖动。

## 依赖库

- **PySide6**：用于构建图形用户界面。
- **requests**：用于发送HTTP请求，获取智能助手回复。
- **yfinance**：用于获取股票历史数据。
- **matplotlib**：用于绘制股票价格预测图表。
- **numpy**、**pandas**：用于数据处理和分析。
- **tensorflow**：用于构建和训练LSTM模型。

## 运行说明

1. 安装依赖库：
    
    ```bash
    pip install PySide6 requests yfinance matplotlib numpy pandas tensorflow
    ```
    
2. 运行主程序：
    
    ```bash
    python main.py
    ```
    
3. 输入股票代码或消息，点击发送按钮，查看智能助手回复和股票分析结果。

### 未来改进方向

1. **多语言支持**：增加多语言支持，提升国际化水平。
2. **更多技术指标**：增加更多技术指标，提升股票分析的全面性。
3. **模型优化**：优化LSTM模型参数，提升预测准确性。
4. **用户体验**：增加更多交互功能，提升用户体验。

[中期报告](https://www.notion.so/14db8ee5cbfa8068b4c8fa308239e294?pvs=21)
