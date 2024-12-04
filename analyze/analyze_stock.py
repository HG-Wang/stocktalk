import datetime

import yfinance as yf
import matplotlib
matplotlib.use('Agg')

import numpy as np
from financeAss import plot_stock_prediction

def fetch_stock(stock_code):
    # 获取股票历史数据（过去1年）
    stock = yf.Ticker(stock_code)
    hist = stock.history(period="1y")
    return hist


def analyze_stock(stock_code,hist):
    report=()
    try:
        # 导入需要的库
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        from sklearn.preprocessing import MinMaxScaler

        # 输出股票数据并提示模型训练中
        print(hist)

        # 计算更多技术指标
        hist['MA20'] = hist['Close'].rolling(window=20).mean()  # 20日均线
        hist['MA50'] = hist['Close'].rolling(window=50).mean()  # 50日均线
        hist['MA200'] = hist['Close'].rolling(window=200).mean()  # 200日均线
        hist['STD'] = hist['Close'].rolling(window=20).std()  # 20日标准差
        hist['UpperBB'] = hist['MA20'] + (hist['STD'] * 2)  # 布林带上轨
        hist['LowerBB'] = hist['MA20'] - (hist['STD'] * 2)  # 布林带下轨
        delta = hist['Close'].diff()  # 收盘价差异
        gain = delta.clip(lower=0)  # 收益
        loss = -delta.clip(upper=0)  # 损失
        average_gain = gain.rolling(window=14).mean()  # 平均收益
        average_loss = loss.rolling(window=14).mean()  # 平均损失
        rs = average_gain / average_loss  # 相对强度
        hist['RSI'] = 100 - (100 / (1 + rs))  # 相对强弱指数 (RSI)
        hist['EMA12'] = hist['Close'].ewm(span=12, adjust=False).mean()  # 12日指数移动平均线
        hist['EMA26'] = hist['Close'].ewm(span=26, adjust=False).mean()  # 26日指数移动平均线
        hist['MACD'] = hist['EMA12'] - hist['EMA26']  # MACD值
        hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()  # 信号线
        hist['Volume_MA20'] = hist['Volume'].rolling(window=20).mean()  # 20日成交量均线
        hist['Volume_MA50'] = hist['Volume'].rolling(window=50).mean()  # 50日成交量均线

        # 检查数据是否有缺失值
        missing_counts = hist.isna().sum()
        if missing_counts.sum() > 0:
            print(f"数据中存在缺失值:\n{missing_counts[missing_counts > 0]}")

        # 用后向填充法填充缺失值
        hist = hist.fillna(method='bfill')

        # 如果仍有缺失值，改为用均值填充
        if hist.isna().sum().sum() > 0:
            print("仍有无法填充的缺失值，改为均值填充")
            hist = hist.fillna(hist.mean())

        # 选择用于LSTM模型训练的特征列
        feature_columns = ['Close', 'MA20', 'MA50', 'MA200', 'RSI', 'MACD', 'Volume_MA20', 'Volume_MA50']
        hist = hist.dropna()  # 删除缺失数据

        # 对数据进行标准化处理
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(hist[feature_columns])

        # 准备LSTM训练数据：将数据转换为时间序列
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:(i + seq_length)])  # 特征数据
                y.append(data[i + seq_length, 0])  # 目标值（即股票收盘价）
            return np.array(X), np.array(y)

        seq_length = 20  # 每个样本的时间步长为20
        X_lstm, y_lstm = create_sequences(scaled_data, seq_length)

        # 构建LSTM模型
        lstm_model = Sequential([
            LSTM(32, return_sequences=True, input_shape=(seq_length, len(feature_columns)), dropout=0.2),
            LSTM(32, dropout=0.2),
            Dense(1)
        ])
        lstm_model.compile(optimizer='adam', loss='mse')  # 使用Adam优化器和均方误差损失函数
        lstm_model.fit(X_lstm, y_lstm, epochs=30, batch_size=32, verbose=0)  # 训练模型

        # 使用训练好的LSTM模型进行预测
        lstm_pred = lstm_model.predict(X_lstm)
        lstm_pred_rescaled = scaler.inverse_transform(
            np.concatenate([lstm_pred, np.zeros((len(lstm_pred), len(feature_columns) - 1))], axis=1)
        )[:, 0]  # 反标准化预测结果

        # 获取实际的收盘价格
        min_length = len(lstm_pred_rescaled)
        actual = hist['Close'][-min_length:]

        # 计算预测的置信区间（99%置信区间）
        residuals = actual - lstm_pred_rescaled
        std = np.std(residuals)
        conf_interval = 2.576 * std  # 99%置信区间

        # 计算未来的趋势（最近5天与之前5天的均值差异）
        future_trend = np.mean(lstm_pred_rescaled[-5:]) - np.mean(lstm_pred_rescaled[-10:-5])
        trend_desc = '上升' if future_trend > 0 else '下降'

        # 计算趋势的幅度
        trend_magnitude = abs(future_trend)
        if trend_magnitude > 3:
            star_rating = "★★★★★"
        elif trend_magnitude > 2:
            star_rating = "★★★★"
        elif trend_magnitude > 1.5:
            star_rating = "★★★"
        elif trend_magnitude > 1:
            star_rating = "★★"
        else:
            star_rating = "★"

        # 绘制预测结果图
        plot_stock_prediction(hist, lstm_pred_rescaled, actual, conf_interval, stock_code)

        # 生成分析报告
        volatility = np.std(hist['Close'].pct_change().dropna())  # 计算市场波动率
        prediction_error = np.mean(np.abs(residuals)) / np.mean(actual)  # 计算预测误差
        trend_consistency = np.corrcoef(actual[:-1], actual[1:])[0, 1]  # 计算趋势一致性

        # 提取技术指标的当前值
        current_rsi = hist['RSI'].iloc[-1]
        current_macd = hist['MACD'].iloc[-1]
        current_signal = hist['Signal'].iloc[-1]
        current_ma20 = hist['MA20'].iloc[-1]
        current_ma50 = hist['MA50'].iloc[-1]
        current_ma200 = hist['MA200'].iloc[-1]

        # 构建报告
        report = (
            f"股票代码：{stock_code}\n"
            f"分析时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"模型分析结果：\n"
            f"• 预测趋势：{trend_desc} (仅供参考)\n"
            f"• 预测幅度：{trend_magnitude:.2f}\n"
            f"• {trend_desc}指数：{star_rating}\n"
            f"• 预测波动区间：±{conf_interval:.2f} (99%置信区间)\n"
            f"• 市场波动率：{volatility:.2%}\n\n"
            f"技术指标分析：\n"
            f"• 相对强弱指数 (RSI)：{current_rsi:.2f} {'超买' if current_rsi > 70 else '超卖' if current_rsi < 30 else '正常'}\n"
            f"• MACD 值：{current_macd:.2f}, 信号线：{current_signal:.2f} "
            f"{'买入信号' if current_macd > current_signal else '卖出信号' if current_macd < current_signal else '无明显信号'}\n"
            f"• MA20 (20日均线)：{current_ma20:.2f}\n"
            f"• MA50 (50日均线)：{current_ma50:.2f}\n"
            f"• MA200 (200日均线)：{current_ma200:.2f}\n"
        )

    except Exception as e:
        print(f"分析过程中发生错误：{e}")

    return report