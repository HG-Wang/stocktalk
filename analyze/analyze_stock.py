import os
import sys
import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import matplotlib
matplotlib.use('Agg')

import numpy as np
from financeAss import plot_stock_prediction
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def fetch_stock(stock_code):
    # 获取股票历史数据（过去2年，增加数据量以提高模型准确度）
    stock = yf.Ticker(stock_code)
    hist = stock.history(period="2y")
    return hist


def analyze_stock(stock_code, hist, analysis_mode='deep'):
    report = ()
    try:
        # 导入需要的库
        from sklearn.metrics import mean_absolute_error
        print(hist)
        print(f"分析模式: {analysis_mode}")

        # 计算技术指标
        hist['MA20'] = hist['Close'].rolling(window=20).mean()  # 20日均线
        hist['MA50'] = hist['Close'].rolling(window=50).mean()  # 50日均线
        hist['MA200'] = hist['Close'].rolling(window=200).mean()  # 200日均线
        hist['STD'] = hist['Close'].rolling(window=20).std()  # 20日标准差
        hist['UpperBB'] = hist['MA20'] + (hist['STD'] * 2)  # 布林带上轨
        hist['LowerBB'] = hist['MA20'] - (hist['STD'] * 2)  # 布林带下轨
        
        # 添加更多高级技术指标
        # 相对强弱指数 (RSI)
        delta = hist['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD指标
        hist['EMA12'] = hist['Close'].ewm(span=12, adjust=False).mean()
        hist['EMA26'] = hist['Close'].ewm(span=26, adjust=False).mean()
        hist['MACD'] = hist['EMA12'] - hist['EMA26']
        hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
        hist['MACD_Hist'] = hist['MACD'] - hist['Signal']
        
        # 添加价格动量指标 (ROC - Rate of Change)
        hist['ROC_5'] = hist['Close'].pct_change(periods=5) * 100
        hist['ROC_10'] = hist['Close'].pct_change(periods=10) * 100
        hist['ROC_20'] = hist['Close'].pct_change(periods=20) * 100
        
        # 计算平均交易量
        hist['Volume_MA20'] = hist['Volume'].rolling(window=20).mean()
        hist['Volume_MA50'] = hist['Volume'].rolling(window=50).mean()
        hist['Volume_Ratio'] = hist['Volume'] / hist['Volume_MA20']  # 成交量比率
        
        # 添加价格与均线的距离百分比指标
        hist['MA20_Distance'] = (hist['Close'] - hist['MA20']) / hist['MA20'] * 100
        hist['MA50_Distance'] = (hist['Close'] - hist['MA50']) / hist['MA50'] * 100
        
        # ATR (平均真实波幅) - 衡量市场波动性
        hist['TR1'] = abs(hist['High'] - hist['Low'])
        hist['TR2'] = abs(hist['High'] - hist['Close'].shift())
        hist['TR3'] = abs(hist['Low'] - hist['Close'].shift())
        hist['TR'] = hist[['TR1', 'TR2', 'TR3']].max(axis=1)
        hist['ATR'] = hist['TR'].rolling(window=14).mean()
        
        # OBV (能量潮指标)
        hist['OBV'] = (np.sign(hist['Close'].diff()) * hist['Volume']).fillna(0).cumsum()

        # 检查数据是否有缺失值
        missing_counts = hist.isna().sum()
        if missing_counts.sum() > 0:
            print(f"数据中存在缺失值:\n{missing_counts[missing_counts > 0]}")

        # 用后向填充法填充缺失值
        hist = hist.bfill()

        # 如果仍有缺失值，改为用均值填充
        if hist.isna().sum().sum() > 0:
            print("仍有无法填充的缺失值，改为均值填充")
            hist = hist.fillna(hist.mean())
        
        # 特征工程：创建更多有意义的特征
        # 准备AutoTS需要的时间序列数据
        df = hist[['Close']].copy()
        df['date'] = df.index
        df = df.reset_index(drop=True)
        
        if analysis_mode == 'deep':
            # 导入AutoTS库，仅在深度分析模式下使用
            from autots import AutoTS
            # 配置AutoTS模型
            model = AutoTS(
                forecast_length=30,  # 预测30天
                frequency='D',      # 日频数据
                prediction_interval=0.95,  # 95%置信区间
                ensemble='simple',  # 使用简单集成避免参数错误
                max_generations=5,  # 平衡速度与准确性
                num_validations=2,  # 减少验证次数
                model_list="fast",  # 使用稳定模型集合
                transformer_list="fast",  # 基本特征变换
                transformer_max_depth=2,  # 控制转换深度
                drop_most_recent=0,  # 不丢弃最近数据
                n_jobs=4,  # 使用全部CPU核心
                verbose=1  # 适度日志输出
            )
            
            # 训练模型
            model = model.fit(df, date_col='date', value_col='Close')
            
            # 生成预测并处理异常情况
            try:
                prediction = model.predict()
                if not hasattr(prediction, 'forecast') or prediction.forecast.empty:
                    print("警告：预测结果格式异常，使用应急预测数据")
                    raise ValueError("预测结果格式异常")
                forecast = prediction.forecast
            except Exception as e:
                print(f"预测失败：{str(e)}，使用应急预测数据")
                last_price = hist['Close'].iloc[-1]
                std_dev = hist['Close'].rolling(30).std().iloc[-1]
                forecast = pd.DataFrame({
                    'Close': [last_price * 1.02 ** i for i in range(1, 31)],
                    'Close_upper_0.95': [last_price * 1.02 ** i + 1.96*std_dev for i in range(1, 31)],
                    'Close_lower_0.95': [last_price * 1.02 ** i - 1.96*std_dev for i in range(1, 31)],
                    'Close_upper': [last_price * 1.05 ** i for i in range(1, 31)],
                    'Close_lower': [last_price * 0.98 ** i for i in range(1, 31)]
                }, index=pd.date_range(start=hist.index[-1], periods=30, freq='D'))
            
            # 获取预测结果并处理列名差异
            forecast = prediction.forecast
            
            # 增强型列名匹配逻辑
            print(f"预测结果列名：{list(forecast.columns)}")  # 调试输出
            
            # 优先匹配完整列名格式
            upper_cols = [col for col in forecast.columns if 'Close_upper' in col or '_upper_' in col]
            lower_cols = [col for col in forecast.columns if 'Close_lower' in col or '_lower_' in col]
            
            # 次优先匹配分位数格式
            if not upper_cols:
                upper_cols = [col for col in forecast.columns if '0.975' in col or '95' in col]
            if not lower_cols:
                lower_cols = [col for col in forecast.columns if '0.025' in col or '05' in col]
                
            # 最后尝试首字母匹配
            if not upper_cols:
                upper_cols = [col for col in forecast.columns if col.startswith('upper')]
            if not lower_cols:
                lower_cols = [col for col in forecast.columns if col.startswith('lower')]
                
            # 获取最终列名或设置默认值
            upper_col = upper_cols[0] if upper_cols else None
            lower_col = lower_cols[0] if lower_cols else None
            
            # 处理置信区间缺失情况
            if not upper_col or not lower_col:
                print("警告：无法确定置信区间列，使用滚动标准差替代")
                rolling_std = hist['Close'].rolling(30).std().iloc[-len(forecast):].values
                conf_interval_values = 1.96 * rolling_std
                upper_col = lower_col = 'Close'  # 设置默认列名
                upper_col = lower_col = 'Close'
                conf_interval = np.std(forecast['Close'].values)
                
            # 检查必要列是否存在并设置默认值
            if 'Close' not in forecast.columns:
                forecast['Close'] = forecast.iloc[:, 0]  # 使用第一列作为Close预测值

            predicted_prices = forecast['Close'].values
            # 计算置信区间范围
            conf_interval = (forecast[upper_col].values - forecast[lower_col].values) / 2
            
            # 获取实际价格用于评估 - 使用与预测数据长度一致的最近历史数据
            actual_prices = hist['Close'].values[-len(predicted_prices):]
            
            # 获取实际的收盘价格 - 使用与预测数据长度一致的最近历史数据
            actual_prices = hist['Close'].values[-len(predicted_prices):]
            
            # 计算预测性能指标
            mae = mean_absolute_error(actual_prices, predicted_prices)
            rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
            mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
            
            # 计算R平方(确定系数)，衡量模型解释了多少目标变量的变异
            ss_res = np.sum((actual_prices - predicted_prices) ** 2)
            ss_tot = np.sum((actual_prices - np.mean(actual_prices)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # 获取置信区间数据（如果可用）
            try:
                upper_values = forecast[upper_col].values
                lower_values = forecast[lower_col].values
                conf_interval_values = (upper_values - lower_values) / 2
            except KeyError as e:
                print(f"警告：无法获取置信区间数据，原因：{e}")
                conf_interval_values = np.zeros_like(predicted_prices)
        else:  # 快速分析模式
            # 快速分析模式下不进行任何预测，也不生成预测数据
            forecast = pd.DataFrame({}, index=pd.date_range(start=hist.index[-1], periods=30, freq='D'))
            predicted_prices = []
            conf_interval = []
            actual_prices = []
            mae = 0.0
            rmse = 0.0
            mape = 0.0
            r_squared = 0.0
            conf_interval_values = []
            # 注释：特征工程将由大语言模型处理
        
        # 计算未来趋势分析（快速分析模式下不进行预测）
        if analysis_mode == 'deep' and len(predicted_prices) > 10:
            future_trend = np.mean(predicted_prices[-5:]) - np.mean(predicted_prices[-10:-5])
            trend_strength = abs(future_trend) / hist['Close'].mean() * 100  # 趋势强度占股价比例
            trend_desc = '上升' if future_trend > 0 else '下降'
        else:
            future_trend = 0
            trend_strength = 0
            trend_desc = '未知'
        
        if analysis_mode == 'deep':
            # 趋势可靠性评级
            if r_squared > 0.85:
                model_reliability = "极高"
                confidence_level = 95
            elif r_squared > 0.75:
                model_reliability = "高"
                confidence_level = 85
            elif r_squared > 0.65:
                model_reliability = "中等"
                confidence_level = 75
            elif r_squared > 0.5:
                model_reliability = "低"
                confidence_level = 60
            else:
                model_reliability = "极低"
                confidence_level = 50
        else:
            model_reliability = "快速分析"
            confidence_level = 0
            
        # 计算趋势的幅度评级（快速分析模式下不进行预测）
        if analysis_mode == 'deep' and trend_strength > 0:
            if trend_strength > 5:
                trend_magnitude_desc = "极强"
                star_rating = "★★★★★"
            elif trend_strength > 3:
                trend_magnitude_desc = "强"
                star_rating = "★★★★"
            elif trend_strength > 2:
                trend_magnitude_desc = "中等"
                star_rating = "★★★"
            elif trend_strength > 1:
                trend_magnitude_desc = "弱"
                star_rating = "★★"
            else:
                trend_magnitude_desc = "极弱"
                star_rating = "★"
        else:
            trend_magnitude_desc = "未知"
            star_rating = "N/A"
            
        # 获取当前技术指标状态
        latest = hist.iloc[-1]
        current_rsi = latest['RSI']
        current_macd = latest['MACD']
        current_signal = latest['Signal']
        current_ma20 = latest['MA20']
        current_ma50 = latest['MA50']
        current_ma200 = latest['MA200']
        current_atr = latest['ATR']
        current_volume_ratio = latest['Volume_Ratio']
        
        # RSI 状态判断
        if current_rsi > 70:
            rsi_status = "超买"
            rsi_action = "谨慎做空"
        elif current_rsi < 30:
            rsi_status = "超卖"
            rsi_action = "谨慎做多"
        elif current_rsi > 60:
            rsi_status = "偏强"
            rsi_action = "观望或轻仓做多"
        elif current_rsi < 40:
            rsi_status = "偏弱"
            rsi_action = "观望或轻仓做空"
        else:
            rsi_status = "中性"
            rsi_action = "观望"
            
        # MACD 信号判断
        if current_macd > current_signal and current_macd > 0:
            macd_signal = "强烈买入信号"
        elif current_macd > current_signal and current_macd < 0:
            macd_signal = "弱买入信号"
        elif current_macd < current_signal and current_macd < 0:
            macd_signal = "强烈卖出信号"
        elif current_macd < current_signal and current_macd > 0:
            macd_signal = "弱卖出信号"
        else:
            macd_signal = "中性"
            
        # 均线系统分析
        ma_trend = ""
        if current_ma20 > current_ma50 and current_ma50 > current_ma200:
            ma_trend = "强劲上升趋势"
            ma_action = "多头市场，适合持有"
        elif current_ma20 < current_ma50 and current_ma50 < current_ma200:
            ma_trend = "强劲下降趋势"
            ma_action = "空头市场，适合观望"
        elif current_ma20 > current_ma50 and current_ma50 < current_ma200:
            ma_trend = "可能形成黄金交叉"
            ma_action = "趋势可能反转向上，关注突破信号"
        elif current_ma20 < current_ma50 and current_ma50 > current_ma200:
            ma_trend = "可能形成死亡交叉"
            ma_action = "趋势可能反转向下，关注突破信号"
        else:
            ma_trend = "盘整趋势"
            ma_action = "区间操作，等待明确信号"
            
        # 波动性分析
        volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100  # 年化波动率
        if volatility > 40:
            volatility_status = "极高"
            volatility_advice = "高风险，降低仓位"
        elif volatility > 30:
            volatility_status = "高"
            volatility_advice = "风险较大，谨慎操作"
        elif volatility > 20:
            volatility_status = "中等"
            volatility_advice = "中等风险，标准仓位"
        elif volatility > 10:
            volatility_status = "低"
            volatility_advice = "风险较低，可增加仓位"
        else:
            volatility_status = "极低"
            volatility_advice = "低风险，积极布局"
            
        # 季节性分析 - 使用 'ME' 替代 'M' 来解决弃用警告
        monthly_returns = hist['Close'].resample('ME').ffill().pct_change().dropna()
        monthly_pattern = monthly_returns.groupby(monthly_returns.index.month).mean() * 100
        best_month = monthly_pattern.idxmax()
        worst_month = monthly_pattern.idxmin()
        month_names = {1:'一月', 2:'二月', 3:'三月', 4:'四月', 5:'五月', 6:'六月', 
                      7:'七月', 8:'八月', 9:'九月', 10:'十月', 11:'十一月', 12:'十二月'}
        
        current_month = datetime.datetime.now().month
        next_month = current_month + 1 if current_month < 12 else 1
        
        month_outlook = f"本月({month_names[current_month]})历史平均收益率: {monthly_pattern.get(current_month, 0):.2f}%\n"
        month_outlook += f"下月({month_names[next_month]})历史平均收益率: {monthly_pattern.get(next_month, 0):.2f}%\n"
        month_outlook += f"历史表现最佳月份: {month_names[best_month]} ({monthly_pattern.loc[best_month]:.2f}%)\n"
        month_outlook += f"历史表现最差月份: {month_names[worst_month]} ({monthly_pattern.loc[worst_month]:.2f}%)"
        
        # 传递完整的历史数据给绘图函数
        hist_subset = hist
        
        # 绘制预测结果图 - 传递完整的forecast DataFrame，并添加日志以验证数据
        print(f"传递给绘图函数的历史数据长度: {len(hist_subset)}，完整历史数据长度: {len(hist)}")
        print(f"预测数据列名: {list(forecast.columns)}")
        plot_stock_prediction(hist_subset, forecast, actual_prices, stock_code)
        
        # 综合评分系统 (0-100分)
        # 基于技术指标、预测准确性和趋势强度
        score_technical = 0
        
        # 技术指标得分 (0-40分)
        # RSI评分
        if 40 <= current_rsi <= 60:
            score_technical += 20
        elif (30 <= current_rsi < 40) or (60 < current_rsi <= 70):
            score_technical += 15
        else:
            score_technical += 10
            
        # MACD评分
        if (current_macd > current_signal and current_macd > 0) or (current_macd < current_signal and current_macd < 0):
            score_technical += 20
        else:
            score_technical += 10
            
        # 预测准确性得分 (0-30分)
        score_accuracy = max(0, min(30, int(30 * (1 - mape/100))))
        
        # 趋势强度得分 (0-30分)
        score_trend = min(30, int(trend_strength * 6))
        
        # 总评分
        total_score = score_technical + score_accuracy + score_trend
        
        # 投资建议
        if total_score >= 80:
            investment_advice = "强烈推荐" if future_trend > 0 else "强烈回避"
            action_advice = "积极介入，建议配置仓位" if future_trend > 0 else "建议清仓观望"
        elif total_score >= 65:
            investment_advice = "推荐" if future_trend > 0 else "回避"
            action_advice = "逢低买入，适度配置仓位" if future_trend > 0 else "逢高减仓，降低风险敞口"
        elif total_score >= 50:
            investment_advice = "中性偏多" if future_trend > 0 else "中性偏空"
            action_advice = "小仓位试探性建仓" if future_trend > 0 else "减持部分头寸，控制风险"
        else:
            investment_advice = "观望"
            action_advice = "等待更明确的市场信号，暂不建议操作"
        
        # 构建专业分析报告
        report = f"📈 {stock_code} 量化分析报告 📊\n" \
                 f"──────────────────────\n" \
                 f"📆 分析时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n" \
                 f"🔄 数据区间：{hist.index[0].strftime('%Y-%m-%d')} 至 {hist.index[-1].strftime('%Y-%m-%d')}\n\n" \
                 f"🔵 核心指标\n" \
                 f"──────────────────────\n" \
                 f"• 分析模式：{model_reliability}\n" \
                 f"• 预测趋势：{trend_desc}趋势 ({trend_magnitude_desc})\n" if analysis_mode == 'deep' else "" \
                 f"• 趋势强度：{star_rating} ({trend_strength:.2f}%)\n" if analysis_mode == 'deep' else "" \
                 f"\n" \
                 f"🟠 技术面分析\n" \
                 f"──────────────────────\n" \
                 f"• RSI(14)：{current_rsi:.1f} → {rsi_status} ({rsi_action})\n" \
                 f"• MACD分析：{current_macd:.3f} | 信号线：{current_signal:.3f} → {macd_signal}\n" \
                 f"• 均线系统：{ma_trend}\n" \
                 f"  - MA20：{current_ma20:.2f} {'↑' if current_ma20 > current_ma50 else '↓'}\n" \
                 f"  - MA50：{current_ma50:.2f} {'↑' if current_ma50 > current_ma200 else '↓'}\n" \
                 f"  - MA200：{current_ma200:.2f}\n" \
                 f"• 成交量状况：{'放量' if current_volume_ratio > 1.5 else '缩量' if current_volume_ratio < 0.6 else '正常'} " \
                 f"(当前/MA20比值 = {current_volume_ratio:.2f}倍)\n" \
                 f"• 波动性指标：ATR={current_atr:.2f}, 年化波动率={volatility:.2f}% ({volatility_status})\n\n" \
                 f"🟢 综合评估\n" \
                 f"──────────────────────\n" \
                 f"• 总体评分：{total_score}/100\n" \
                 f"• 投资建议：{investment_advice}\n" \
                 f"• 操作策略：{action_advice}\n" \
                 f"• 风险提示：{volatility_advice}\n\n" \
                 f"🟣 季节性分析\n" \
                 f"──────────────────────\n" \
                 f"{month_outlook}\n\n" \
                 f"⚠️ 风险提示：本分析仅供参考，投资决策请结合多因素综合考量，注意控制风险。\n" \
                 f"──────────────────────"

    except Exception as e:
        print(f"分析过程中发生错误：{e}")
        import traceback
        traceback.print_exc()
        # 确保即使发生错误也返回一个基本的报告
        report = (
            f"📈 {stock_code} 量化分析报告 📊\n"
            f"──────────────────────\n"
            f"📆 分析时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"🔄 数据区间：{hist.index[0].strftime('%Y-%m-%d')} 至 {hist.index[-1].strftime('%Y-%m-%d')}\n\n"
            f"⚠️ 分析过程中发生错误：{str(e)}\n"
            f"⚠️ 风险提示：本分析仅供参考，投资决策请结合多因素综合考量，注意控制风险。\n"
            f"──────────────────────"
        )
        print(f"错误报告内容：{report}")

    print(f"返回的报告内容：{report}")
    return report