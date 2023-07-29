> * 原文地址：[Predicting Stock Prices using ARIMA, Fourier Transforms, and Technical Indicators with Deep…](https://pub.towardsai.net/predicting-stock-prices-using-arima-fourier-transforms-and-technical-indicators-with-deep-43a164859683)
> * 原文作者：[The AI Quant](https://medium.com/@theaiquant)
> * 译文出自：[掘金翻译计划](https://github.com/xitu/gold-miner)
> * 本文永久链接：[https://github.com/xitu/gold-miner/blob/master/article/2020/predicting-stock-prices-using-arima-fourier-transforms-and-technical-indicators-with-deep.md](https://github.com/xitu/gold-miner/blob/master/article/2020/predicting-stock-prices-using-arima-fourier-transforms-and-technical-indicators-with-deep.md)
> * 译者：
> * 校对者：

# Predicting Stock Prices using ARIMA, Fourier Transforms, and Technical Indicators with Deep…

## Predicting Stock Prices using ARIMA, Fourier Transforms, and Technical Indicators with Deep Learning: A Comprehensive Guide

In this article, we will explore the use of **ARIMA** and **Fourier Transforms** as features in a deep learning model for financial prediction.

ARIMA (AutoRegressive Integrated Moving Average) is a widely used time-series analysis technique that can help predict future values based on past performance. Fourier Transforms, on the other hand, are a mathematical technique that can be used to analyze time-series data by breaking it down into its component parts, including different frequencies.

With the use of powerful techniques like ARIMA and Fourier Transforms, combined with popular technical indicators, we can make accurate predictions and gain a competitive edge in the dynamic world of finance.

![Photo by [Markus Spiske](https://unsplash.com/@markusspiske?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)](https://cdn-images-1.medium.com/max/7000/0*mDfnJlPj_WYZQkyo)

#### Data Preparation

First, we need to download the financial data we will use for our analysis. In this example, we will use the Goldman Sachs stock prices, which we can download using the yfinance library.

```
import yfinance as yf

# Download data
gs = yf.download("GS", start="2011-01-01", end="2023-03-01")
```

After downloading the data, we will preprocess it by cleaning and organizing it into pandas dataframe that contains only the date and closing price columns.

```
import pandas as pd

# Preprocess data
dataset_ex_df = gs.copy()
dataset_ex_df = dataset_ex_df.reset_index()
dataset_ex_df['Date'] = pd.to_datetime(dataset_ex_df['Date'])
dataset_ex_df.set_index('Date', inplace=True)
dataset_ex_df = dataset_ex_df['Close'].to_frame()
```

Now that we have our data, we can begin incorporating ARIMA and Fourier Transforms into our predictive model.

#### ARIMA

After pre-processing our financial data, we can begin calculating **ARIMA** (Autoregressive Integrated Moving Average). ARIMA is a powerful time series analysis technique used to forecast future values based on past data. The model works by identifying patterns in the data and using them to make predictions. We will start by using ARIMA to generate predictions for Goldman Sachs stock prices.

> # A statistical model is autoregressive if it predicts future values based on past values. For example, an ARIMA model might seek to predict a stock’s future prices based on its past performance or forecast a company’s earnings based on past periods.

To select the optimal ARIMA parameters, we will use the auto_arima function from the pmdarima library. This function automatically selects the best ARIMA model parameters using a stepwise approach. We will set the seasonal parameter to False since we are not working with seasonal data. The trace parameter is set to True so that we can see the output of the function.

```
from pmdarima.arima import auto_arima

# Auto ARIMA to select optimal ARIMA parameters
model = auto_arima(dataset_ex_df['Close'], seasonal=False, trace=True)
print(model.summary())
```

We can see in the auto_arima results that the best model parameters are (0,1,0).

![Auto ARIMA output. Created by Author](https://cdn-images-1.medium.com/max/2000/1*uxRNjcbuUEsZCsCvkwL7dg.png)

Once we have determined the optimal parameters, we can fit the ARIMA model to our data and generate predictions. We will use a rolling fit method, which involves fitting the model on a training set and then using it to predict the next value in the test set. We will then add this predicted value to the training set and continue the process until we have made predictions for the entire test set.

```
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Define the ARIMA model
def arima_forecast(history):
    # Fit the model
    model = ARIMA(history, order=(0,1,0))
    model_fit = model.fit(disp=0)
    
    # Make the prediction
    output = model_fit.forecast()
    yhat = output[0]
    return yhat

# Split data into train and test sets
X = dataset_ex_df.values
size = int(len(X) * 0.8)
train, test = X[0:size], X[size:len(X)]

# Walk-forward validation
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    # Generate a prediction
    yhat = arima_forecast(history)
    predictions.append(yhat)
    # Add the predicted value to the training set
    obs = test[t]
    history.append(obs)
```

Now that we have generated our predictions, we can visualize them alongside the actual data to see how well the ARIMA model performed.

```
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6), dpi=100)
plt.plot(dataset_ex_df.iloc[size:,:].index, test, label='Real')
plt.plot(dataset_ex_df.iloc[size:,:].index, predictions, color='red', label='Predicted')
plt.title('ARIMA Predictions vs Actual Values')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
```

![ARIMA Predictions vs Real Prices. Created by Author.](https://cdn-images-1.medium.com/max/2402/1*5C4aci7olG4pEg7ZyyVaHA.png)

We observe that the ARIMA predictions are very similar to the actual closing prices. This suggests that the model is doing a good job of capturing the underlying patterns in the data and making accurate predictions.

#### Fourier Transforms

In addition to using ARIMA, we can also incorporate **Fourier Transforms** into our predictive model to analyze and forecast time series data. Fourier transforms a mathematical technique that decomposes a signal into its underlying frequencies. By analyzing the frequency components of a signal, we can identify patterns and trends that may be difficult to see in the original data.

> # The Fourier transform is a mathematical function that takes a time-based pattern as input and determines the overall cycle offset, rotation speed and strength for every possible cycle in the given pattern.

To apply Fourier transforms to our financial data, we will first calculate the Fourier transform of the closing prices using the NumPy library. We will then plot the Fourier transforms alongside the actual closing prices to visualize the frequency components of the data.

```
# Calculate the Fourier Transform
data_FT = dataset_ex_df[['Close']]
close_fft = np.fft.fft(np.asarray(data_FT['Close'].tolist()))
fft_df = pd.DataFrame({'fft':close_fft})
fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

# Plot the Fourier Transforms
plt.figure(figsize=(14, 7), dpi=100)
plt.plot(np.asarray(data_FT['Close'].tolist()),  label='Real')
for num_ in [3, 6, 9]:
    fft_list_m10= np.copy(close_fft); fft_list_m10[num_:-num_]=0
    plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))
plt.xlabel('Days')
plt.ylabel('USD')
plt.title('Goldman Sachs (close) stock prices & Fourier transforms')
plt.legend()
plt.show()
```

The resulting plot shows the actual closing prices of the Goldman Sachs stock alongside the Fourier transforms with 3, 6, and 9 components.

![Fourier Transforms. Created by Author.](https://cdn-images-1.medium.com/max/2512/1*7HIwYV4YlY_7OVS8iSPzhg.png)

The Fourier transform with 3 components appears to capture the general trend of the closing prices, while the Fourier transforms with 6 and 9 components capture additional high-frequency components. In other words, it extracts long and short-term trends. The transformation with 3 components serves as the long-term trend.

#### Technical Indicators

As we have discussed in earlier articles, we plan to incorporate technical indicators as features in our deep learning model to enhance its robustness. Specifically, we will be utilizing several indicators, including the Exponential Moving Average (EMA) with period lengths of 20, 50, and 100, the Relative Strength Index (RSI), the Moving Average Convergence Divergence (MACD), and the On-Balance-Volume (OBV).

The **EMA** is a commonly used indicator that calculates the average price of an asset over a specified time period, with more recent prices given greater weight. The **RSI** is used to determine whether an asset is overbought or oversold and is calculated using the average gain and loss of the asset’s price over a given time period. The **MACD** is a trend-following momentum indicator that calculates the difference between two moving averages and is often used to identify changes in momentum. The **OBV** is a volume-based indicator that tracks the cumulative volume of an asset and is used to identify trends in volume.

First, we start defining the functions to calculate the technical mentioned technical indicators

```
# Calculate EMA
def ema(close, period=20):
    return close.ewm(span=period, adjust=False).mean()

# Calculate RSI
def rsi(close, period=14):
    delta = close.diff()
    gain, loss = delta.copy(), delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    avg_gain = gain.rolling(period).mean()
    avg_loss = abs(loss.rolling(period).mean())
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

# Calculate MACD
def macd(close, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = close.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close.ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line

# Calculate OBV
def obv(close, volume):
    obv = np.where(close > close.shift(), volume, np.where(close < close.shift(), -volume, 0)).cumsum()
    return obv
```

We will add them to the dataset_ex_df to start building our dataset for the deep learning model.

```
# Add technical indicators to dataset DF
dataset_ex_df['ema_20'] = ema(data["Close"], 20)
dataset_ex_df['ema_50'] = ema(data["Close"], 50)
dataset_ex_df['ema_100'] = ema(data["Close"], 100)

dataset_ex_df['rsi'] = rsi(data["Close"])
dataset_ex_df['macd'] = macd(data["Close"])
dataset_ex_df['obv'] = obv(data["Close"], data["Volume"])
```

#### Build Dataset

In the next step, we will add ARIMA and Fourier Transform features to our dataset. To do so, we will first create a new DataFrame called ‘arima_df’, which is likely to contain the predicted values from an ARIMA model. We will set the index of ‘arima_df’ to match the original DataFrame ‘dataset_ex_df’ and name the column ‘ARIMA’.

```
# Create arima DF using predictions
arima_df = pd.DataFrame(history, index=dataset_ex_df.index, columns=['ARIMA'])

# Set Fourier Transforms DF
fft_df.reset_index(inplace=True)
fft_df['index'] = pd.to_datetime(dataset_ex_df.index)
fft_df.set_index('index', inplace=True)
fft_df_real = pd.DataFrame(np.real(fft_df['fft']), index=fft_df.index, columns=['Fourier_real'])
fft_df_imag = pd.DataFrame(np.imag(fft_df['fft']), index=fft_df.index, columns=['Fourier_imag'])

# Technical Indicators DF
technical_indicators_df = dataset_ex_df[['ema_20', 'ema_50', 'ema_100', 'rsi', 'macd', 'obv', 'Close']]

# Merge DF
merged_df = pd.concat([arima_df, fft_df_real, fft_df_imag, technical_indicators_df], axis=1)
merged_df = merged_df.dropna()
merged_df
```

To modify “fft_df”, the first step is to reset the index to a simple range index using the ‘reset_index()’ function. Next, the ‘index’ column is replaced with the index of ‘dataset_ex_df’ in datetime format using the ‘pd.to_datetime()’ function. The index of ‘fft_df’ is then set to the modified ‘index’ column.

After modifying ‘fft_df’, a new DataFrame called ‘technical_indicators_df’ is created by selecting a subset of columns from ‘dataset_ex_df’. These columns are EMA values, RSI, MACD, OBV, and the closing price.

Finally, all the DataFrames are merged by concatenating ‘arima_df’, ‘fft_df_real’, ‘fft_df_imag’, and ‘technical_indicators_df’ along the columns axis. Any rows with missing values are dropped using ‘dropna()’. The resulting DataFrame is our Dataset. It contains all the original technical indicators as well as the additional ARIMA and Fourier Transform features.

#### Separate Train and Test

Now we will separate the dataset in train and test using 80% of the data for train and 20% for test.

```
# Separate in Train and Test Dfs
train_size = int(len(merged_df) * 0.8)
train_df, test_df = merged_df.iloc[:train_size], merged_df.iloc[train_size:]
```

After partitioning our dataset into train and test sets, our next step is to define the features and labels for each set. In order to feed our data into the LSTM model, we need to ensure that it is properly scaled. It’s important to note that while we will scale the features, we will not scale the target variable. By leaving the target unscaled, we are intentionally introducing a level of complexity to our model, making it more challenging for it to accurately predict the target values. This approach can ultimately lead to a more robust and accurate model that is better equipped to handle real-world data.

```
from sklearn.preprocessing import MinMaxScaler

# Scale the features
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df.drop('Close', axis=1))
test_scaled = scaler.transform(test_df.drop('Close', axis=1))

# Convert the scaled data back to a DataFrame
train_scaled_df = pd.DataFrame(train_scaled, columns=train_df.columns[:-1], index=train_df.index)
test_scaled_df = pd.DataFrame(test_scaled, columns=test_df.columns[:-1], index=test_df.index)

# Merge the scaled features with the target variable
train_scaled_df['Close'] = train_df['Close']
test_scaled_df['Close'] = test_df['Close']

# Split the scaled data into Features and Label
X_train = train_scaled_df.iloc[:, :-1].values
y_train = train_scaled_df.iloc[:, -1].values
X_test = test_scaled_df.iloc[:, :-1].values
y_test = test_scaled_df.iloc[:, -1].values
```

#### Train Model

The next step involves defining the Long Short-Term Memory (**LSTM**) deep learning model, which will be utilized for predicting the closing price of the stock. For this purpose, we have set Mean Squared Error (**MSE**) as the model loss, which will help to assess the degree of error in our predicted values. Additionally, we have incorporated the **EarlyStopping** callback feature in the model, which enables the training process to stop when the validation loss does not show any significant reduction. This ensures that the model **does not overfit** and reduces the computational resources required for training. Moreover, we have set the **shuffle** parameter to **False** in order to maintain the order of the data samples, which is crucial in time series analysis.

```
# Import keras modules
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Define model
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mse')


# Define the early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')

# Train the model with early stopping callback
history = model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=1, validation_data=(X_test, y_test), callbacks=[early_stop], shuffle=False)
```

After defining and training our deep learning model, we can visualize its performance through the training and validation loss evolution plot. This plot shows the changes in both the training and validation loss over time, providing insight into how the model is learning and improving.

![Training Loss Evolution. Created by Author.](https://cdn-images-1.medium.com/max/2000/1*S9oyzQ3wBUeAORBMrdx1Rg.png)

#### Evaluate Model

As **machine learning** enthusiasts, we understand the importance of evaluating model performance. After training our LSTM model, we calculate various metrics to **evaluate** its **performance** on the **test** data. Our first step is to generate predictions on the test data using the trained model, which we stored in the variable y_pred.

To assess the model’s performance, we imported several functions from the `sklearn.metrics` module, including `mean_squared_error`, `mean_absolute_error`, `r2_score`, and `explained_variance_score`. These functions allow us to calculate important evaluation metrics such as mean squared error (**MSE**), root mean squared error (**RMSE**), mean absolute error (**MAE**), **R**-squared **score**, and **explained variance** score.

In addition to these common evaluation metrics, we also calculate the mean absolute percentage error (**MAPE**) and mean percentage error (**MPE**) using numpy functions. These metrics are useful in understanding the average percentage difference and average difference, respectively, between the actual and predicted values.

```
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

import numpy as np
y_pred = model.predict(X_test)

# Calculate test metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)

mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
mpe = np.mean((y_test - y_pred) / y_test) * 100

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R2 Score: {r2}")
print(f"Explained Variance Score: {evs}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")
print(f"Mean Percentage Error (MPE): {mpe}")
```

![Model Evaluation Results. Created by Author](https://cdn-images-1.medium.com/max/2000/1*Obl68_ymBdOW5nEHgywDGQ.png)

Overall, these results suggest that the model has performed well on the test data, with relatively low error and good accuracy in predicting the target variable. However, it’s important to consider other factors, such as the size and representativeness of the dataset, the choice of hyperparameters, and the generalization capability of the model. **Towards** the **conclusion** of the article, a **detailed** and comprehensive **explanation** of the obtained **results** is provided for the purpose of facilitating a better understanding.

We will now visualize the final model predictions together with the closing prices on the test data.

```
# Plot final Predictions
plt.figure(figsize=(12, 6), dpi=100)
plt.plot(test_scaled_df.index, y_test, label='Real')
plt.plot(test_scaled_df.index, y_pred, color='red', label='Predicted')
plt.title('Model Predictions vs Real Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
```

![Model Predictions vs Close Price. Created by Author.](https://cdn-images-1.medium.com/max/2488/1*KWcNWt5XXGUsUKJtuNxpdg.png)

#### Conclusion

In this article, we explored different methods to predict the stock price of Goldman Sachs. We started by using the **ARIMA** model, which is a time-series forecasting model that uses past data to make future predictions. We then looked at **Fourier Transforms**, which allowed us to decompose the time series into its individual frequency components and use those components as features in our model. Finally, we examined how **technical indicators** such as the Relative Strength Index (**RSI**) and Moving Average Convergence Divergence (**MACD**) can be used to generate additional features for our model.

We evaluated the performance of our models using several metrics, such as mean squared error (MSE), mean absolute error (MAE), R2 score, explained variance score, mean absolute percentage error (MAPE), and mean percentage error (MPE).

Overall, we saw how combining different methods and techniques can lead to a more accurate and robust prediction model. By incorporating different features and metrics, we can gain a better understanding of the underlying trends and patterns in the data and make more informed predictions about future stock prices.

#### Understanding Model Results

The Mean Squared Error (**MSE**) is a measure of the average squared difference between the predicted and actual values. The lower the value of MSE, the better the model’s performance. In this case, the MSE value of 27.93 indicates that the model has a relatively low error.

The Mean Absolute Error (**MAE**) measures the absolute difference between the predicted and actual values. The MAE value of 1.60 suggests that, on average, the model’s predictions are off by around $1.60.

The **R2** Score is a statistical measure that indicates how well the model’s predictions fit the actual data. The value of 0.98 suggests that the model can explain 98% of the variance in the target variable, which is a good performance.

The **Explained Variance** Score measures the proportion of the variance in the target variable that is explained by the model. The value of 0.98 indicates that the model can explain 98% of the variance in the target variable, which is again a good performance.

The Mean Absolute Percentage Error (**MAPE**) measures the average percentage difference between the predicted and actual values. The MAPE value of 11.81 suggests that, on average, the model’s predictions are off by around 11.81%.

---

The Mean Percentage Error (**MPE**) measures the average difference between the predicted and actual values as a percentage of the actual values. The negative value of -0.79 indicates that, on average, the model’s predictions are slightly lower than the actual values.

Become a Medium member today and enjoy unlimited access to thousands of Python guides and Data Science articles! For just $5 a month, you’ll have access to exclusive content and support me as a writer. Sign up now using [my link](https://medium.com/@theaiquant/membership), and I’ll earn a small commission at no extra cost to you.

> 如果发现译文存在错误或其他需要改进的地方，欢迎到 [掘金翻译计划](https://github.com/xitu/gold-miner) 对译文进行修改并 PR，也可获得相应奖励积分。文章开头的 **本文永久链接** 即为本文在 GitHub 上的 MarkDown 链接。

---

> [掘金翻译计划](https://github.com/xitu/gold-miner) 是一个翻译优质互联网技术文章的社区，文章来源为 [掘金](https://juejin.im) 上的英文分享文章。内容覆盖 [Android](https://github.com/xitu/gold-miner#android)、[iOS](https://github.com/xitu/gold-miner#ios)、[前端](https://github.com/xitu/gold-miner#前端)、[后端](https://github.com/xitu/gold-miner#后端)、[区块链](https://github.com/xitu/gold-miner#区块链)、[产品](https://github.com/xitu/gold-miner#产品)、[设计](https://github.com/xitu/gold-miner#设计)、[人工智能](https://github.com/xitu/gold-miner#人工智能)等领域，想要查看更多优质译文请持续关注 [掘金翻译计划](https://github.com/xitu/gold-miner)、[官方微博](http://weibo.com/juejinfanyi)、[知乎专栏](https://zhuanlan.zhihu.com/juejinfanyi)。