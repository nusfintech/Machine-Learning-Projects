import numpy as np

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.metrics import mean_squared_error

import pandas as pd
import talib as ta


class BasicTemplateAlgorithm(QCAlgorithm):

    def Initialize(self):

        self.SetStartDate(2018, 11, 7)  # Set Start Date
        self.SetEndDate(2018, 11, 21)  # Set End Date
        self.SetCash(100000)  # Set Strategy Cash
        self.currency = "EURUSD"  # Set currency pair
        self.symbol = self.AddForex(self.currency, Resolution.Daily).Symbol
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage,
                               AccountType.Margin)  # Set Brokerage Model
        self.SetWarmUp(365)

        self.hist_data = pd.DataFrame(
            self.History([self.currency], 365,
                         Resolution.Daily))  # Asking for past 1 year of historical data
        self.long_list = []  # To track short/long position
        self.short_list = []
        self.model = Sequential()
        self.atr = [0]  # Track current average true range

        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Monday, DayOfWeek.Tuesday,  # Schedule function to enter a position
                                 DayOfWeek.Wednesday, DayOfWeek.Thursday,
                                 DayOfWeek.Friday),
            self.TimeRules.Every(TimeSpan.FromMinutes(360)),
            Action(self.Rebalance))

        self.Schedule.On(  # Schedule function to exit a position
            self.DateRules.EveryDay(self.symbol),
            self.TimeRules.Every(TimeSpan.FromMinutes(10)),
            Action(self.Rebalance2))

    def OnData(self, data):
        if self.IsWarmingUp:
            return

        df1 = self.hist_data  # Calculate and update new atr everyday
        df1 = df1[['open', 'high', 'low', 'close']]
        self.atr = ta.ATR(df1['high'], df1['low'], df1['close'], timeperiod=14)
        self.Debug("atr" + str(self.atr[-1]))

    def Rebalance(self):
        self.hist_data = pd.DataFrame(
            self.History([self.currency], 365,
                         Resolution.Daily))  # Updating historical data for training

        if not self.hist_data.empty and self.currency not in self.long_list and self.currency not in self.short_list:
            df = self.hist_data

            df = df[['open', 'high', 'low', 'close']]  # Features to be included
            df['avg_price'] = (df['low'] + df['high']) / 2  # Average Price
            df['range'] = df['high'] - df['low']  # Price Range
            df['ohlc_price'] = (  # Open-High-Low-Close Average Price
                                       df['low'] + df['high'] + df['open'] + df['close']) / 4
            df['oc_diff'] = df['open'] - df['close']  # Open Close Difference
            df['percentage_change'] = (df['close'] - df['open']) / df['open']  # Percentage change in price for the day
            df['cp_change'] = (df['close'] - df['close'].shift(1)) / (
                df['close'].shift(1))  # Change in closing price from the previous day
            df['SMA'] = df['close'].rolling(window=5).mean()  # Simple Moving Average of 5 days
            df.dropna(inplace=True)

            dataset = df.copy().values.astype('float32')
            pca_features = df.columns.tolist()

            pca = PCA(n_components=1)  # Price Component Analysis
            df['pca'] = pca.fit_transform(dataset)

            target_index = df.columns.tolist().index('close')
            dataset = df.values.astype('float32')

            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset)  # Normalise the dataset to a common scale between 0 and 1

            y_scaler = MinMaxScaler(feature_range=(0, 1))  # Normalise the dataset to a common scale between 0 and 1
            t_y = df['close'].values.astype('float32')
            t_y = np.reshape(t_y, (-1, 1))  # Reshape the output data
            y_scaler = y_scaler.fit(t_y)  # Normalise the dataset to a common scale between 0 and 1

            dataX, dataY = [], []

            for i in range(len(dataset) - 10 - 1):  # 10 days lookback/windowing
                a = dataset[i:(i + 10)]
                dataX.append(a)
                dataY.append(dataset[i + 10])

            X, y = np.array(dataX), np.array(dataY)
            y = y[:, target_index]

            train_size = int(len(X) * 0.95)  # Test-Train split, 98% train and 5% test
            trainX = X[:train_size]
            trainY = y[:train_size]
            testX = X[train_size:]
            testY = y[train_size:]

            self.model = Sequential()  # Setting up of LSTM Model
            self.model.add(LSTM(20, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
            self.model.add(LSTM(10, return_sequences=True))
            self.model.add(Dropout(0.2))
            self.model.add(LSTM(10, return_sequences=False))
            self.model.add(
                Dense(1, kernel_initializer='he_normal', activation='relu'))

            self.model.compile(
                loss='mean_squared_error',
                optimizer='adam',
                metrics=['mae', 'mse'])

            self.model.fit(
                trainX, trainY, epochs=300, verbose=0)

            self.Debug("Model is fitted")

            pred = self.model.predict(testX)  # Predict on test data
            pred = y_scaler.inverse_transform(pred)
            close = y_scaler.inverse_transform(
                np.reshape(testY, (testY.shape[0], 1)))
            predictions = pd.DataFrame()  # Create dataframe for evaluation of prediction on test data
            predictions['predicted'] = pd.Series(
                np.reshape(pred, (pred.shape[0])))
            predictions['actual'] = pd.Series(
                np.reshape(close, (close.shape[0])))
            predictions = predictions.astype(float)
            predictions['diff'] = predictions['predicted'] - predictions[
                'actual']  # Calculate difference between predicted and actual value

            p = df[-pred.shape[0]:].copy()
            predictions.index = p.index
            predictions = predictions.astype(float)
            predictions = predictions.merge(
                p[['low', 'high']], right_index=True,
                left_index=True)  # Add in high low price to see whether predicted value falls between this range
            self.Debug(predictions)

            trainscore = mean_squared_error(close, pred)
            self.Debug('MSE' + str(trainscore))

            pred_data = np.array([dataset[-10:]])  # Prepare data for prediction of next day close price
            output = self.model.predict(pred_data)
            output = y_scaler.inverse_transform(output)
            predictions2 = pd.DataFrame()
            predictions2['predicted'] = pd.Series(
                np.reshape(output, (output.shape[0])))
            predictions2 = predictions2.astype(float)
            output = predictions2['predicted'][0]  # Get predicted value

            self.Debug("Predicted price is " + str(output))

            prev_price = predictions['actual'][-1]  # Get previous day's value
            self.Debug("Prev_price is " + str(prev_price))

            # Buy/Sell Execution conditions

            if output > prev_price and self.currency not in self.long_list and self.currency not in self.short_list:
                self.SetHoldings(self.currency, 1)
                self.long_list.append(self.currency)
                self.Debug("long")

            if output < prev_price and self.currency not in self.long_list and self.currency not in self.short_list:
                self.SetHoldings(self.currency, -1)
                self.short_list.append(self.currency)
                self.Debug("short")

    def Rebalance2(self):
        # curr_price = self.Securities["EURUSD"].Price  # Get current market price
        curr_price = pd.DataFrame(
            self.History([self.currency], 120,
                         Resolution.Hour))
        curr_price = curr_price['close'][-1]


        # Exit execution Conditions
        df1 = self.hist_data  # Calculate and update new atr everyday
        df1 = df1[['open', 'high', 'low', 'close']]
        self.atr = ta.ATR(df1['high'], df1['low'], df1['close'], timeperiod=14)
        self.Debug("atr " + str(self.atr[-1]))
        self.Debug("curr_price " + str(curr_price))

        if self.currency in self.long_list:
            cost_basis = self.Portfolio[self.currency].AveragePrice  # To calculate the price of our FOREX currency pair

            if ((curr_price <= float(cost_basis) - float(self.atr[-1]))  # Stop loss-Take Profit conditions
                    or (curr_price >=  # Use of Average True Range to determine price to sell
                        float(self.atr[-1] * 1) + float(cost_basis))):
                self.Debug("price is: " + str(curr_price))
                self.SetHoldings(self.currency, 0)
                self.long_list.remove(self.currency)


        if self.currency in self.short_list:
            cost_basis = self.Portfolio[self.currency].AveragePrice

            if ((curr_price <= float(cost_basis) - float(self.atr[-1] * 1))  # Stop loss-Take Profit conditions
                    or  # Use of Average True Range to determine price to buy back
                    (curr_price >= float(self.atr[-1]) + float(cost_basis))):
                self.Debug("price is: " + str(curr_price))
                self.SetHoldings(self.currency, 0)
                self.short_list.remove(self.currency)