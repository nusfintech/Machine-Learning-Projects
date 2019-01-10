from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import floor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import talib 
from talib import MA_Type

class BasicTemplateAlgorithm(QCAlgorithm):

    def Initialize(self):
        
        '''
        NOTE: 
            BLOCK COMMENTS  - USER-DEFINED PARAMETERS
                            - DEBUG BLOCKS
            LINE COMMENTS   - CODE-SNIPPET DOCUMENTATION
        '''
        
        ''' Set Start Date [User-Defined] '''
        self.SetStartDate(2018, 11, 7)   
        ''' Set End Date [User-Defined] '''
        self.SetEndDate(2018, 11, 21)    
        ''' Set Strategy Cash [User-Defined] '''
        self.SetCash(100000)            
        
        '''
        Currencies Selected to Provide Extra Information for Prediction Model [User-Defined]
        '''
        self.symbols = ["NZDUSD", "AUDUSD", "GBPUSD", "EURUSD",] 
        
        
        '''
        Extra Currency Pairs to be Chosen by User for Extra Features [User-Defined]
        NOTE: The Best-Found Combination is the 4 Above.
        NOTE: Add Currency Pair from List Below into 'Self.Extra_Sym'
        NOTE: Overcrowding Currency Pairs May Cause Overfitting

        List of Currency Pairs to Choose From:
            "USDSEK"
            "USDJPY"
            "USDCAD"
            "USDNOK"
            "USDSEK"
            "USDCHF"
            "USDZAR"
        '''
        self.extra_sym = []
        self.symbols = self.symbols + self.extra_sym # Extend Symbols List
        self.sym_n = len(self.extra_sym) # For Dynamically Locating Signals
        
        ''' Target Trading Symbol [User-Defined] '''
        self.trading_symbol = "EURUSD"
        
        for i in self.symbols:
            ''' [User-Defined Resolution] '''
            self.AddForex(i, Resolution.Daily) 
        
        self.long_list = []     # List of Long Positions 
        self.short_list = []    # List of Short Positions
        
        ''' Historical Data Period [User-Defined] '''
        self.data_period = 300 
        ''' Rolling Window Period [User-Defined] '''
        self.window_period = 55

    def OnData(self, data):
        
        data_dict = {}  # Dictionary of Currency & List of Data as Key Pair Values
        price = 0       # Selected Trading Currency's Current Price
        
        # For All Currency Selected for Prediction Model
        for i in self.symbols: 
            
            ''' Download Historical Data [User-Defined Resolution] '''
            currency_slices = self.History([i], self.data_period, Resolution.Daily)
            
            # Get Entire Currency Slice
            currency_bars = currency_slices.loc[i]
            
            ''' 
            Get Attribute (Feature Selection) [User-Defined]
            NOTE: This is a 'Choose-One'
            NOTE: Multi-Attribute Features can be a Future Improvement
            
            Possible Features to Choose From:
                high, low, close
                askopen, askhigh, asklow, askclose
                bidopen, bidhigh, bidlow, bidclose
            '''
            currency_close = currency_bars['close']
            
            # Store in Dictionary
            data_dict[i] = currency_close
            
            # Grab Target Trading Currency's Current Price [User-Defined Currency]
            if i == self.trading_symbol:
                price = currency_close[-1]
         
        '''
        Debug 1: Checking Information is Stored Correctly in Dictionary   
        '''
        #self.Debug("Columns Check: " + str(list(data_dict.keys())))
        #self.Debug("Value Check: " + str(data_dict))        
        
        # Convert Dictionary to DataFrame
        prices_in_df = pd.DataFrame(data_dict, columns = data_dict.keys())
        # Reverse all USD-base currency pairs
        prices_in_df = self.reverseCurr(prices_in_df, self.extra_sym)

        '''
        Debug 2: Checking if Dictionary is Properly Converted to DataFrame
        '''
        #self.Debug("Prices in Dataframe Check: ")
        #self.Debug(prices_in_df.head())
        
        # Calls to Run Ensemble - Decision Tree, Random Forest, Logistic Regression [User-Defined Threshold]
        # Returns a DataFrame of Predicted Signals Based on Selected Features
        '''
        Signals:
            1   -   Take Long Position
            0   -   Hold Position
            -1  -   Take Short Position
        '''
        signals_DT_in_df = self.predict('DecisionTree', prices_in_df, period = self.window_period, threshold = 0.03)
        signals_RF_in_df = self.predict('RandomForest', prices_in_df, period = self.window_period, threshold = 0.03)
        signals_LR_in_df = self.predict('LogisticRegression', prices_in_df, period =  self.window_period, threshold = 0.03)

        '''
        Debug 10: Checking Generated Signals
        '''
        #self.Debug("Generated Signals from DT: ")
        #self.Debug(signals_DT_in_df) 
        #self.Debug("Generated Signals from RF: ")
        #self.Debug(signals_RF_in_df) 
        #self.Debug("Generated Signals from LR: ")
        #self.Debug(signals_LR_in_df) 
            
        # Concatenate all Information into a Master Table
        prices_in_df = prices_in_df[-len(signals_DT_in_df):]
        master_table = pd.concat([prices_in_df, signals_DT_in_df, signals_RF_in_df, signals_LR_in_df], axis = 1).dropna()

        '''
        Debug 11: Checking Master Table is in Order
        '''
        #self.Debug("Master Table Check: ")
        #self.Debug(master_table.head())
        #self.Debug("Current OnData Check: ")
        self.Debug(master_table.tail(1).iloc[:,4:]) # Check Information on Current OnData
        
        if master_table.tail(1).iloc[0][4 + self.sym_n] == 1.0 and \
           master_table.tail(1).iloc[0][5 + self.sym_n] == 1.0 and \
           master_table.tail(1).iloc[0][6 + self.sym_n] == 1.0 and \
           self.trading_symbol not in self.long_list and \
           self.trading_symbol not in self.short_list :
               
            self.SetHoldings(self.trading_symbol, 1)
            self.long_list.append(self.trading_symbol)
            self.Debug("long")
                
        if self.trading_symbol in self.long_list:
            
                cost_basis = self.Portfolio[self.trading_symbol].AveragePrice
                
                if  ((price <= float(0.995) * float(cost_basis)) or (price >= float(1.01) * float(cost_basis))):
                    self.SetHoldings(self.trading_symbol, 0)
                    self.long_list.remove(self.trading_symbol)
                    self.Debug("liquidate long")
        
        if master_table.tail(1).iloc[0][4 + self.sym_n] == -1.0 and \
           master_table.tail(1).iloc[0][5 + self.sym_n] == -1.0 and \
           master_table.tail(1).iloc[0][6 + self.sym_n] == -1.0 and \
           self.trading_symbol not in self.long_list and \
           self.trading_symbol not in self.short_list :
               
            self.SetHoldings(self.trading_symbol, -1)
            self.short_list.append(self.trading_symbol)
            self.Debug("short")
                
        if self.trading_symbol in self.short_list:
            
                cost_basis = self.Portfolio[self.trading_symbol].AveragePrice
                
                if  ((price <= float(0.99) * float(cost_basis)) or (price >= float(1.005) * float(cost_basis))):
                    self.SetHoldings(self.trading_symbol, 0)
                    self.short_list.remove(self.trading_symbol)
                    self.Debug("liquidate short")

    # Reverse the Currency Pair Ratio
    def reverseCurr(self, prices_in_df, extra_sym):
        
        for i in extra_sym:
            prices_in_df[i] = 1/prices_in_df[i]
            
        return prices_in_df

    # Run the Prediction Model
    def predict(self, model, prices_in_df, period, threshold):

        # Number of Rolling Windows
        no_of_windows = int(len(prices_in_df) / period)
        # Total Number of Records Processable (To Prevent Index Out of Bounds)
        length = no_of_windows * period
        
        '''
        Debug 3: Checking all Obtained Parameters are Correct
        '''
        #self.Debug("Model: " + model)
        #self.Debug("Period: " + str(period))
        #self.Debug("Threshold: " + str(threshold))
        #self.Debug("Length of Dataframe: " + str(len(prices_in_df)))
        #self.Debug("No of Windows: " + str(no_of_windows))
        #self.Debug("Length: " + str(length))
        
        # Subsetting Latest N Records that is Within Range
        prices_in_df = prices_in_df[-(length):]
        
        signals = []    # List of Signals
        dates = []      # List of Dates
        
        # For Each Window
        for i in range(0, no_of_windows - 2):
            
            # Retrieve Projection Scores & Weights from Principle Component Analysis
            proj_scores = self.PCA(prices_in_df[(i * period):((i + 3) * period)])  
            
            '''
            Debug 5: Checking Projection Scores
            '''
            #self.Debug("Projection Scores: ")
            #self.Debug(proj_scores.head(5))
        
            # Retrieve Technical Indicators (Extended Features)
            predictor_in_df = self.indicator(period, proj_scores)
            
            '''
            Debug 6: Checking Technical Indicators
            '''
            #self.Debug("Technical Indicators: ")
            #self.Debug(predictor_in_df.head(5))
            
            ''' Get Train & Test Data [User-Defined Sizes] '''
            train_x = predictor_in_df[:int(len(predictor_in_df) / 2)]
            train_y = self.generateSignal(proj_scores[int(len(proj_scores) / 3 - 1):int(len(proj_scores) / 3 * 2)],\
                                          threshold)
            test_x = predictor_in_df[int(len(predictor_in_df) / 2):]
            
            '''
            Debug 8: Checking Test and Train Data
            '''
            #self.Debug("Train X: ")
            #self.Debug(train_x.head(5))      
            #self.Debug("Generated Train Data Signals: ")
            #self.Debug(train_y.head(5))      
            #self.Debug("Test X: ")
            #self.Debug(test_x.head(5))      
            
            # Get List of Predictions
            prediction_in_list = self.classifier(model, train_x, train_y.values.ravel(), test_x)
            
            '''
            Debug 9: Checking Predictions from Ensemble
            '''
            #self.Debug("Predictions from Ensemble: ")
            #self.Debug(prediction_in_list)     
            
            dates.extend(test_x.index)          # Populate List of Dates
            signals.extend(prediction_in_list)  # Populate List of Signals
    
        # Converting List of Signals & Dates into DataFrame
        return pd.DataFrame({'signal': signals}, index = dates)
        
    # Principle Component Analysis        
    def PCA(self, prices_in_df):
        #self.Debug("INSIDE PCA")
        
        # Normalize all Values in DataFrame
        normalized_price = (prices_in_df - prices_in_df.mean())/prices_in_df.std()

        # Obtain Sample Covariance by 
        # 1. Taking 'normalized_price' DataFrame as a Matrix
        # 2. Transposing it
        # 3. Executing Dot Product on Original Against Transposed 'normalized_price'
        # 4. Dividing by Length of Matrix
        covariance = normalized_price.T.dot(normalized_price) / (len(normalized_price) - 1) 
        
        '''
        Debug 4: Checking if Prices are Normalized and Sample Covariance Values Are Obtained Properly
        '''
        #self.Debug("Normalized Price: ")
        #self.Debug(normalized_price.head(5))
        #self.Debug("Covariance: ")
        #self.Debug(covariance.head(5))
        
        # Retrieve Eigen Decomposition of Sample Covariance Matrix
        eigenvalues, eigenvectors = np.linalg.eig(covariance.dropna())
        
        # Retrieve Projection Scores by
        # 1. Taking 'normalized_price' DataFrame and 'eigenvectors[0]' as Matrices
        # 2. Transposing 'eigenvectors[0]'
        # 2. Executing Dot Product on 'normalized_price' Against the Transposed 'eigenvectors[0]' 
        proj_scores = normalized_price.dot(eigenvectors[0].T)
    
        return proj_scores
        
    # Generating Technical Indicators
    def indicator(self, period, proj_scores):

        # Relative position
        Relative_Position = []
        for i in range(period, len(proj_scores)):
            Relative_Position.append((proj_scores[i] - min(proj_scores[(i - period):i])) \
                                    /(max(proj_scores[(i - period):i]) - min(proj_scores[(i - period):i])))
    
        # Relative Strength Index    
        RSI = talib.RSI(np.array(proj_scores), period)
        RSI = RSI[~np.isnan(RSI)]
    
        # Momentum
        MOM = (proj_scores / proj_scores.shift(period)).dropna() * 100
    
        # Moving Average Convergence-Divergence
        MACD_slow = period
        MACD_fast = int(floor(period * 0.5))
        MACD, MACD_signal, MACD_hist = talib.MACDEXT(np.array(proj_scores), fastperiod = MACD_fast, \
                                                     fastmatype = MA_Type.EMA, slowperiod = MACD_slow, \
                                                     slowmatype = MA_Type.EMA, signalperiod = 2, \
                                                     signalmatype = 0)
        MACD = MACD[~np.isnan(MACD)]
    
        # All Technical Indicators as Independent Variables for Prediction Models in DataFrame
        predictor_in_df = pd.DataFrame({'RP': Relative_Position, 'RSI': RSI, 'MOM': MOM, 'MACD': MACD}, \
                                        index = proj_scores.index[period:])
    
        return predictor_in_df
        
    # Generating Response Variables
    def generateSignal(self, proj_scores, threshold):

        signals = [] # List of Signals
        percentage_change = proj_scores.pct_change().dropna() # Percentage Change Among Projection Scores
    
        '''
        Debug 7: Checking % Change of Projection Scores
        '''
        #self.Debug("% Change of Projection Scores: ")
        #self.Debug(percentage_change.head(5))
    
        # Generate Signals Based on Percentage Change of Projection Scores
        '''
        Signals:
            1   -   Take Long Position
            0   -   Hold Position
            -1  -   Take Short Position
        '''
        for i in percentage_change:
            
            if i  > threshold:
                signals.append(1)
            
            elif i < -threshold:
                signals.append(-1)
                
            else:
                signals.append(0)
                
        # Returns a DataFrame Consisting of Signals and Percentage Change Indexes
        return pd.DataFrame({'signal': signals}, index = percentage_change.index)
        
    # Generating Predicted Values from Classifiers
    def classifier(self, model, train_x, train_y, test_x):

        if model == 'LogisticRegression':
            model_execution = LogisticRegression()
            
        if model == 'DecisionTree':
            model_execution = tree.DecisionTreeClassifier()
        
        if model == 'RandomForest':
            model_execution = RandomForestClassifier(n_estimators = 100, max_depth = 2)
            
        model_execution = model_execution.fit(train_x, train_y)
        prediction_in_list = model_execution.predict(test_x)
        
        # Returns a List of Predictions
        return list(prediction_in_list)