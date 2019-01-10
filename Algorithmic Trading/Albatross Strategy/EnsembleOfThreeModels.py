# Team members:
# Lim Jean Teng Rene, A0149965J
# Ng Jian Lai, A0155721J
# Soh Jun Xuan Benedict, A0171412U

import numpy as np
from numpy.random import seed
import decimal
import random

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor

from keras.models import Sequential

from keras.layers import Dense, Activation, LSTM, Dropout
from keras.utils import to_categorical
from keras import optimizers
from keras import metrics

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit


import pandas as pd


class Forex_Ensemble(QCAlgorithm):
    
#####  34 to 64:  Initialization of Algo ####
    def Initialize(self):
  
        #self.Debug("START: Initialize")
        self.SetStartDate(2018,11,7)    #Set Start Date
        self.SetEndDate(2018,11,21)     #Set End Date
        self.SetCash(100000)           #Set Strategy Cash
        self.SetBrokerageModel(BrokerageName.OandaBrokerage, AccountType.Cash) #Set Brokerage Model
        
        self.currency = "EURUSD" # We are going to invest in this
        self.variable1 = "GBPUSD" # 2 variables to help predict movement of EURUSD
        self.variable2 = "EURJPY"
        
        # Add the relevant currencies
        self.AddForex(self.currency, Resolution.Hour)
        self.AddForex(self.variable1, Resolution.Hour)
        self.AddForex(self.variable2, Resolution.Hour)

        # Helper variables for logistic regression
        self.long_list =[] # To ensure that we only buy once per signal/model
        self.short_list =[]
        self.model = LogisticRegression()
        self.model_LSTM = Sequential()
        self.x=0            # To ensure that our model only build once per currency
        
        # Hyper parameters for random forest
        self.lookback = 3           # Number of days to lookback to use as input for model
        self.history_range = 300    # Number of days of past data to look at
        self.model_forest = RandomForestRegressor()
        
        self.decision = [] # Decision to buy or to sell
        #self.Debug("End: Initialize")

#####  67 to 80 : Defining OnData function and Geting the Historical Data  ####
    def OnData(self, data):
        
        #self.Debug("START: Ondata")
        
        # Getting historical data
        currency_data  = self.History([self.currency], 500, Resolution.Hour)
        currency_data1 = self.History([self.variable1], 500, Resolution.Hour)
        currency_data2 = self.History([self.variable2], 500, Resolution.Hour)
        
        # Checking length of data to ensure we got the correct amount of data to be compared
        L= len(currency_data)
        L1= len(currency_data1)
        L2= len(currency_data2)
        #self.Debug("The length is " + str (L))
    
#####  83 to 172 : Check condition for required data and prepare X and Y for modelling  ####    
        # Making sure that the data is not empty and then only proceed with the algorithm
        if (not currency_data.empty and  not currency_data1.empty and  not currency_data2.empty and L == L1 ==L2): 
            
            ################################# Logistic Regression #################################
            
            # Getting close prices and storing into a dataframe
            data = pd.DataFrame(currency_data.close)
            data1 = pd.DataFrame(currency_data1.close) 
            data2 = pd.DataFrame(currency_data2.close)
            
            # Data Preparation for input to Logistic Regression
            stored = {} # To prepare and store data
            for i in range(11): # For getting 10 lags ...Can be increased if more lags are required
                stored['currency_lag_{}'.format(i)] = data.shift(i).values[:,0].tolist() #creating lags
                stored['variable1_lag_{}'.format(i)] = data1.shift(i).values[:,0].tolist()
                stored['variable2_lag_{}'.format(i)] = data2.shift(i).values[:,0].tolist()
            
            # Convert to a dataframe
            stored = pd.DataFrame(stored)
            stored = stored.dropna()
            stored = stored.reset_index(drop=True)
            
            # Get the percent change from previous time
            stored["Y"] = stored["currency_lag_0"].pct_change() 
            
            for i in range(len(stored)): # loop to make Y as categorical
                if stored.loc[i,"Y"] > 0:       # If the percentage change is positive, meaning a rise in price
                    stored.loc[i,"Y"] = "UP"    # We store as "UP"
                else:
                    stored.loc[i,"Y"] = "DOWN"  # Else, means price is going down
                    
            #self.Debug("All X_data is " +str(stored))    
            
            X_data = stored.iloc[:,np.r_[3:33]] # Extract only lag1, Lag2, lag3.. 
                                                # As Lag 0 is the data itself and will  not be available
                                                # during prediction
            #self.Debug( "X data is" +str(X_data))
            
            Y_data = stored["Y"]
            #self.Debug( "Y data is" +str(Y_data))
            
            ################################# End of logistic regression #################################
            
            
            ################################# LSTM #################################
            
            data_LSTM = np.array([currency_data.close])  # Get the close prices and make an array
            self.Debug("Close prices after making an array" + str(data))
            
            #Data Preparation for input to LSTM
            X1_LSTM = data_LSTM[:,0:L-5] #(0 to 5 data)
            X2_LSTM = data_LSTM[:,1:L-4] #(1 to 6 data)
            X3_LSTM = data_LSTM[:,2:L-3] #(#2 to 7 data) 
            X4_LSTM = data_LSTM[:,3:L-2] #(#3 to 8 data)
            X5_LSTM = data_LSTM[:,4:L-1] #(#4 to 9 data)
        
            X_LSTM= np.concatenate([X1_LSTM,X2_LSTM,X3_LSTM,X4_LSTM,X5_LSTM ],axis=0)

            X_data_LSTM= np.transpose(X_LSTM) 
        
            Y_data_LSTM = np.transpose(data_LSTM[:,5:L]) 

            
            #Normalize the data 
            scaler_LSTM_X = MinMaxScaler() 
            scaler_LSTM_X.fit(X_data_LSTM)
            X_data_LSTM = scaler_LSTM_X.transform(X_data_LSTM)
            self.Debug("X after transformation is " + str(X_data_LSTM))
         
            scaler_LSTM_Y = MinMaxScaler()
            scaler_LSTM_Y.fit(Y_data_LSTM)
            Y_data_LSTM = scaler_LSTM_Y.transform(Y_data_LSTM)
            self.Debug("Y after transformation is " + str(Y_data_LSTM))
            
            ################################# End of LSTM #################################
            
            ################################# Random forest #################################

            price_changes = np.diff(currency_data.close).tolist() # Find the change in price daily

            X = []  # List to store independent variable
            Y = []  # List to store dependent variable
            
            # Loop through all the past data and store them in batches of 3 days
            for i in range(self.history_range-self.lookback-1):
                X.append(price_changes[i:i+self.lookback])
                Y.append(price_changes[i+self.lookback-1])
            
            ################################# End of random forest  #################################

#####  175 to 222 : Build the model, check the training accuracy and coefficients  ####     
            if self.x==0:  #To make sure the model is build only once and avoid computation at every new data
            
                ################################# Logistic regression #################################
                
                # Normalize data
                scaler = MinMaxScaler() 
                scaler.fit(X_data)
                X_data = scaler.transform(X_data)
                
                self.model.fit(X_data,Y_data)
                score = self.model.score(X_data, Y_data)
                #self.Debug("Train Accuracy of final model: " + str(score))
                
                # To get the coefficients from model
                #A = pd.DataFrame(X_data.columns)
                #B = pd.DataFrame(np.transpose(self.model.coef_))
                #C =pd.concat([A,B], axis = 1)
                #self.Debug("The coefficients are: "+ str(C))
                
                ################################# End of logistic regression #################################
            
                ################################# LSTM #################################
                X_data1_LSTM= np.reshape(X_data_LSTM, (X_data_LSTM.shape[0],1,X_data_LSTM.shape[1]))
                O_cells = 100
                O_epochs = 200
                O_dropout = 0.15
                O_opt = 'Adagrad'
                
                
                self.model_LSTM.add(LSTM(O_cells, input_shape = (1,5), return_sequences = True))
                self.model_LSTM.add(Dropout(O_dropout))
                #self.model.add(LSTM(O_cells,return_sequences = True))
                self.model_LSTM.add(LSTM(O_cells))
                self.model_LSTM.add(Dropout(O_dropout))
                self.model_LSTM.add(Dense(1))
                self.model_LSTM.compile(loss= 'mean_squared_error',optimizer = O_opt, metrics = ['mean_squared_error'])
                self.model_LSTM.fit(X_data1_LSTM,Y_data_LSTM,epochs=O_epochs,verbose=0)
                self.Debug("END: Final_LSTM Model")
                
                ################################# End of LSTM #################################
                
                ################################# Random forest #################################
                self.model_forest = RandomForestRegressor()
                self.model_forest.fit(X,Y)
                        
                ################################# End of random forest #################################
            
            self.x=1     # End the model
   
#####  225 to 272 : Prepare data for prediction   ####             
            ################################# Logistic regression #################################
            
            # Prepare test data similar way as earlier
            test = {}
            for i in range(10):
                test['currency_lag_{}'.format(i+1)] = data.shift(i).values[:,0].tolist()
                test['variable1_lag_{}'.format(i+1)] = data1.shift(i).values[:,0].tolist()
                test['variable2_lag_{}'.format(i+1)] = data2.shift(i).values[:,0].tolist()
    
            test = pd.DataFrame(test)
            test = pd.DataFrame(test.iloc[-1, :]) # Take the last values 
            test = pd.DataFrame(np.transpose(test)) # Transpose to get in desired model shape
            
            ################################# End of logistic regression #################################
            
            
            ################################# LSTM #################################
            
            data = np.array([currency_data.close])
            
            X1_new_LSTM = data[:,-5]
            #self.Debug(X1_new)
            X2_new_LSTM = data[:,-4]
            #self.Debug(X2_new)
            X3_new_LSTM = data[:,-3]
            #self.Debug(X3_new)
            X4_new_LSTM = data[:,-2]
            #self.Debug(X3_new)
            X5_new_LSTM = data[:,-1]
            #self.Debug(X3_new)
            X_new_LSTM= np.concatenate([X1_new_LSTM,X2_new_LSTM,X3_new_LSTM,X4_new_LSTM,X5_new_LSTM],axis=0)
            X_new_LSTM= np.transpose(X_new_LSTM)
            #self.Debug(X_new)
            scaler = MinMaxScaler() 
            scaler.fit(X_data_LSTM)
            X_new_LSTM = scaler.transform([X_new_LSTM])
            X_new_LSTM= np.reshape(X_new_LSTM,(X_new_LSTM.shape[0],1,X_new_LSTM.shape[1]))
            
            ################################# End of LSTM #################################
            
            ################################# Random forest #################################
            
            # When the model is ready
            pc = price_changes[-3:]
                    
            ################################# End of random forest #################################

#####  276 to 331 : Make Prediction   #### 
            
            self.decision = [] # Clear the decision list
            self.see = []
            
            ################################# Logistic regression #################################

            # Normalize data
            scaler = MinMaxScaler() 
            scaler.fit(test)
            test = scaler.transform(test)
            
            output = self.model.predict(test)
            self.Debug("Output from LR model is" + str(output))
            if output == "UP":
                self.decision.append("UP")
                self.Debug("log reg")
            else:
                self.decision.append("DOWN")
                self.see.append("Down_LOG")

            
            # Checking the current price for shorting
            price = currency_data.close[-1]
            
            #self.Debug("Current price is" + str(price))
            
            # Make decision for trading based on the output from LR and the current price.
            # If output (forecast) is UP we will buy the currency; else, Short.
            # Only one trade at a time and therefore made a list " self.long_list". 
            # As long as the currency is in that list, no further buying can be done.
            # Risk and Reward are defined: Exit the trade at 1% loss or 1 % profit.
 
            
            ################################# End of logistic regression #################################

            ################################# LSTM #################################


            Predict = self.model_LSTM.predict(X_new_LSTM)
            output = scaler_LSTM_Y.inverse_transform(Predict)
            
            price = currency_data.close[-1]
            output = output-price
            if output>1:
                self.decision.append("UP")
                self.Debug("lstm")
            else:
                self.decision.append("DOWN")
                self.see.append("Down_LSTM")
            
            ################################# End of LSTM #################################
                        
            ################################# Random forest #################################
            
            prediction = self.model_forest.predict([pc])
            if prediction > 0:      # Positive return
                self.decision.append("UP")
                self.Debug("randomforest")
            else:                   # Negative return
                self.decision.append("DOWN")
                self.see.append("Down_RF")
            
            ################################# End of random forest #################################

##### 335 to 382 : Entry /Exit Conditions for trading  #### 
            
            # Differential investment by voting 
            # 3 "UP" we buy currency with 90% of holding
            # 2 "UP" we buy with 80%
            # 1 "UP" we buy with 70%
            numUp = 0
            numDown = 0
            for i in range(len(self.decision)):
                if self.decision[i] == "UP":
                    numUp += 1
                else:
                    numDown += 1
            self.Debug("numUp" + str(numUp))
            self.Debug("numDown" + str(numDown))
            for entry in self.see:
                self.Debug(entry)

            if numUp ==3  and self.currency not in self.long_list and self.currency not in self.short_list :
                
                #self.Debug("output is greater")
                # Buy the currency with X% of holding in this case 90%
                self.SetHoldings(self.currency, 0.9)
                self.long_list.append(self.currency)
                #self.Debug("long")
                
            if numUp ==2  and self.currency not in self.long_list and self.currency not in self.short_list :
                
                #self.Debug("output is greater")
                # Buy the currency with X% of holding in this case 90%
                self.SetHoldings(self.currency, 0.8)
                self.long_list.append(self.currency)
                #self.Debug("long")
                
            if numUp ==1  and self.currency not in self.long_list and self.currency not in self.short_list :
                
                #self.Debug("output is greater")
                # Buy the currency with X% of holding in this case 90%
                self.SetHoldings(self.currency, 0.7)
                self.long_list.append(self.currency)
                #self.Debug("long")
                
                
            if self.currency in self.long_list:
                cost_basis = self.Portfolio[self.currency].AveragePrice
                #self.Debug("cost basis is " +str(cost_basis))
                if  ((price <= float(0.995) * float(cost_basis)) or (price >= float(1.01) * float(cost_basis))):
                    #self.Debug("SL-TP reached")
                    #self.Debug("price is" + str(price))
                    #If true then sell
                    self.SetHoldings(self.currency, 0)
                    self.long_list.remove(self.currency)
                    #self.Debug("squared long")
                    
                    
            if numDown >= 2  and self.currency not in self.long_list and self.currency not in self.long_list:
                
                #self.Debug("output is lesser")
                self.SetHoldings(self.currency, -0.9)
                self.short_list.append(self.currency)
                #self.Debug("short")
                
            if self.currency in self.short_list:
                cost_basis = self.Portfolio[self.currency].AveragePrice
                #self.Debug("cost basis is " +str(cost_basis))
                if  ((price <= float(0.99) * float(cost_basis)) or (price >= float(1.005) * float(cost_basis))):
                    #self.Debug("SL-TP reached")
                    #self.Debug("price is" + str(price))
                    #If true then buy back
                    self.SetHoldings(self.currency, 0)
                    self.short_list.remove(self.currency)
                    #self.Debug("squared short")

            #self.Debug("END: Ondata")
