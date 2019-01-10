import numpy as np
from numpy.random import seed

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

import pandas as pd


class Forex_Trade_Logreg(QCAlgorithm):
    
    
#####  17 to 34:  Initialization of Algo ####
    def Initialize(self):
  
        #self.Debug("START: Initialize")
        self.SetStartDate(2018,10,15)    #Set Start Date 
        self.SetEndDate(2018,10,30)     #Set End Date
        self.SetCash(100000)           #Set Strategy Cash
        self.SetBrokerageModel(BrokerageName.OandaBrokerage, AccountType.Margin) #Set Brokerage Model
        self.currency = "EURUSD"
        self.variable1 = "GBPJPY"   
        self.variable2 = "USDJPY"
        self.variable3 = "USDCHF"       
        self.AddForex(self.currency,Resolution.Hour)       
        self.AddForex(self.variable1,Resolution.Hour)
        self.AddForex(self.variable2,Resolution.Hour)
        self.AddForex(self.variable3,Resolution.Hour)      
        self.long_list =[]
        self.short_list =[]
        self.model = LogisticRegression(penalty='l1') 
        self.x=0
        self.modelOutput = "NONE"
        self.actualOutput = "NONE"
        self.ystdPrice = 0
        self.tdayPrice = 0
        self.firstRound = True
        self.wrongModel = False
        self.nextDay = True
        self.todayDay = None
        self.rsi = self.RSI(self.currency, 14, MovingAverageType.Wilders, Resolution.Hour)
        
        #self.Debug("End: Initialize")

#####  37 to 48 : Defining OnData function and Geting the Historical Data  ####
    def OnData(self, data): #This function runs on every resolution of data mentioned. 
                            
        todayDay = str(self.Time)[8:10] #e.g. if date is 2018-10-15 05:00:00, this returns 15
        if todayDay != self.todayDay: #this simply checks if we have passed into the next day
            self.nextDay = True
            self.todayDay = todayDay
        else:
            self.nextDay= False
        
        currency_data  = self.History([self.currency], 50, Resolution.Daily) # Asking for historical data
        currency_data1 = self.History([self.variable1], 50, Resolution.Daily)
        currency_data2 = self.History([self.variable2], 50, Resolution.Daily)
        currency_data3 = self.History([self.variable3], 50, Resolution.Daily) 
        
        
        
        L= len(currency_data) # Checking the length of data
        L1= len(currency_data1)
        L2= len(currency_data2)
        L3= len(currency_data3) 
        #self.Debug("The length is " + str (L))
    
#####  72 to 114 : Check condition for required data and prepare X and Y for modeling  ####    
        # Making sure that the data is not empty and then only proceed with the algo
        if ( not currency_data.empty and  not currency_data1.empty and  not currency_data2.empty and  not currency_data3.empty and L == L1 ==L2 ==L3): 
            #New condition above
            
            data = pd.DataFrame(currency_data.close)  #Get the close prices. Also storing as dataframe
            data1 = pd.DataFrame(currency_data1.close) 
            data2 = pd.DataFrame(currency_data2.close)
            data3 = pd.DataFrame(currency_data3.close)  
            
            
            
            #Data Preparation for input to Logistic Regression
            stored = {} # To prepare and store data
            for i in range(11): # For getting 10 lags 
                stored['EURUSD_lag_{}'.format(i)] = data.shift(i).values[:,0].tolist() #creating lags
                stored['GBPJPY_lag_{}'.format(i)] = data1.shift(i).values[:,0].tolist()
                stored['USDJPY_lag_{}'.format(i)] = data2.shift(i).values[:,0].tolist()
                stored['USDCHF_lag_{}'.format(i)] = data3.shift(i).values[:,0].tolist() 
    
            stored = pd.DataFrame(stored)
            
            stored = stored.dropna() # drop na values
            stored = stored.reset_index(drop=True)
            
            
            stored["Y"] = stored["EURUSD_lag_0"].pct_change()# get the percent change from previous time
            
            for i in range(len(stored)): # loop to make Y as categorical
                if stored.loc[i,"Y"] > 0:
                    stored.loc[i,"Y"] = "UP"
                else:
                    stored.loc[i,"Y"] = "DOWN"
                    
            #self.Debug("All X_data is " +str(stored))    
            
            
            #self.Debug(stored)
            X_data = stored.iloc[:,np.r_[4:44]]  # extract only lag1, Lag2, lag3.. As Lag 0 is the data itself and will  not be available during prediction
            #self.Debug( "X data is: ")
            #self.Debug(str(X_data))
            
            Y_data = stored["Y"]
            #self.Debug( "Y data is: ")
            #self.Debug(str(Y_data))
            
#####  118 to 134 : Build the Logistic Regression model, check the training accuracy and coefficients  ####     
            if self.x==0:  #To make sure the model is build only once and avoid computation at every new data
                
                self.model.fit(X_data,Y_data)
                score = self.model.score(X_data, Y_data)
                self.accuracy = score
                self.Debug("First time training model: ")
                self.Debug(self.model)
                self.Debug("Train Accuracy of final model: " + str(score))
                
                # To get the coefficients from model
                A = pd.DataFrame(X_data.columns)
                B = pd.DataFrame(np.transpose(self.model.coef_))
                C = pd.concat([A,B], axis = 1)
                #self.Debug("The coefficients are: "+ str(C))
                
            self.x=1     # End the model
            
#####  139 to 149 : Prepare data for prediction   ####             
            
            #Prepare test data similar way as earlier
            test = {}
            
            for i in range(10):
                test['EURUSD_lag_{}'.format(i+1)] = data.shift(i).values[:,0].tolist()
                test['GBPJPY_lag_{}'.format(i+1)] = data1.shift(i).values[:,0].tolist()
                test['USDJPY_lag_{}'.format(i+1)] = data2.shift(i).values[:,0].tolist()
                test['USDCHF_lag_{}'.format(i+1)] = data3.shift(i).values[:,0].tolist() #New line
    
            test = pd.DataFrame(test)
            test = pd.DataFrame(test.iloc[-1, :]) # take the last values 
            test = pd.DataFrame(np.transpose(test)) # transpose to get in desired model shape
            #self.Debug(test)
    
#####  154 to 155 : Make Prediction   #### 
    
            output = self.model.predict(test)
            self.Debug("Output from LR model is " + str(output))
            
#####  157 to 283 : Reinforcement learning. Check if model made right prediction   ####
            #Checking the current price 
            price = currency_data.close[-1]
            self.tdayPrice = price
            self.Debug("Today price is " + str(price))
            
            
            #We need at least one day to determine if the model has predicted accurately
            if not self.firstRound and self.nextDay:             #We have more than 1 day's worth of data, can compare with previous day
                if self.tdayPrice >= self.ystdPrice:
                    self.actualOutput = "UP"
                elif self.tdayPrice < self.ystdPrice:
                    self.actualOutput = "DOWN"
                #self.Debug("Ystd Price: " + str(self.ystdPrice))
                
                self.ystdPrice = self.tdayPrice
                if self.actualOutput != self.modelOutput:   #check if our model predicted correctly

                    self.wrongModel=True
                    #self.modelOutput=output
                    
                    if self.wrongModel:
                        self.Debug("Retrain!")
                        # Model predicted wrongly! 
                        # We need to retrain with new data!
                        # We cannot use the current model to make a prediction for the next day's closing price!
                        # Need a new model now: retrain immediately
                        new_model = LogisticRegression(penalty="l1") #New model!
                        currency_data  = self.History([self.currency], 50, Resolution.Daily) # Asking for historical data
                        currency_data1 = self.History([self.variable1], 50, Resolution.Daily)
                        currency_data2 = self.History([self.variable2], 50, Resolution.Daily)
                        currency_data3 = self.History([self.variable3], 50, Resolution.Daily) #New line
                        
                        L= len(currency_data) # Checking the length of data
                        L1= len(currency_data1)
                        L2= len(currency_data2)
                        L3= len(currency_data3) 
                        
                    
                
                        # Making sure that the data is not empty and then only proceed with the algo
                        if ( not currency_data.empty and  not currency_data1.empty and  not currency_data2.empty and  not currency_data3.empty and L == L1 ==L2 ==L3): 
                            
                            data = pd.DataFrame(currency_data.close)  #Get the close prices. Also storing as dataframe
                            data1 = pd.DataFrame(currency_data1.close) 
                            data2 = pd.DataFrame(currency_data2.close)
                            data3 = pd.DataFrame(currency_data3.close)  
                            
                            
                            
                            #Data Preparation for input to Logistic Regression
                            stored = {} # To prepare and store data
                            for i in range(11): # For getting 10 lags
                                stored['EURUSD_lag_{}'.format(i)] = data.shift(i).values[:,0].tolist() #creating lags
                                stored['GBPJPY_lag_{}'.format(i)] = data1.shift(i).values[:,0].tolist()
                                stored['USDJPY_lag_{}'.format(i)] = data2.shift(i).values[:,0].tolist()
                                stored['USDCHF_lag_{}'.format(i)] = data3.shift(i).values[:,0].tolist() #New line
                    
                            stored = pd.DataFrame(stored)
                            
                            stored = stored.dropna() # drop na values
                            stored = stored.reset_index(drop=True)
                            
                            
                            stored["Y"] = stored["EURUSD_lag_0"].pct_change()# get the percent change from previous time
                            
                            for i in range(len(stored)): # loop to make Y as categorical
                                if stored.loc[i,"Y"] > 0:
                                    stored.loc[i,"Y"] = "UP"
                                else:
                                    stored.loc[i,"Y"] = "DOWN"
                                    
                            #self.Debug("All X_data is " +str(stored))    
                            
                            
                            X_data = stored.iloc[:,np.r_[4:44]]  
                            #self.Debug( "X data is: ")
                            #self.Debug(str(X_data))
                            
                            Y_data = stored["Y"]
                            #self.Debug( "Y data is: ")
                            #self.Debug(str(Y_data))
                            
                            new_model.fit(X_data,Y_data)
                            newScore = new_model.score(X_data, Y_data)
                            #self.Debug("Train Accuracy of final model: " + str(score))
                            
                            # To get the coefficients from model
                            A = pd.DataFrame(X_data.columns)
                            B = pd.DataFrame(np.transpose(self.model.coef_))
                            C = pd.concat([A,B], axis = 1)
                            #self.Debug("The coefficients are: "+ str(C))
                                    
                            #self.Debug("New model after retraining: ")
                            #self.Debug(new_model)
                            
                            
                            self.model = new_model
                            self.Debug("**************replaced old model*********************")
                   
                            
                            #Prepare test data 
                            test = {}
                            
                            for i in range(10):
                                test['EURUSD_lag_{}'.format(i+1)] = data.shift(i).values[:,0].tolist()
                                test['GBPJPY_lag_{}'.format(i+1)] = data1.shift(i).values[:,0].tolist()
                                test['USDJPY_lag_{}'.format(i+1)] = data2.shift(i).values[:,0].tolist()
                                test['USDCHF_lag_{}'.format(i+1)] = data3.shift(i).values[:,0].tolist() #New line
                    
                            test = pd.DataFrame(test)
                            test = pd.DataFrame(test.iloc[-1, :]) # take the last values 
                            test = pd.DataFrame(np.transpose(test)) # transpose to get in desired model shape
                            #self.Debug(test)
                            self.modelOutput = self.model.predict(test)
                            self.wrongModel = False
                    else:
                        #else if our model predicted correctly, we will continue to use our model prediction
                        self.modelOutput = output
            elif self.firstRound:
                #this checks if its the first day of the algo
                #We need at least one day of actual data, before we can decide if our model made a right decision or not the next day
                self.Debug("First day over")
                self.firstRound = False
                self.ystdPrice = self.tdayPrice
                self.modelOutput = output
            
            
            #Make decision for trading based on the output from LR and the current price.
            #If output ( forecast) is UP we will buy the currency; else, Short.
            # Only one trade at a time and therefore made a list " self.long_list". 
            #As long as the currency is in that list, no further buying can be done.
            # Risk and Reward are defined:
                #Long position: Stop loss at 0.5% below cost_basis, take profit at 1% above cost basis
                #Short position: Stop loss at 0.5% above cost_basis, take profit at 1% below cost basis
 
##### 293 to 333 : Entry /Exit Conditions for trading  #### 
            self.Debug(str(self.Time))
            #self.Debug(self.modelOutput)
            if self.modelOutput == "UP"  and self.currency not in self.long_list and self.currency not in self.short_list and self.rsi.Current.Value < 30:
                
                self.Debug("output is greater")
                #At least 70% invested on average each NY business day.
                self.SetHoldings(self.currency, 0.9)
                self.long_list.append(self.currency)
                self.Debug("long")
                
            if self.currency in self.long_list:
                cost_basis = self.Portfolio[self.currency].AveragePrice
                #self.Debug("cost basis is " +str(cost_basis))
                if  ((price <= float(0.995) * float(cost_basis)) or (price >= float(1.01) * float(cost_basis))):
                    self.Debug("SL-TP reached")
                    #self.Debug("price is" + str(price))
                    #If true then sell
                    self.SetHoldings(self.currency, 0)
                    self.long_list.remove(self.currency)
                    self.Debug("squared long")
                    
                    
            if self.modelOutput =="DOWN"  and self.currency not in self.long_list and self.currency not in self.short_list and self.rsi.Current.Value > 70:
                
                self.Debug("output is lesser")
                # Sell the currency with X% of holding in this case 90%
                self.SetHoldings(self.currency, -0.9)
                self.short_list.append(self.currency)
                self.Debug("short")
                
            if self.currency in self.short_list:
                cost_basis = self.Portfolio[self.currency].AveragePrice
                #self.Debug("cost basis is " +str(cost_basis))
                if  ((price <= float(0.99) * float(cost_basis)) or (price >= float(1.005) * float(cost_basis))):
                    self.Debug("SL-TP reached")
                    #self.Debug("price is" + str(price))
                    #If true then buy back
                    self.SetHoldings(self.currency, 0)
                    self.short_list.remove(self.currency)
                    self.Debug("squared short")
