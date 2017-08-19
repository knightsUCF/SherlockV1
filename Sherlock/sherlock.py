#!/usr/bin/env python
import datetime

import pandas as pd 
import quandl as data
data.ApiConfig.api_key = 'insert your api here'
data.ApiConfig.api_version = '2015-04-09'

import math
import numpy as np

from sklearn import preprocessing, model_selection, svm # new format
# from sklearn import preprocessing, cross_validation, svm # old format
from sklearn.linear_model import LinearRegression


# from includes.IBM.examples import text_to_speech_v1 once sign up for paid plan
from includes import talkamaton


import visualization
visualization = visualization.Visualization()

import matplotlib.pyplot as plt 
from matplotlib import style




class Sherlock():

    def __init__(self):
        self.talkamaton = talkamaton.Talkamaton()
        


    def GetSampleData(self, ticker):
        # ticker: "WIKI/GOOGL"
        self.data_frame = data.get(ticker)



    def PrintDataHead(self):
        print(self.data_frame.head())



    def PrintDataTail(self):
        print(self.data_frame.tail())


    
    def CalculateNewColumnBasedOnCrudeVolatility(self, label):
        # use any calculation here, and replace with formula
        # a crude way of calculating volatility: high - low / low 
        self.data_frame[label] = (self.data_frame['Adj. High'] - self.data_frame['Adj. Low']) / self.data_frame['Adj. Close'] * 100.0



    def CalculateNewColumnBasedOnDailyPercentageChange(self, label):
        # use any calculation here, and replace with formula
        # a crude way of calculating volatility: high - low / low 
        self.data_frame[label] = (self.data_frame['Adj. Close'] - self.data_frame['Adj. Open']) / self.data_frame['Adj. Open'] * 100.0


    
    def DefineNewDataFrame(self, featureOne, featureTwo, featureThree, featureFour):
        # label must be attached to data set
        # for this example: Adj. Close, HL_PCT, PCT_change, Adj. Volume
        self.data_frame = self.data_frame[[featureOne, featureTwo, featureThree, featureFour]]



    # cleaning data methods
    def StripData(self):
        self.data_frame = self.data_frame[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
        # now we just have the adjusted columns, and the volume column



    def FillHolesWithDummyPlaceholder(self):
        # fill new column with -99999 place holder
        self.data_frame.fillna(value=-99999, inplace=True)



    def DropHolesInData(self):
        self.data_frame.dropna(inplace=True)



    # def ConvertDataToNegativeOnePlusOneRangeValues() <-- placed after numpy conversion



    def CleanData(self):
        self.FillHolesWithDummyPlaceholder()
        self.DropHolesInData()


    
    def NameForecastDataColumn(self):
        self.forecast_column = 'Adj. Close'



    def ForecastOutOnePercentOfData(self, columnName):
        self.forecast_column = columnName
        # self.forecast_column = 'Adj. Close'
        self.forecast_out = int(math.ceil(0.01 * len(self.data_frame)))
        self.data_frame['label'] = self.data_frame[self.forecast_column].shift(-self.forecast_out)



    def CreateForecastDataColumn(self):
        self.forecast_column = 'Adj. Close'
        # order of tutorial:
        self.FillHolesWithDummyPlaceholder()
        # forecasting out 1 percent of the entire dataset 
        self.forecast_out = int(math.ceil(0.01 * len(self.data_frame)))
        # call the prediction column, label -> features are known data, labels are predictive data
        # add a new column
        self.data_frame['label'] = self.data_frame[self.forecast_column].shift(-self.forecast_out)


    
    def TurnDataIntoCompatibleNumpyArrayFormat(self):
        # define X - features as our entire dataframe EXCEPT for the label column (hence the drop)
        self.X = np.array(self.data_frame.drop(['label'], 1))
        # define y - labels, simply the label column of the dataframe, converted to numpy array
        self.y  = np.array(self.data_frame['label'])



    # replacing above method
    def GetXReadyForProcessing(self):
        # define X - features as our entire dataframe EXCEPT for the label column (hence the drop)
        self.X = np.array(self.data_frame.drop(['label'], 1))
        self.X = preprocessing.scale(self.X)
        self.X_lately = self.X[-self.forecast_out:] # create another x with the most recent features we are going to predict against, notice colon is on other side
        self.X = self.X[:-self.forecast_out]
        self.data_frame.dropna(inplace=True) # not sure if this should be part of this method here



    def GetYReadyForProcessing(self):
        self.y = np.array(self.data_frame['label'])




    def ConvertDataToNegativeOnePlusOneRangeValues(self):
        self.X = preprocessing.scale(self.X) # from scikit module



    def CreateLabelY(self):
        self.y = np.array(self.data_frame['label'])



    # now the training and testing
    def TrainingAndTestingOfFeaturesAndLabels(self):
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(self.X, self.y, test_size=0.2)

    

    def TrainClassifier(self):
        self.clf.fit(self.X_train, self.y_train)



    def TestConfidence(self):
        self.confidence = self.clf.score(self.X_test, self.y_test)
        return self.confidence



    # now ready to define the classifier - how to pick the best one? trial and error
    # if docs of classifier contain "n_jobs" means threading is supported 
    def UseSVRClassifier(self):
        self.clf = svm.SVR() # Support Vector Regression (many different classifiers are available)
        self.TrainClassifier()



    # supports multithreading 
    def UseLinearRegressionClassifier(self):
        # self.clf = LinearRegression() # regular option
        self.clf = LinearRegression(n_jobs=-1) # threading option, this is probably why we're getting different results
        self.TrainClassifier()




    def UseMultipleClassifiers(self):
        # in the future, run through different classifiers and choose the one with the highest R score
        for k in ['linear', 'poly', 'rbf', 'sigmoid']:
            self.clf = svm.SVR(kernel = k)
            self.TrainClassifier()
            # self.confidence = self.clf.score(self.X_test, self.y_test)
            # print(k, self.confidence)
            print(k, self.TestConfidence())
        


    def ForecastSet(self):
        self.forecast_set = self.clf.predict(self.X_lately)
        return self.forecast_set



    def ForecastTable(self):
        print(self.ForecastSet(), self.confidence, self.forecast_out)



    def AddForecastColumn(self):
        self.data_frame['Forecast'] = np.nan



    def AssignEachNewForecastToEachNewDay(self):
        last_date = self.data_frame.iloc[-1].name
        last_unix = last_date.timestamp()
        one_day = 86400
        self.next_unix = last_unix + one_day



    def AddForecastToExistingDataFrame(self):
        for i in self.forecast_set:
            next_date = datetime.datetime.fromtimestamp(self.next_unix)
            self.next_unix += 86400
            self.data_frame.loc[next_date] = [np.nan for _ in range(len(self.data_frame.columns)-1)]+[i]

        # plotting them here for testing
        self.data_frame['Adj. Close'].plot()
        self.data_frame['Forecast'].plot()
        plt.legend(loc=4)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()


    def Visual(self):
        # visualiation.Plot(featureOne, featureTwo, legend, xLabel, yLabel)
        visualization.Plot(self.data_frame['Adj. Close'], self.data_frame['Forecast'], 4, 'Date', 'Price')





    def LitmusTest(self):
        df = data.get("WIKI/GOOGL")
        df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
        df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
        df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
        
        df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
        forecast_col = 'Adj. Close'
        df.fillna(value=-99999, inplace=True)
        forecast_out = int(math.ceil(0.01 * len(df)))
        df['label'] = df[forecast_col].shift(-forecast_out)
        
        X = np.array(df.drop(['label'], 1))
        X = preprocessing.scale(X)
        X = X[:-forecast_out]
        df.dropna(inplace=True)
        y = np.array(df['label'])
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
        
        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)
        confidence = clf.score(X_test, y_test)
        print('\n\nGetting confidence of experiment B\n') 
        print(confidence)








    def Log(self):

        sherlock = Sherlock()
        self.talkamaton.Say('The game\'s afoot!')

        sherlock.GetSampleData("WIKI/GOOGL")
        sherlock.StripData()
        
        # create new data
        sherlock.CalculateNewColumnBasedOnCrudeVolatility('HL_PCT')
        sherlock.CalculateNewColumnBasedOnDailyPercentageChange('PCT_change')
        sherlock.DefineNewDataFrame('Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume')

        # clean data 
        sherlock.FillHolesWithDummyPlaceholder()

        # create the label - predictive data
        sherlock.ForecastOutOnePercentOfData('Adj. Close') # again not sure about the order

        # get ready in NumPy format for SciKit processing 
        sherlock.GetXReadyForProcessing()
        sherlock.GetYReadyForProcessing()
        
        # training and testing
        sherlock.TrainingAndTestingOfFeaturesAndLabels()

        # now can set classifiers
        print('\n\nGetting confidence of experiment A - multiple classifiers\n') 
        # sherlock.UseMultipleClassifiers() # use this one to test multiple ways
        sherlock.UseLinearRegressionClassifier()
        sherlock.TestConfidence()
        # sherlock.TrainClassifier()


        # self.LitmusTest()


        print('Forecast >>> \n\n')
        # sherlock.ForecastTable()

        sherlock.ForecastSet()
        # creating the visual forecast on the chart
        sherlock.AddForecastColumn()
        sherlock.AssignEachNewForecastToEachNewDay()
        sherlock.AddForecastToExistingDataFrame()

        # finally get the visual 
        # sherlock.Visual()
      



        self.talkamaton.Say('Excellent' + '...' + 'Elementary!' )


         











    ''' defunct log
    def Log(self):
        sherlock = Sherlock()
        self.talkamaton.Say('The game\'s afoot!' )
        sherlock.GetSampleData("WIKI/GOOGL")
        sherlock.PrintDataHead()
        sherlock.StripData()
        
        print('\n\nNow printing stripped data >>> >>>\n\n')
        sherlock.PrintDataHead()
        
        print('\n\nNow calculating new column based on volatility delta >>> >>>\n\n')
        sherlock.CalculateNewColumnBasedOnCrudeVolatility('HL_PCT')
        sherlock.PrintDataHead()
        
        print('\n\nNow calculating new column based on daily delta >>> >>>\n\n')
        sherlock.CalculateNewColumnBasedOnDailyPercentageChange('PCT_change')
        sherlock.PrintDataHead()
        
        print('\n\nNow defining a new data frame with four features >>> >>>\n\n')
        sherlock.DefineNewDataFrame('Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume')
        sherlock.PrintDataHead()


        # This order of operations might be problematic
        # starting on the prediction process of the 'label' based on the 'features'
        # Q: why create dummy place holders and then drop any remanining NaN? Ask
        sherlock.CreateForecastDataColumn()
        # sherlock.DropHolesInData()
        sherlock.CleanData()
        # sherlock.CreateForecastDataColumn() <-- get error when run this after clean data, with too many NaN values
        

        # name forecast column
        # sherlock.NameForecastDataColumn() # preferrably should have name parameter

        # fill na with -99999
        sherlock.FillHolesWithDummyPlaceholder()

        # forecast out 1 percent
        # sherlock.ForecastOutOnePercentOfData()

        # recheck order, if breaks 
        sherlock.ForecastOutOnePercentOfData('Adj. Close') # again not sure about the order

        # convert to numpy array
        sherlock.GetXReadyForProcessing()
        sherlock.GetYReadyForProcessing()

        # testing and training
        sherlock.TrainingAndTestingOfFeaturesAndLabels()

        # now ready to define classifier
        sherlock.UseMultipleClassifiers()
        sherlock.TrainClassifier()

        # check results 
        sherlock.TestConfidence()


        print('\n\nTurning data into NumPy array format >>> >>>\n\n')
        sherlock.TurnDataIntoCompatibleNumpyArrayFormat()
        sherlock.ConvertDataToNegativeOnePlusOneRangeValues()

        sherlock.CreateLabelY()
        
        # now the training and testing
        print('\n\nTraining and testing... >>> >>>\n\n')
        sherlock.TrainingAndTestingOfFeaturesAndLabels()


        # now ready to define the classifier
        print('\n\nDefining and training classifier >>> >>>\n\n')
        # sherlock.UseSVRClassifier()
        # sherlock.UseLinearRegressionClassifier()
        sherlock.UseMultipleClassifiers()
        # sherlock.TrainClassifier()

        
        print('\n\nTesting confidence >>> >>>\n\n')
        sherlock.TestConfidence()
        # print(self.confidence)


        # review why code is off before moving on to part 5
        # https://pythonprogramming.net/forecasting-predicting-machine-learning-tutorial/?completed=/training-testing-machine-learning-tutorial/



        self.talkamaton.Say('Excellent' + '...' + 'Elementary!' )

        # defunct log '''


sherlock = Sherlock()
sherlock.Log()




import math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime

style.use('ggplot')

df = data.get("WIKI/GOOGL")
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()











