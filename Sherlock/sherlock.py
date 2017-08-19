#!/usr/bin/env python

import pandas as pd 
import quandl as data

import math
import numpy as np

from sklearn import preprocessing, model_selection, svm # new format
# from sklearn import preprocessing, cross_validation, svm # old format
from sklearn.linear_model import LinearRegression


# from includes.IBM.examples import text_to_speech_v1 once sign up for paid plan
from includes import talkamaton




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



    def ConvertDataToNegativeOnePlusOneRangeValues(self):
        self.X = preprocessing.scale(self.X) # from scikit module



    def CreateLabelY(self):
        self.y = np.array(self.data_frame['label'])



    # now the training and testing
    def TrainingAndTestingOfFeaturesAndLabels(self):
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(self.X, self.y, test_size=0.2)

    

    def TrainClassifier(self):
        self.clf.fit(self.X_train, self.y_train)



    # now ready to define the classifier - how to pick the best one? trial and error
    # if docs of classifier contain "n_jobs" means threading is supported 
    def UseSVRClassifier(self):
        self.clf = svm.SVR() # Support Vector Regression (many different classifiers are available)
        self.TrainClassifier()



    # supports multithreading 
    def UseLinearRegressionClassifier(self):
        # self.clf = LinearRegression() # regular option
        self.clf = LinearRegression(n_jobs = -1) # threading option
        self.TrainClassifier()



    def UseMultipleClassifiers(self):
        for k in ['linear', 'poly', 'rbf', 'sigmoid']:
            self.clf = svm.SVR(kernel = k)
            self.TrainClassifier()
            self.confidence = self.clf.score(self.X_test, self.y_test)
            print(k, self.confidence)



    def TestConfidence(self):
        self.confidence = self.clf.score(self.X_test, self.y_test)
        print(self.confidence)





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

        # starting on the prediction process of the 'label' based on the 'features'
        # Q: why create dummy place holders and then drop any remanining NaN? Ask
        sherlock.CreateForecastDataColumn()
        # sherlock.DropHolesInData()
        sherlock.CleanData()
        # sherlock.CreateForecastDataColumn() <-- get error when run this after clean data, with too many NaN values

        print('\n\nTurning data into NumPy array format >>> >>>\n\n')
        sherlock.TurnDataIntoCompatibleNumpyArrayFormat()
        # sherlock.ConvertDataToNegativeOnePlusOneRangeValues()

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



        self.talkamaton.Say('Excellent' + '...' + 'Elementary!' )




sherlock = Sherlock()
sherlock.Log()




















