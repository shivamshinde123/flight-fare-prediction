import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.simplefilter(action='ignore', category=FutureWarning)

from Logging.logging import Logger


class modelTuner:

    """
    Description: This class contains  the methods which will be used to find the model with the highest accuracy

    Written By: Shivam Shinde

    Version: 1.0

    Revision: None
    """

    def __init__(self):
        self.logger = Logger()
        self.file_obj = open("../TrainingLogs/bestModelFindingLogs.txt","a+")
        self.ridge = Ridge()
        self.rfr = RandomForestRegressor()
        self.svr = SVR()
        self.xgb = XGBRegressor()

    def r2_adjusted_score(self,r,data):

        """
        Description: This method is used to find the r2 adjusted score using the r2 score of the model

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param data: independent feature data

        :return: r2-adjusted score
        """
        try:
            # finding the number of rows in the data
            n = data.shape[0]

            # finding the number of columns in the data
            m = data.shape[1]

            # finding the r2-adjusted score
            result = 1 - ((1 - r**2)* ((n -1 )/(n-m-1)))

            # returning the obtained result
            return result

        except Exception as e:
            self.logger.log(self.file_obj, f"Exception occurred while calculating the r2 adjusted score. Exception: {str(e)}")
            raise e

    def tuneRidge(self,xtrain,ytrain):

        """
        Description: This method is used to tune the hyperparameters of the elastic net regressor.

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param xtrain: training independent feature data
        :param ytrain: training dependent feature data

        :return: tuned elastic net model
        """

        try:
            self.logger.log(self.file_obj,
                            "Finding the best hyperparameters for the elastic net machine learning model")

            # creating a dictionary containing the possible options for the hyperparameters
            best_parameters = dict()
            best_parameters['alpha'] = np.arange(0, 11, 1)
            best_parameters['solver'] = ['auto', 'svd','lsqr','sparse_cg','sag','saga']

            # performing the grid search cv on the elastic net model using the earlier created dictionary
            rr_grid = RandomizedSearchCV(self.ridge, best_parameters, cv=5, n_jobs=-1)
            rr_grid.fit(xtrain, ytrain)

            # finding the best hyperparameters obtained through the grid search cv
            alpha = rr_grid.best_params_["alpha"]
            solver = rr_grid.best_params_["solver"]

            # creating a ridge regression model using the best parameters obtained earlier
            rr = Ridge(alpha=alpha, solver=solver, random_state=90345)
            rr.fit(xtrain, ytrain)

            # returning the ridge regression model
            return rr

        except Exception as e:
            self.logger.log(self.file_obj, "Exception occurred while finding the best hyperparameters for the ridge "
                                           "regression machine learning model")
            raise e


    def tuneRandomForestRegressor(self,xtrain,ytrain):

        """
        Description: This method is used to tune the hyperparameters of the random forest regressor

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param xtrain: training independent feature data
        :param ytrain: training dependent feature data

        :return: tuned random forest regressor model
        """

        try:
            self.logger.log(self.file_obj, "Finding the best hyperparameters for the random forest regressor machine learning model")

            # creating a dictionary containing the possible values of the hyperparameters
            best_parameters = dict()
            best_parameters['n_estimators'] = np.arange(100, 600, 100)
            best_parameters['criterion'] = ['squared_error', 'absolute_error', 'poisson']
            best_parameters['max_depth'] = [2, 3]
            best_parameters['max_features'] = ['auto', 'sqrt', 'log2']
            best_parameters['bootstrap'] = [True, False]

            # performing the grid search cv using the hyperparameter dictionary created earlier
            rfr_grid = RandomizedSearchCV(self.rfr, best_parameters, cv=5, n_jobs=-1)
            rfr_grid.fit(xtrain,ytrain)

            # getting the best hyperparameters for the random forest regressor using the fitted rfr_grid
            n_estimators = rfr_grid.best_params_['n_estimators']
            criterion = rfr_grid.best_params_['criterion']
            max_depth = rfr_grid.best_params_['max_depth']
            max_features = rfr_grid.best_params_['max_features']
            bootstrap = rfr_grid.best_params_['bootstrap']

            # creating a random forest regressor using the best hyperparameters obtained earlier
            randomforestreg = RandomForestRegressor(n_estimators=n_estimators,criterion=criterion,max_depth=max_depth,
                                             max_features=max_features,bootstrap=bootstrap,n_jobs=-1,random_state=2394)
            randomforestreg.fit(xtrain,ytrain)

            # returning the tuned random forest regressor
            return randomforestreg

        except Exception as e:
            self.logger.log(self.file_obj, f"Exception occurred while tuning the random forest regressor. Exception: {str(e)}")
            raise e


    def tuneSVR(self,xtrain,ytrain):

        """
        Description: This method is used to tune the hyperparameters of the SVR machine learning model.

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param xtrain: training independent feature data
        :param ytrain: training dependent feature data

        :return: tuned SVR model
        """
        try:
            self.logger.log(self.file_obj, "Finding the best hyperparameters for the SVR machine learning model")

            # creating a dictionary containing all the possible values of the hyperparameters for the random forest regressor
            best_parameters = dict()
            best_parameters['kernel'] = ['linear', 'rbf', 'poly', 'sigmoid']
            best_parameters['C'] = np.arange(1, 11, 1)

            # performing the grid search cv to find the best hyperparameters for the random forest regressor
            svr_grid = RandomizedSearchCV(self.svr, best_parameters, cv=5, n_jobs=-1)
            svr_grid.fit(xtrain, ytrain)

            # getting the found best hyperparameters
            kernel = svr_grid.best_params_['kernel']
            C = svr_grid.best_params_['C']

            # creating a support vector regressor model using the best hyperparameters found in last step earlier
            supportVectorReg = SVR(kernel=kernel,C=C)
            supportVectorReg.fit(xtrain, ytrain)

            # returning the tuned support vector regressor machine learning algorithm
            return supportVectorReg

        except Exception as e:
            self.logger.log(self.file_obj, "Exception occurred while tuning the support vector regressor machine "
                                           "learning algorithm ")
            raise e

    def tunexgboost(self,xtrain,ytrain):


        """
        Description: This method is used to tune the hyperparameters of the xgboost regressor

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param xtrain: training independent feature data
        :param ytrain: training dependent feature data

        :return: tuned xgboost regressor model
        """
        try:
            self.logger.log(self.file_obj, "Finding the best hyperparameters for the xgboost regressor machine learning algorithm")

            # creating a dictionary containing all the possible values of the hyperparameters
            best_parameters = dict()
            best_parameters['learning_rate'] = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ]
            best_parameters['max_depth'] = [ 3, 4, 5, 6, 8, 10, 12, 15]
            best_parameters['min_child_weight'] = [1, 3, 5, 7]
            best_parameters['gamma'] = [ 0.0, 0.1, 0.2 , 0.3, 0.4 ]
            best_parameters["colsample_bytree"] = [0.3, 0.4, 0.5 , 0.7]

            #  performing the grid search cv to find the best hyperparameters for the xgboost regressor
            xgboost_grid = RandomizedSearchCV(self.xgb,best_parameters,n_jobs=-1,cv=5)
            xgboost_grid.fit(xtrain,ytrain)

            # finding the best estimator
            xgboostReg = xgboost_grid.best_estimator_
            xgboostReg.fit(xtrain,ytrain)

            # returning the best estimator
            return xgboostReg

        except Exception as e:
            self.logger.log(self.file_obj, "Exception occurred while tuning the xgboost regressor machine learning algorithm")
            raise e



    def createStackingRegressor(self,xtrain,ytrain):
        
        """
        Description: This method is used to create a stacking regressor using SVR, random forest regressor and 
        elastic net regressor as a base estimator and xgboost as a final (meta) estimator
        
        Written By: Shivam Shinde
        
        Version: 1.0
        
        Revision: None
        
        :return: stacking regressor model
        """
        
        try:
            # fetching the base and mets models
            model1 = self.tuneSVR(xtrain, ytrain)
            model2 = self.tuneRidge(xtrain, ytrain)
            model3 = self.tuneRandomForestRegressor(xtrain, ytrain)
            metamodel = self.tunexgboost(xtrain, ytrain)

            # creating a list of base estimators
            estimators = [
                ('svr',model1),
                ('rr',model2),
                ('rfr',model3)
            ]

            # creating and fitting a stacking regressor
            stackingReg = StackingRegressor(estimators=estimators,final_estimator=metamodel)
            stackingReg.fit(xtrain,ytrain)

            # returning a created stacking regressor
            return stackingReg

        except Exception as e:
            self.logger.log(self.file_obj,f"Exception occurred while creating a stacking regressor. Exception: {str(e)}")
            raise e


    def bestModelFinder(self,xtrain,xtest,ytrain,ytest):

        """
        Description: This method is used to find the best model using the r2-adjusted score

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param xtrain: training independent feature data
        :param ytrain: training dependent feature data
        :param xtest: testing independent feature data
        :param ytest: testing dependent feature data
        :return:
        """

        try:
            # finding the r2-adjusted score for SVR model
            svr = self.tuneSVR(xtrain, ytrain)
            svr_predictions = svr.predict(xtest)
            r2_svr = r2_score(ytest, svr_predictions)
            r2_adj_svr = self.r2_adjusted_score(r2_svr, xtrain)

            # finding the r2-adjusted score for random forest regressor
            rfr = self.tuneRandomForestRegressor(xtrain, ytrain)
            rfr_prediction = rfr.predict(xtest)
            r2_rfr = r2_score(ytest, rfr_prediction)
            r2_adj_rfr = self.r2_adjusted_score(r2_rfr, xtrain)

            # finding the r2-adjusted score for elastic net model
            rr = self.tuneRidge(xtrain, ytrain)
            rr_prediction = rr.predict(xtest)
            r2_rr = r2_score(ytest, rr_prediction)
            r2_adj_rr = self.r2_adjusted_score(r2_rr, xtrain)

            # finding the r2 adjusted score for xgboost regressor model
            xgb = self.tunexgboost(xtrain, ytrain)
            xgb_prediction = xgb.predict(xtest)
            r2_xgb = r2_score(ytest, xgb_prediction)
            r2_adj_xgb = self.r2_adjusted_score(r2_xgb, xtrain)

            # finding the r2 adjusted score for the stacking regressor
            sr = self.createStackingRegressor(xtrain, ytrain)
            sr_prediction = sr.predict(xtest)
            r2_sr = r2_score(ytest, sr_prediction)
            r2_adj_sr = self.r2_adjusted_score(r2_sr, xtrain)

            max_r2_adj = max([r2_adj_svr, r2_adj_rfr, r2_adj_rr, r2_adj_xgb, r2_adj_sr])


            if max_r2_adj == r2_adj_sr:
                return "StackingRegressor", sr
            elif max_r2_adj == r2_adj_xgb:
                return "XGBRegressor", xgb
            elif max_r2_adj == r2_adj_rfr:
                return "RandomForestRegressor", rfr
            elif max_r2_adj == r2_adj_svr:
                return "SVR", svr
            else:
                return "RidgeRegressor", rr

        except Exception as e:
            self.logger.log(self.file_obj, f"Exception occurred while finding the best machine learning model. "
                                           f"Exception: {str(e)}")
            raise e