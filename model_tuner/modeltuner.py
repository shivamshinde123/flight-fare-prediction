import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor, AdaBoostRegressor, \
    VotingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
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
        self.file_obj = open("TrainingLogs/bestModelFindingLogs.txt", "a+")
        self.ridge = Ridge()
        self.lasso = Lasso()
        self.svr = SVR()
        self.knr = KNeighborsRegressor()
        self.rfr = RandomForestRegressor()
        self.gbr = GradientBoostingRegressor()
        self.abr = AdaBoostRegressor()
        self.xgb = XGBRegressor()

    def tuneSVR(self, xtrain, ytrain):

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
            best_parameters['C'] = np.arange(1, 6, 1)

            # performing the grid search cv to find the best hyperparameters for the random forest regressor
            svr_grid = GridSearchCV(self.svr, best_parameters, cv=5, n_jobs=-1)
            svr_grid.fit(xtrain, ytrain)

            # getting the found best hyperparameters
            C = svr_grid.best_params_['C']

            # creating a support vector regressor model using the best hyperparameters found in last step earlier
            supportVectorReg = SVR(C=C)
            supportVectorReg.fit(xtrain, ytrain)

            # returning the tuned support vector regressor machine learning algorithm
            return supportVectorReg

        except Exception as e:
            self.logger.log(self.file_obj, f"Exception occurred while tuning the support vector regressor machine "
                                           f"learning algorithm. Exception: {str(e)}")
            raise e

    def tuneRidge(self, xtrain, ytrain):

        """
        Description: This method is used to tune the hyperparameters of the elastic net regressor.

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param xtrain: training independent feature data

        :param ytrain: training dependent feature data

        :return: tuned ridge regression model
        """

        try:
            self.logger.log(self.file_obj,
                            "Finding the best hyperparameters for the ridge regression machine learning model")

            # creating a dictionary containing the possible options for the hyperparameters
            best_parameters = dict()
            best_parameters['alpha'] = np.arange(1, 6, 1)

            # performing the grid search cv on the elastic net model using the earlier created dictionary
            rr_grid = GridSearchCV(self.ridge, best_parameters, cv=5, n_jobs=-1)
            rr_grid.fit(xtrain, ytrain)

            # finding the best hyperparameters obtained through the grid search cv
            alpha = rr_grid.best_params_["alpha"]

            # creating a ridge regression model using the best parameters obtained earlier
            rr = Ridge(alpha=alpha,random_state=90345)
            rr.fit(xtrain, ytrain)

            # returning the ridge regression model
            return rr

        except Exception as e:
            self.logger.log(self.file_obj, f"Exception occurred while finding the best hyperparameters for the ridge "
                                           f"regression machine learning model. Exception: {str(e)}")
            raise e

    def tuneLasso(self, xtrain, ytrain):

        """
        Description: This method is used to tune the hyperparameters of the lasso regression model

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param xtrain: training independent feature data

        :param ytrain: training dependent feature data

        :return: tuned lasso regression model
        """

        try:
            self.logger.log(self.file_obj,
                            "Finding the best hyperparameters for the lasso regression machine learning model")

            # creating a dictionary which will hold all the possible values of the hyperparameters for the machine learning model
            best_parameters = dict()
            best_parameters['alpha'] = np.arange(1, 6, 1)

            # performing randomized search on the training data to find the best hyperparameters
            lr_grid = GridSearchCV(self.lasso, best_parameters, n_jobs=-1, cv=5)
            lr_grid.fit(xtrain, ytrain)

            # getting the best parameters obtained from the randomized search cv
            alpha = lr_grid.best_params_['alpha']

            # training the lasso regression model using the best hyperparameters that were found out by randomized search cv
            lassoreg = Lasso(alpha=alpha,max_iter=2000, random_state=23923)
            lassoreg.fit(xtrain, ytrain)

            # returning the tuned model
            return lassoreg

        except Exception as e:
            self.logger.log(self.file_obj, f"Exception occurred while tuning the lasso regression model. Exception: {str(e)}")
            raise e


    def tuneKNRegressor(self,xtrain,ytrain):

        """
        Description: This method is used to tune the hyperparameters of the KNeighbours regressor

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param xtrain: training independent feature data

        :param ytrain: training dependent feature data

        :return: tuned KNeighbour regressor model
        """

        try:
            self.logger.log(self.file_obj, "Finding the best hyperparameters for the KNeighbours regression machine learning model")

            # creating a dictionary containing all the possible values of the hyperparameters
            best_parameters = dict()
            best_parameters['n_neighbors'] = np.arange(3,6,1)

            # performing randomized search cv to find the best hyperparameter
            knr_grid = GridSearchCV(self.knr, best_parameters,n_jobs=-1,cv=5)
            knr_grid.fit(xtrain,ytrain)

            # getting the best hyperparameters obtained using the randomized search cv
            n_neighbors = knr_grid.best_params_['n_neighbors']

            # training the k-neighbors regressor using the hyperparameters found using the randomized search cv
            knr = KNeighborsRegressor(n_neighbors=n_neighbors,n_jobs=-1)
            knr.fit(xtrain,ytrain)

            # returing the tuned model
            return knr

        except Exception as e:
            self.logger.log(self.file_obj, f"Exception occurred while tuning the KNeighbours regression model. Exception: {str(e)}")
            raise e

    def tuneRandomForestRegressor(self, xtrain, ytrain):

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
            self.logger.log(self.file_obj,
                            "Finding the best hyperparameters for the random forest regressor machine learning model")

            # creating a dictionary containing the possible values of the hyperparameters
            best_parameters = dict()
            best_parameters['n_estimators'] = np.arange(100, 400, 100)
            best_parameters['max_depth'] = [2, 3]

            # performing the grid search cv using the hyperparameter dictionary created earlier
            rfr_grid = GridSearchCV(self.rfr, best_parameters, cv=5, n_jobs=-1)
            rfr_grid.fit(xtrain, ytrain)

            # getting the best hyperparameters for the random forest regressor using the fitted rfr_grid
            n_estimators = rfr_grid.best_params_['n_estimators']
            max_depth = rfr_grid.best_params_['max_depth']


            # creating a random forest regressor using the best hyperparameters obtained earlier
            randomforestreg = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth,
                                                    bootstrap=True, n_jobs=-1,
                                                    random_state=2394)
            randomforestreg.fit(xtrain, ytrain)

            # returning the tuned random forest regressor
            return randomforestreg

        except Exception as e:
            self.logger.log(self.file_obj,
                            f"Exception occurred while tuning the random forest regressor. Exception: {str(e)}")
            raise e

    def tuneGradientBoostingRegressor(self,xtrain,ytrain):

        """
        Description: This method is used to tune the hyperparameters of the gradient boost regressor

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param xtrain: training independent feature data

        :param ytrain: training dependent feature data

        :return: tuned gradient boost regressor model
        """

        try:
            self.logger.log(self.file_obj, "Finding the best hyperparameters for the gradient boosting regressor model")

            # creating a dictionary which will contain all the possible values of the hyperparameters
            best_parameters = dict()
            best_parameters['n_estimators'] = np.arange(100, 400, 100)
            best_parameters['learning_rate'] = np.arange(0.1, 0.6, 0.1)

            # finding the best hyperparameters for the model using the GridSearchCV
            gbr_grid = GridSearchCV(self.gbr, best_parameters, n_jobs=-1, cv=5)
            gbr_grid.fit(xtrain, ytrain)

            # getting the best hyperparameters found
            n_estimators = gbr_grid.best_params_['n_estimators']
            learning_rate = gbr_grid.best_params_['learning_rate']

            # using the found hyperparameters to train the model
            gbr = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_estimators, random_state=239084)
            gbr.fit(xtrain, ytrain)

            return gbr

        except Exception as e:
            self.logger.log(self.file_obj, f"Exception occurred while tuning the gradient boosting regressor machine "
                                           f"learning model. Exception: {str(e)}")
            raise e

    def tunexgboost(self, xtrain, ytrain):

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
            self.logger.log(self.file_obj,
                            "Finding the best hyperparameters for the xgboost regressor machine learning algorithm")

            # creating a dictionary containing all the possible values of the hyperparameters
            best_parameters = dict()
            best_parameters['max_depth'] = [5, 6, 8, 10]
            best_parameters['gamma'] = [0.0, 0.1, 0.2, 0.3]

            #  performing the grid search cv to find the best hyperparameters for the xgboost regressor
            xgboost_grid = GridSearchCV(self.xgb, best_parameters, n_jobs=-1, cv=5)
            xgboost_grid.fit(xtrain, ytrain)

            # finding the best estimator
            xgboostReg = xgboost_grid.best_estimator_
            xgboostReg.fit(xtrain, ytrain)

            # returning the best estimator
            return xgboostReg

        except Exception as e:
            self.logger.log(self.file_obj,
                            f"Exception occurred while tuning the xgboost regressor machine learning algorithm. Exception: {str(e)}")
            raise e


    def tuneAdaBoostingRegressor(self,xtrain,ytrain):

        """
        Description: This method is used to tune the hyperparameters of the ada boost regressor

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param xtrain: training independent feature data

        :param ytrain: training dependent feature data

        :return: tuned ada boosting regressor model
        """

        try:
            self.logger.log(self.file_obj, "Finding the best hyperparameters for the ada boosting regressor model")

            # creating a dictionary which will contain all the possible values of the hyperparameters
            best_parameters = dict()
            best_parameters['n_estimators'] = np.arange(100, 400, 100)
            best_parameters['learning_rate'] = np.arange(0.1, 0.6, 0.1)

            # finding the best hyperparameters for the model using the GridSearchCV
            abr_grid = GridSearchCV(self.abr, best_parameters, n_jobs=-1, cv=5)
            abr_grid.fit(xtrain, ytrain)

            # getting the best hyperparameters found
            n_estimators = abr_grid.best_params_['n_estimators']
            learning_rate = abr_grid.best_params_['learning_rate']

            # using the found hyperparameters to train the model
            abr = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_estimators, random_state=239084)
            abr.fit(xtrain, ytrain)

            return abr

        except Exception as e:
            self.logger.log(self.file_obj, f"Exception occurred while tuning the ada boosting regressor machine "
                                           f"learning model. Exception: {str(e)}")
            raise e

    def tuneVotingRegressor(self,xtrain,ytrain):

        """
        Description: This method is used to return the voting classifier made up of ada boosting regressor,
        gradient boosting regressor and xgboost regressor machine learning models

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param xtrain: training independent feature data

        :param ytrain: training dependent feature data

        :return: voting regressor machine learning model
        """
        try:
            self.logger.log(self.file_obj, "Creating a voting regressor using Ada boosting regressor, Gradient "
                                           "boosting regressor and xgboosting regressor")

            abr = self.tuneAdaBoostingRegressor(xtrain,ytrain)
            gbr = self.tuneGradientBoostingRegressor(xtrain,ytrain)
            xgb = self.tunexgboost(xtrain,ytrain)

            vr = VotingRegressor(estimators=[
                ('abr', abr),
                ('gbr', gbr),
                ('xgb', xgb)])

            vr.fit(xtrain,ytrain)
            return vr

        except Exception as e:
            self.logger.log(self.file_obj, f"Exception occurred while creating voting regressor machine "
                                           f"learning model. Exception: {str(e)}")
            raise e

    def createStackingRegressor(self, xtrain, ytrain):

        """
        Description: This method is used to create a stacking regressor using SVR, random forest regressor and 
        K neighbors regressor as a base estimator and xgboost as a final (meta) estimator
        
        Written By: Shivam Shinde
        
        Version: 1.0
        
        Revision: None
        
        :return: stacking regressor model
        """

        try:
            # fetching the base and metas models
            model1 = self.tuneVotingRegressor(xtrain, ytrain)
            model2 = self.tuneAdaBoostingRegressor(xtrain, ytrain)
            model3 = self.tuneRandomForestRegressor(xtrain, ytrain)
            metamodel = self.tunexgboost(xtrain, ytrain)

            # creating a list of base estimators
            estimators = [
                ('vr', model1),
                ('abr', model2),
                ('rfr', model3)
            ]

            # creating and fitting a stacking regressor
            stackingReg = StackingRegressor(estimators=estimators, final_estimator=metamodel)
            stackingReg.fit(xtrain, ytrain)

            # returning a created stacking regressor
            return stackingReg

        except Exception as e:
            self.logger.log(self.file_obj,
                            f"Exception occurred while creating a stacking regressor. Exception: {str(e)}")
            raise e


    def bestModelFinder(self, xtrain, xtest, ytrain, ytest):

        """
        Description: This method is used to find the best model using the r2-adjusted score

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param xtrain: training independent feature data

        :param ytrain: training dependent feature data

        :param xtest: testing independent feature data

        :param ytest: testing dependent feature data
        
        :return: best model that fits the concerned cluster data
        """

        try:
            self.logger.log(self.file_obj, "*********finding best model for cluster************")
            # finding the r2-adjusted score for SVR model
            svr = self.tuneSVR(xtrain, ytrain)
            svr_predictions = svr.predict(xtest)
            r2_svr = r2_score(ytest, svr_predictions)
            self.logger.log(self.file_obj, f"The r2 score for the SVR model is {np.round(r2_svr,3)}")

            # finding the r2-adjusted score for elastic net model
            rr = self.tuneRidge(xtrain, ytrain)
            rr_prediction = rr.predict(xtest)
            r2_rr = r2_score(ytest, rr_prediction)
            self.logger.log(self.file_obj,
                            f"The r2 score for the ridge regression model is {np.round(r2_rr, 3)}")

            # finding the r2 adjusted score for the lasso regression model
            lasso = self.tuneLasso(xtrain, ytrain)
            lasso_prediction = lasso.predict(xtest)
            r2_lasso = r2_score(ytest, lasso_prediction)
            self.logger.log(self.file_obj, f"The r2 score for the lasso regression model is {np.round(r2_lasso, 3)}")

            # finding the r2-adjusted score for the KNeighbors regression model
            knr = self.tuneKNRegressor(xtrain,ytrain)
            knr_prediction = knr.predict(xtest)
            r2_knr = r2_score(ytest,knr_prediction)
            self.logger.log(self.file_obj, f"The r2 score for the KNeighbors regression model is {np.round(r2_knr,3)}")

            # finding the r2-adjusted score for random forest regressor
            rfr = self.tuneRandomForestRegressor(xtrain, ytrain)
            rfr_prediction = rfr.predict(xtest)
            r2_rfr = r2_score(ytest, rfr_prediction)
            self.logger.log(self.file_obj,
                            f"The r2 score for the random forest regressor model is {np.round(r2_rfr,3)}")


            gbr = self.tuneGradientBoostingRegressor(xtrain,ytrain)
            gbr_prediction = gbr.predict(xtest)
            r2_gbr = r2_score(ytest,gbr_prediction)
            self.logger.log(self.file_obj,
                            f"The r2 score for the gradient boosting regressor model is {np.round(r2_gbr, 3)}")

            # finding the r2 adjusted score for xgboost regressor model
            xgb = self.tunexgboost(xtrain, ytrain)
            xgb_prediction = xgb.predict(xtest)
            r2_xgb = r2_score(ytest, xgb_prediction)
            self.logger.log(self.file_obj,
                            f"The r2 score for the xgboost regression model is {np.round(r2_xgb, 3)}")

            abr = self.tuneAdaBoostingRegressor(xtrain, ytrain)
            abr_prediction = abr.predict(xtest)
            r2_abr = r2_score(ytest, abr_prediction)
            self.logger.log(self.file_obj,
                            f"The r2 score for the ada boosting regressor model is {np.round(r2_abr, 3)}")

            vr = self.tuneVotingRegressor(xtrain,ytrain)
            vr_prediction = vr.predict(xtest)
            r2_vr = r2_score(ytest, vr_prediction)
            self.logger.log(self.file_obj,
                            f"The r2 score for the voting regressor model is {np.round(r2_vr, 3)}")

            # finding the r2 adjusted score for the stacking regressor
            sr = self.createStackingRegressor(xtrain, ytrain)
            sr_prediction = sr.predict(xtest)
            r2_sr = r2_score(ytest, sr_prediction)
            self.logger.log(self.file_obj,
                            f"The r2 score for the stacking regressor model is {np.round(r2_sr,3)}")

            self.logger.log(self.file_obj, "*********found the best model for cluster************")

            max_r2 = max([r2_svr, r2_rfr, r2_rr, r2_xgb, r2_sr, r2_lasso, r2_knr,r2_gbr,r2_abr,r2_vr])

            if max_r2 == r2_sr:
                return "StackingRegressor", sr
            elif max_r2 == r2_vr:
                return "VotingRegressor", vr
            elif max_r2 == r2_xgb:
                return "XGBRegressor", xgb
            elif max_r2 == r2_rfr:
                return "RandomForestRegressor", rfr
            elif max_r2 == r2_gbr:
                return "GradientBoostingRegressor", gbr
            elif max_r2 == r2_abr:
                return "AdaBoostingRegressor", abr
            elif max_r2 == r2_knr:
                return "KNeighborsRegressor", knr
            elif max_r2 == r2_svr:
                return "SVR", svr
            elif max_r2 == r2_rr:
                return "RidgeRegressor", rr
            else:
                return "LassoRegression", lasso



        except Exception as e:
            self.logger.log(self.file_obj, f"Exception occurred while finding the best machine learning model. "
                                           f"Exception: {str(e)}")
            raise e
