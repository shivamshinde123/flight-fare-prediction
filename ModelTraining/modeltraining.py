import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

from Logging.logging import Logger
from Training_Clustering.clustering import Cluster
from Training_Preprocessing.preprocessor import Preprocessor
from model_methods.model_methods import modelMethods
from model_tuner.modeltuner import modelTuner

warnings.simplefilter(action='ignore', category=FutureWarning)


class modelTraining:

    """
    Description: This class a method which is used to train a machine learning model for each data cluster

    Written By: Shivam Shinde

    Version: 1.0

    Revision: None

    :returns: None
    """

    def __init__(self):
        self.file_obj = open("../TrainingLogs/ModelTraining.txt","a+")
        self.logger = Logger()

    def trainingModels(self):
        """
        Description: This method is used to train the machine learning model for the every cluster of the data

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None
        
        :return: None
        """
        try:
            self.logger.log(
                self.file_obj, "*************MACHINE LEARNING MODEL TRAINING FOR ALL THE CLUSTERS STARTED**************")

            # preprocessing the obtained data
            self.logger.log(self.file_obj, "Training_Preprocessing of the data started!!")
            p = Preprocessor()
            p.preprocess()
            X = pd.read_csv("../Training_PreprocessedData/XPreprocessed.csv")
            y = pd.read_csv("../Training_PreprocessedData/yDataframe.csv")
            self.logger.log(self.file_obj, "Training_Preprocessing of the data completed!!")

            # clustering the training and testing data into the same number of clusters
            self.logger.log(self.file_obj, "Training_Clustering of the data started!!")
            c = Cluster()
            noOfClusters = c.createElbowPlot(X)
            X = c.createCluster(X, noOfClusters)
            self.logger.log(self.file_obj, "Training_Clustering of the data completed!!")

            # Adding one more column to X i.e. dependent feature
            X['Price'] = y

            # finding the unique numbers in the ClusterNumber column of the X
            clusters = X['ClusterNumber'].unique()


            for i in clusters:
                self.logger.log(self.file_obj, f"*************for the cluster number {i}**************")
                clusterData = X[X['ClusterNumber'] == i]

                clusterFeatures = clusterData.drop(
                    columns=['ClusterNumber', 'Price'], axis=1)
                clusterLabel = clusterData['Price']

                # splitting the cluster data into train and test data
                X_train, X_test, y_train, y_test = train_test_split(
                    clusterFeatures, clusterLabel, test_size=0.2, random_state=348724)

                self.logger.log(self.file_obj,f"Finding the best model for the cluster {i}")

                # finding the best model for this cluster
                mt = modelTuner()
                bestModelName, bestModel = mt.bestModelFinder(
                    X_train, X_test, y_train, y_test)

                # saving the best model obtained
                mm = modelMethods()
                mm.modelSaving(bestModel, bestModelName, i)

                self.logger.log(
                    self.file_obj,
                    f"Training of the machine learning model for the data cluster {i} successfully completed")

            self.logger.log(self.file_obj, "***************MACHINE LEARNING MODEL TRAINING FOR ALL CLUSTERS COMPLETED "
                                           "SUCCESSFULLY*************")

        except Exception as e:
            self.logger.log(
                self.file_obj, f"Exception occurred while training the machine learning model. Exception: {str(e)}")
            raise e



