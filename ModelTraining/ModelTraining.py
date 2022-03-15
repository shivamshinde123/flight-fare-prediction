from sklearn.model_selection import train_test_split

from Clustering.clustering import Cluster
from Logging.logging import Logger
from Preprocessing.preprocessor import Preprocessor
from data_ingestion.data_loading_train import DataGetter
from model_methods.model_methods import modelMethods
from model_tuner.modeltuner import modelTuner


class modelTraining:

    """
    Description: This class contains the method used to train the machine learning models

    Written By: Shivam Shinde

    Version: 1.0

    Revision: None
    """

    def __int__(self):
        self.logger = Logger()
        self.file_obj = open("../TrainingLogs/ModelTraining.txt", "a+")

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
                self.file_obj, "Machine learning model training started")

            # getting the data
            f = open("../Training_Data_From_Client/dataIngestion.txt", "a+")
            dg = DataGetter(f, self.logger)
            data = dg.getData()

            # preprocessing the obtained data
            p = Preprocessor()
            X, y = p.preprocess()

            # clustering the training and testing data into the same number of clusters
            c = Cluster()
            noOfClusters = c.createElbowPlot(X)
            X = c.createCluster(X, noOfClusters)

            # Adding one more column to X i.e. dependent feature
            X['Price'] = y

            # finding the unique numbers in the ClusterNumber column of the X
            clusters = X['ClusterNumber'].unique()

            for i in clusters:
                clusterData = X[X['ClusterNumber'] == i]

                clusterFeatures = clusterData.drop(
                    columns=['ClusterNumber', 'Price'], axis=1)
                clusterLabel = clusterData['Price']

                # splitting the cluster data into train and test data
                X_train, X_test, y_train, y_test = train_test_split(
                    clusterFeatures, clusterLabel, test_size=0.2, random_state=348724)

                # finding the best model for this cluster
                mt = modelTuner()
                bestModelName, bestModel = mt.bestModelFinder(
                    X_train, X_test, y_train, y_test)

                # saving the best model obtained
                mm = modelMethods()
                mm.modelSaving(bestModel, bestModelName, i)

                self.logger.log(
                    self.file_obj, f"Training of the machine learning model for the data cluster {i} successfully completed")

        except Exception as e:
            self.logger.log(
                self.file_obj, f"Exception occurred while training the machine learning model. Exception: {str(e)}")
            raise e


m = modelTraining()
m.trainingModels()
