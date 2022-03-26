import os
import pathlib
import pickle
import shutil
import warnings

from Logging.logging import Logger

warnings.simplefilter(action='ignore', category=FutureWarning)

class modelMethods:
    """

    Description: This class will contain the methods used for saving, loading and finding the correct model for
    correct cluster

    Written By: Shivam Shinde

    Version: 1.0

    Revision: None

    """

    def __init__(self):
        self.model_directory = "Models"
        self.logger = Logger()
        self.file_object = open("TrainingLogs/modelMethodsLogs.txt","a+")

    def modelSaving(self, model, filename,clusterno):

        """
        Description: This method is used to save the created model as a python pickle file

        :param model: Reference of the created model

        :param filename: Name of the model after saving

        :return: None
        """

        self.logger.log(self.file_object, "Saving the created model into the python pickle file")
        try:
            if filename == "KMeansCluster":
                path = os.path.join(self.model_directory, "ClusteringModel")
            else:
                path = os.path.join(self.model_directory, "ModelForClusterNo"+str(clusterno))

            # removing the previously created models
            if os.path.exists(path):
                shutil.rmtree(self.model_directory)
                os.makedirs(path)
            else:
                os.makedirs(path)

            # saving the model as a python pickle file
            pickle.dump(model, open(os.path.join(path, f"{filename}.pkl"),"wb"))

            self.logger.log(self.file_object, f"Model {model} saved successfully in {path}")

        except Exception as e:
            self.logger.log(self.file_object, f"Exception occurred while saving the model {model}. Exception: {str(e)}")
            raise e

    def loadingSavedModel(self, filename, clusterno):

        """
        Description: This method is used to load the saved method for the respective cluster

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param clusterno: Cluster number for which the model is to be loaded
        :param filename: Name of the model that needs to be saved

        :return: Model
        """
        try:
            self.logger.log(self.file_object, f"Loading the model {filename}.pkl")
            if filename == "KMeansCluster":
                path = os.path.join(self.model_directory, "ClusteringModel")
            else:
                path = os.path.join(self.model_directory, "ModelForClusterNo" + str(clusterno))

            # loading the saved model
            path1 = os.path.join(path, f"{filename}.pkl")
            model = pickle.load(open(path1,"rb"))
            self.logger.log(self.file_object, f"Model {filename} loaded successfully")

            # returning the model
            return model

        except  Exception as e:
            self.logger.log(self.file_object, f"Exception occurred while loading the model {filename}. "
                                              f"Exception: {str(e)}")
            raise e

    def findCorrectModel(self, clusterNumber):

        """
        Description: This method is used to find the correct model given the  cluster number

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param clusterNumber: Cluster number
        
        :return: Model name
        """

        self.logger.log(self.file_object, f"Finding the appropriate model for cluster number {clusterNumber}")
        self.clusterNumber = clusterNumber
        try:
            # finding the appropriate model for the given cluster number
            for file in os.listdir(self.model_directory):

                path = os.path.join(self.model_directory,file)
                path = pathlib.Path(path)
                if (path.stem[-1]) == str(clusterNumber):
                    for file1 in os.listdir(path):
                        model_name = file1.split('.')[0]
                        self.logger.log(self.file_object,
                                        f"Successfully found the name of the model for the cluster number "
                                        f"{clusterNumber}")
                        # returning the model
                        return model_name
                else:
                    continue

        except Exception as e:
            self.logger.log(self.file_object, f"Exception occurred while finding the name of the model for the "
                                              f"cluster number {clusterNumber}. Exception: {str(e)}")
            raise e


