import os
import shutil

import joblib

from Logging.logging import Logger


class modelMethods:

    """

    Description: This class will contain the methods used for saving, loading and finding the correct model for
    correct cluster

    Written By: Shivam Shinde

    Version: 1.0

    Revision: None

    """

    def __int__(self):
        self.logger = Logger()
        self.file_object = open("../TrainingLogs/modelMethodsLogs.txt","a+")
        self.model_directory = "../Models"


    def modelSaving(self,model,filename):

        """
        Description: This method is used to save the created model as a python pickle file

        :param model: Reference of the created model

        :param filename: Name of the model after saving

        :return: None
        """

        self.logger.log(self.file_object, "Saving the created model into the python pickle file")
        try:
            path = os.path.join(self.model_directory, filename)

            if os.path.exists(path):
                shutil.rmtree(self.model_directory)
                os.makedirs(path)
            else:
                os.makedirs(path)

            joblib.dump(model,os.path.join(path, f"{filename}.pkl"))

            self.logger.log(self.file_object, f"Model {model} saved successfully in {path}")

        except Exception as e:
            self.logger.log(self.file_object, f"Exception occurred while saving the model {model}. Exception: {str(e)}")
            raise e


    def loadingSavedModel(self,filename):

        """
        Description: This method is used to load the saved method for the respective cluster

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param filename: Name of the model that needs to be saved
        :return: Model
        """
        try:
            self.logger.log(self.file_object, f"Loading the model {filename}.pkl")
            path = os.path.join(self.model_directory, filename, f"{filename}.pkl")
            model = joblib.load(path)
            self.logger.log(self.file_object, f"Model {filename} loaded successfully")
            return model

        except  Exception as e:
            self.logger.log(self.file_object, f"Exception occurred while loading the model {filename}. "
                                              f"Exception: {str(e)}")
            raise e


    def findCorrectModel(self,clusterNumber):

        """
        Description: This method is used to find the correct model given the  cluster number

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param clusterNumber: Cluster number
        :return: Model name
        """

        self.logger.log(self.file_object, f"Finding the appropriate model for cluster number {clusterNumber}")
        try:
            for file in os.listdir(self.model_directory):

                if (file.index(str(clusterNumber))) == -1:
                    self.model_name = file.split(".")[0]
                else:
                    continue

            self.logger.log(self.file_object, f"Successfully found the name of the model for the cluster number "
                                              f"{clusterNumber}")
            return self.model_name

        except Exception as e:
            self.logger.log(self.file_object, f"Exception occurred while finding the name of the model for the "
                                              f"cluster number {clusterNumber}. Exception: {str(e)}")
            raise e