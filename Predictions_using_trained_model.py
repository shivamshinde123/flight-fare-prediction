import os

import pandas as pd

from Logging.logging import Logger
from Prediction_Preprocessing.preprocessor_prediction import PreprocessorPrediction
from Prediction_raw_data_validation.rawDataValidation_prediction import rawPredictionDataValidation
from model_methods.model_methods import modelMethods


class predictionsUsingTheTrainedModels:

    """
    Descriptions: This class contains the methods which will predict the flight fare for the given records in the csv file

    Written By: Shivam Shinde

    Version: 1.0

    Revision: None
    """

    def __init__(self,path):
        self.logger = Logger()
        self.file_obj = open("PredictionLogs/predictions.txt", "a+")
        self.prediction_data_validation = rawPredictionDataValidation(path)

    def predictUsingModel(self):

        """
        Description : This method is used to predict the flight fare for the observations given in the data file using
        the trained model

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None
        :return: path of the csv file containing the predicted values
        """

        try:
            # deleting the prediction files from the previous code run
            self.prediction_data_validation.deletePredictionOutputFiles()

            # now that previous files are deleted, it is time for preprocessing of validated files
            p = PreprocessorPrediction()
            data = p.preprocessPrediction()

            # clustering the data into 4 clusters
            # loading the trained kmeans model
            mm = modelMethods()
            kmeans_model1 = mm.loadingSavedModel("KMeansCluster", 4)
            clusterNumbers = kmeans_model1.predict(data)

            # creating a column in the data which will contains the cluster number for the particular observation
            data['clusterNo'] = clusterNumbers

            clusters = data['clusterNo'].unique()

            # creating an empty list which contains the prediction results
            prediction_results = []

            # creating an empty list which will contain the indices of the predictions
            prediction_indices = []

            for i in clusters:

                clusterData = data[data['clusterNo'] == i]

                # adding the indices of the observations for the cluster number i
                for j in clusterData.index:
                    prediction_indices.append(j)

                clusterFeatures = clusterData.drop(columns=['clusterNo'], axis=1)

                # finding the saved model for cluster number i
                model_name = modelMethods().findCorrectModel(i)
                model = modelMethods().loadingSavedModel(model_name, i)

                # predicting the flight fare for each of the observations in the data
                predictions = model.predict(clusterFeatures)

                # appending the prediction results found in an empty list
                for prediction in predictions:
                    prediction_results.append(prediction)

            if not os.path.exists("Prediction_output_files"):
                os.makedirs("Prediction_output_files")

            predictions = pd.DataFrame(prediction_results,columns=["Flight_Fare"])
            predictions['Index_column'] = prediction_indices

            predictions.to_csv("Prediction_output_files/predicted_flight_fare_data.csv",header=True)

            self.logger.log(self.file_obj, f"Prediction results placed at the path: Prediction_output_files/predicted_flight_fare_data.csv")

        except Exception as e:
            self.logger.log(self.file_obj, f"Exception occurred while predicting the flight fare using the saved "
                                           f"models. Exception: {str(e)}")
            raise e


