import pandas as pd

class DataGetter:

    """
    Description: This class is used to read the data from the location where the client has put it.
    Written By: Shivam Shinde
    Version: 1.0
    Revision: None
    """

    def __init__(self, fileObject, loggerObject):
        self.fileObject = fileObject
        self.loggerObject = loggerObject
        self.trainingData = "../fileFromDb_prediction/inputFile.csv"

    def getData(self):

        """

        Description: This method is used to read the data file from the provided location.

        On failure: Raises an exception

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :return: pandas.DataFrame

        """

        self.loggerObject.log(self.fileObject, "Entered the getData method of DataGetter class")
        try:
            self.data = pd.read_csv(self.trainingData)
            self.loggerObject.log(self.fileObject,"Successfully loaded the data using getData method of DataGetter class")
            self.loggerObject.log(self.fileObject, "Exiting the getData method of DataGetter class")
            return self.data
        except Exception as e:
            self.loggerObject.log(self.fileObject, "Exception occurred while loading data using getData method of DataGetter class. Error message: str(e")
            self.loggerObject.log(self.fileObject, "Data loading unsuccessful using the getData method of DataGetter class due to exception")
            raise e

