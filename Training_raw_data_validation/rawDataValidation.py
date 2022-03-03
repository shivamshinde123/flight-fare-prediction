import os
import shutil
from logging import Logger
import json


class rawDataValidation:

    """
    Description: This class contains the methods used for the validation of the raw training data
    Written By: Shivam Shinde
    Version: 1.0
    Revision: None

    """

    def __init__(self,path):
        self.Batch_Directory = path
        self.logger = Logger()
        self.schema = "schema_training.json"

    def valuesFromSchema(self):

        """
        Description: This method is used for fetching the information from the schema_training json file.
        Raises: Exception on failure
        Written By: Shivam Shinde
        Version: 1.0
        Revision: None
        :return: Info such as number of columns or column names in the provided data.
        """

        try:
            with open(self.schema, 'r') as f:
                dict = json.load(f)

            pattern = dict['SampleFileName']
            LengthOfDateStampInFile  = dict['LengthOfDateStampInFile']
            LengthOfTimeStampInFile = dict['LengthOfTimeStampInFile']
            NumberOfColumns = dict['NumberOfColumns']
            ColumnNames = dict['ColumnNames']

            l = open('TrainingLogs/valuesFromSchemaLog','a+')
            message = f"Length of datestamp and timestamp in the file: {LengthOfDateStampInFile} and {LengthOfTimeStampInFile}, number of columns in the data: {NumberOfColumns}"
            self.logger.log(l,message)
            l.close()
            return LengthOfDateStampInFile, LengthOfTimeStampInFile, NumberOfColumns, ColumnNames

        except Exception as e:
            l = open('TrainingLogs/valuesFromSchemaLog', 'a+')
            message1 = f"Exception occurred while fetching the info about the data from the schema_training json file. Exception: {str(e)}"
            self.logger.log(l,message1)
            message2 = f"Fetching info about the data from the schema training json file unsuccessful"
            self.logger.log(l,message2)
            l.close()
            raise e


    def manualRegexCreation(self):

        """

        Description: This method is used for the creation of the regular expression which should be matched with the data file name provided by the client.
        Raises: Exception on failure
        Writen by: Shivam Shinde
        Version: 1.0
        Revision: None
        :return: Python regular expression

        """

        regex = r"flight\_fare\_\d{8}\_\d{6}\.csv"
        return regex

    def createDirectoryForGoodAndBadRawData(self):

        """
        Description: This method is used to create directory for the good (data which passed the validation) and bad date (data which did not pass the validation)
        Raises: Exception on failure
        Writen by: Shivam Shinde
        Version: 1.0
        Revision: None

        """
        try:
            path = os.path.join('Training_raw_data_validated/','GoodData/')
            if not os.path.isdir(path):
                os.path.makedirs(path)

            path = os.path.join('Training_raw_data_validated/','BadData/')
            if not os.path.isdir(path):
                os.path.makedirs(path)

        except OSError as oe:
            f = open('TrainingLogs/GeneralLogs','a+')
            message = f"Exception occurred while creating directories for the bad and good training validated data. Exception: {str(oe)}"
            self.logger.log(f,message)
            f.close()
            raise oe

    def deleteExistingGoodRawTrainingDataFolder(self):
        
        """
        Description: This method is used to delete the directory of the raw good (data which passed the validation) training data
        Raises: Exception on failure
        Writen By: Shivam Shinde
        Version: 1.0
        Revision: None
        
        """
        
        try:
            path = 'Training_raw_data_validated/'
            if os.path.isdir(os.path.join(path,'GoodData/')):
                shutil.rmtree(os.path.join(path,'GoodData/'))

            f = open("TrainingLogs/GeneralLogs", "a+")
            message = "Deleted the folder for good raw training data successfully."
            self.logger.log(f,message)
            f.close()

        except OSError as oe:
            f = open("TrainingLogs/GeneralLogs", "a+")
            message = f"Exception occurred while deleting the folder for good raw training data. Exception: {str(oe)}"
            self.logger.log(f, message)
            f.close()
            raise oe

    def deleteExistingBadRawTrainingDataFolder(self):

        """
        Description: This method is used to delete the directory of the raw bad (data which did not pass the validation) training data
        Raises: Exception on failure
        Writen By: Shivam Shinde
        Version: 1.0
        Revision: None

        """

        try:
            path = 'Training_raw_data_validated/'
            if os.path.isdir(os.path.join(path, 'BadData/')):
                shutil.rmtree(os.path.join(path, 'BadData/'))

            f = open("TrainingLogs/GeneralLogs", "a+")
            message = "Deleted the folder for bad raw training data successfully."
            self.logger.log(f, message)
            f.close()

        except OSError as oe:
            f = open("TrainingLogs/GeneralLogs", "a+")
            message = f"Exception occurred while deleting the folder for bad raw training data. Exception: {str(oe)}"
            self.logger.log(f, message)
            f.close()
            raise oe

