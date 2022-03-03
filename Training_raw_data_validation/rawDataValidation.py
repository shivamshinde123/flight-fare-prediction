import os
import re
import shutil
from logging import Logger
import json
import datetime

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

            l = open('TrainingLogs/valuesFromSchemaLog.txt','a+')
            message = f"Length of datestamp and timestamp in the file: {LengthOfDateStampInFile} and {LengthOfTimeStampInFile}, number of columns in the data: {NumberOfColumns}"
            self.logger.log(l,message)
            l.close()
            return LengthOfDateStampInFile, LengthOfTimeStampInFile, NumberOfColumns, ColumnNames

        except Exception as e:
            l = open('TrainingLogs/valuesFromSchemaLog.txt', 'a+')
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
            f = open('TrainingLogs/GeneralLogs.txt','a+')
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

            f = open("TrainingLogs/GeneralLogs.txt", "a+")
            message = "Deleted the folder for good raw training data successfully."
            self.logger.log(f,message)
            f.close()

        except OSError as oe:
            f = open("TrainingLogs/GeneralLogs.txt", "a+")
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

            f = open("TrainingLogs/GeneralLogs.txt", "a+")
            message = "Deleted the folder for bad raw training data successfully."
            self.logger.log(f, message)
            f.close()

        except OSError as oe:
            f = open("TrainingLogs/GeneralLogs,txt", "a+")
            message = f"Exception occurred while deleting the folder for bad raw training data. Exception: {str(oe)}"
            self.logger.log(f, message)
            f.close()
            raise oe


    def moveBadDataFilesToArchievedBad(self):

        """

        Description: This method is used to move the bad training data files to the archived bad data folder.
        Raises: Exception on failure
        Writen By: Shivam Shinde
        Version: 1.0
        Revision: None

        """
        now = datetime.now()
        date = now.date()
        time = now.time()
        try:
            source = 'Training_raw_data_validated/BadData/'
            if os.path.isdir(source):

                path = 'TrainingRawBadDataArchived/'
                if not os.path.isdir(path):
                    os.path.makedirs(path)

                destination = 'TrainingRawBadDataArchived/BadData_' + str(date) + "_" + str(time)
                if not os.path.isdir(destination):
                    os.path.makedirs(destination)

                files = os.path.listdir(source)
                for f in files:
                    if f not in os.path.listdir(destination):
                        shutil.move(source + f,destination)
                p = open('TrainingLogs/GeneralLogs.txt','a+')
                self.logger.log(p,"Bad data file were successfully moved from training raw data folder to the archived bad foler")
                p.close()

                if os.path.isdir(source):
                    shutil.rmtree(source)
                p = open('TrainingLogs/GeneralLogs.txt','a+')
                self.logger.log(p,"Bad data folder from training raw data folder deleted successfully.")
                p.close()

        except OSError as oe:
            p = open('TrainingLogs/GeneralLogs.txt', 'a+')
            self.logger.log(p,"Exception occurred while moving the bad data files from training raw data folder to the archived bad data folder")
            p.close()
            raise oe


    def validateTrainingDataFileName(self,regex):

        """
        Description: This method is used to validate the data file name provided by the client.
        Written By: Shivam Shinde
        Version: 1.0
        Revision: None
        :param regex: This parameter is the regular expression that would be matched against the data file name provided by the client.
        :return: None

        """

        ## Deleting the good and bad data folders in case previous execution failed
        self.deleteExistingGoodRawTrainingDataFolder()
        self.deleteExistingBadRawTrainingDataFolder()

        raw_data_files = [f for f in os.path.listdir(self.Batch_Directory)]

        f = open('TrainingLogs/RawDataFileNameValidation.txt''a+')

        try:
            for file in raw_data_files:

                ## creating new directories for good and bad data
                self.createDirectoryForGoodAndBadRawData()

                if re.match(regex, file):
                    shutil.move(f"self.Batch_Directory+{file}", "Training_raw_data_validated/GoodData/")
                    self.logger.log(f, "Valid file name! File moved to the good data folder")
                    f.close()
                else:
                    self.logger.log(f, "Invalid file name! File moved to the bad data folder")
                    f.close()

        except Exception as e:
            self.logger.log(f, f"Exception occurred while validating the file name. Exception: {str(e)}")
            f.close()
            raise e











