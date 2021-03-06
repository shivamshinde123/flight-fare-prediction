from Logging.logging import Logger
from Prediction_DB_Operation.dataInsertionIntoDB_prediction import DBOperationsPrediction
from Prediction_Raw_Data_Transformation.RawDataTransformation_prediction import RawPredictionDataTransformation
from Prediction_raw_data_validation.rawDataValidation_prediction import rawPredictionDataValidation


class PredictionValidationAndDBInsertion:

    """

    Description: This class is used to validate the data received from the client and for the insertion of validated good data into the database.

    Written By: Shivam Shinde

    Version: 1.0

    Revision: None

    """

    def __init__(self,path):
        self.raw_data_validation = rawPredictionDataValidation(path)
        self.raw_data_transformation = RawPredictionDataTransformation()
        self.raw_data_db_insertion = DBOperationsPrediction()
        self.file_object = open('PredictionLogs/trainingValidationAndDBInsertion.txt', 'a+')
        self.logger = Logger()

    def prediction_validation_and_db_insertion(self):


        """

        Description: This method is used to validate the training data provided by the client and its insertion into
        the database after its validation.

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :return: True

        """
        try:
            self.logger.log(self.file_object, "Prediction data validation started!!")

            ## extracting feature info from the schema training json file
            LengthOfDateStampInFile, LengthOfTimeStampInFile, NumberOfColumns, ColumnNames = self.raw_data_validation.valuesFromSchema()

            ## getting regular expression pattern to match with the data file name
            reg_exp = self.raw_data_validation.manualRegexCreation()

            ## validating the data file name
            self.raw_data_validation.validateTrainingDataFileName(reg_exp)

            ## validating the number of columns
            self.raw_data_validation.validateNumberOfColumns(10)

            ## checking if there is any column with all of its values are missing i.e. nan
            self.raw_data_validation.validateMissingValuesInWholeColumn()

            ## checking the format of the dates in  the Date_of_Journey column of the dataframe
            self.raw_data_validation.validatingDateFormat()

            self.logger.log(self.file_object, "Validation of the raw prediction data completed!!")

            self.logger.log(self.file_object, "Performing data transformation on  the data before inserting it into the database so that there won't be any exception..")

            self.logger.log(self.file_object, "Adding the quotes to the data values in the columns with the string datatype...")
            self.raw_data_transformation.addingQuotesToStringColumns()

            self.logger.log(self.file_object,"Removing the hyphen from the column names (headers) of the data...")
            self.raw_data_transformation.removeHyphenFromColumnName()


            self.logger.log(self.file_object, "Starting the database operations...")
            self.logger.log(self.file_object, "Creating a table into the database...")
            self.raw_data_db_insertion.createTableIntoDb(ColumnNames)
            self.logger.log(self.file_object, "Created table into the database...")

            self.logger.log(self.file_object, "Inserting the data into created table...")
            self.raw_data_db_insertion.insertGoodDataIntoTable()
            self.logger.log(self.file_object, "Data insertion into the table completed successfully...")

            self.logger.log(self.file_object,
                            "Removing good data folder as good data insertion into the database is completed...")
            self.raw_data_validation.deleteExistingGoodRawTrainingDataFolder()
            self.logger.log(self.file_object, "Good data folder removed successfully...")

            self.logger.log(self.file_object, "Moving the bad data folder to the archived bad data folder...")
            self.raw_data_validation.moveBadDataFilesToArchievedBad()
            self.logger.log(self.file_object,
                            "Bad data moved to archived data folder successfully and also deleted bad data folder...")

            self.logger.log(self.file_object, "Getting the raw data from the database as a csv file...")
            self.raw_data_db_insertion.getDataFromDbTableIntoCSV()

            ## closing the file object
            self.file_object.close()

            ## returning True to validate the successful completion of the validation and database insertion
            return True

        except  Exception as e:
            self.logger.log(self.file_object, f"Exception occurred in validation or database insertion step. Exception: {str(e)}")
            self.file_object.close()
            raise e

