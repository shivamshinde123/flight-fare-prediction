import warnings

from Logging.logging import Logger
from Training_Preprocessing.preprocessingMethods import PreprocessingMethods

warnings.simplefilter(action='ignore', category=FutureWarning)

class Preprocessor:
    """
    Description: This class will contain a method which will implement all the preprocessing techniques on the date
    to make it into clean data

    Written By: Shivam Shinde

    Version: 1.0

    Revision: None
    """

    def __init__(self):
        self.logger_obj = Logger()
        self.process_data = PreprocessingMethods()
        self.file_object = open("../TrainingLogs/preprocessingLogs.txt", "a+")

    def preprocess(self):
        """
        Description: This method will implement all the preprocessing techniques on the date to make the data clean

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :return: None
        """

        try:
            # removing unnecessary columns
            self.process_data.removeUnnecessaryFeatureColumn('Route')
            self.process_data.removeUnnecessaryFeatureColumn('Dep_Time')
            self.process_data.removeUnnecessaryFeatureColumn('Arrival_Time')

            # changing the datatype of the datetime column i.e. Date_of_Journey from string to datetime
            self.process_data.datatypeToDatetime('Date_of_Journey')

            # splitting the datetime column into three newly created columns
            self.process_data.splittingDatetimeColumnIntoThree('Date_of_Journey')

            # converting the duration of flight into minutes and then creating a new column for it while removing the
            # original one
            self.process_data.convertDurationIntoMinutes()

            # converting the Total_Stop column values into the integer from the string
            self.process_data.makeTotalStopsInteger()

            # removing the duplicate rows from the dataframe
            self.process_data.removingdDuplicateRows()

            # correcting the typos in the Additional_Info column
            self.process_data.correctingTyposInAdditionalInfoColumn()

            # replacing the outliers with the nan values
            self.process_data.replacingOutliersWithNan()
            
            # removing the feature column with zero variance
            self.process_data.removingColumnsWithZeroVariance()

            # performing encoding,scaling and null values imputation on the dataframe
            self.process_data.transformPipeline()

            # getting preprocessed X and y
            self.process_data.getPreprocessedXAndy()

            self.logger_obj.log(self.file_object,
                                f"Preprocessing of the data finished successfully!!")

        except Exception as e:
            self.logger_obj.log(self.file_object, f"Exception occurred while preprocessing the data. Exception: {str(e)}")
            raise e




