import os

import pandas as pd

from Logging.logging import Logger


class RawTrainingDataTransformation:

    """

    Description: This class includes the methods which transforms the data so that there won't be any exception while
    dumping the data into the database.

    Written By: Shivam Shinde

    Version: 1.0

    Revision: None

    """

    def __init__(self):
        self.logger = Logger()

    def addingQuotesToStringColumns(self):

        """

        Description: This method is used to add the quotes to the columns which contains the string values. This is
        done so that there won't be any exception while adding this data to the database

        Written By: Shivam Shinde
        Version: 1.0

        Revision: None

        :return: None

        """

        f = open('TrainingLogs/RawTrainingDataTransformation.txt','a+')
        try:
            for file in os.listdir('Training_raw_data_validated/GoodData/'):
                csv_file = pd.read_csv('Training_raw_data_validated/GoodData/' + file)
                column_lst = ['Airline','Date_of_Journey','Source','Destination','Route','Dep_Time','Arrival_Time','Duration','Total_Stops','Additional_Info']
                for column in column_lst:
                    csv_file[column] = csv_file[column].apply(lambda a: "'" + str(a) + "'")
                csv_file.to_csv('Training_raw_data_validated/GoodData/' + file, index=None, header=True)
                self.logger.log(f,'Quotes added successfully to the values of columns having string values')
            f.close()

        except Exception as e:
            self.logger.log(f,"Exception occurred while adding the quotes to the values in the columns having string values. Exception: "+ str(e))
            f.close()
            raise e

    def removeHyphenFromColumnName(self):

        """

        Description: This method is used to remove the hyphen from the column names of data.

        Written By:  Shivam Shinde

        Version: 1.0

        Revision: None

        :return: None

        """

        f = open('TrainingLogs/RawTrainingDataTransformation.txt', 'a+')
        try:
            for file in os.listdir('Training_raw_data_validated/GoodData/'):
                csv_file = pd.read_csv('Training_raw_data_validated/GoodData/' + file)
                columns = csv_file.columns

                for column in columns:
                    if '-' in str(column):
                        new_column = column.replace('-','')
                        csv_file.rename(columns={column:new_column},inplace=True)
                self.logger.log(f,"Removed the hyphens from the column names successfully")

            f.close()
        except Exception as e:
            self.logger.log(f,"Exception occurred while removing the hyphens from the column names. Exception: "+str(e))
            f.close()
            raise e

