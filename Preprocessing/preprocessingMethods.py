import numpy as np
import pandas as pd
import re
from Logging.logging import Logger


class PreprocessingMethods:

    """
    Description: This class will contain the methods which will be used for the data transformation
    before the date clustering and model training

    Written By: Shivam Shinde

    Version: 1.0

    Revision: None
    """

    def __int__(self):
        self.df = pd.read_csv("../fileFromDb/inputFile.csv")
        self.file_object = open("../TrainingLogs/preprocessingLogs.txt", "a+")
        self.logger = Logger()


    def removeUnnecessaryFeatureColumn(self,column_name):

        """
        Description: This method is used to remove any unnecessary columns from the dataframe

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param column_name: Name of column that needs to be removed from the dataframe
        :return: None

        """
        try:
            self.df.drop(columns=[column_name], inplace=True)
            self.logger.log(self.file_object, f"Feature column named {column_name} removed from the dataframe")
        except Exception as e:
            self.logger.log(self.file_object, f"Exception occurred while removing a feature column {column_name} from the dataframe. Exception: {str(e)} ")



    def datatypeToDatetime(self,column_name):

        """
        Description: This method is used to change the datatype of a column (Date_of_Journey) to datetime

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param column_name: Name of the column for which the datatype needs to be changed to datetime
        :return: None

        """
        try:
            self.df[column_name] = pd.to_datetime(self.df[column_name], format="%d/%m/%Y")
            self.logger.log(self.file_object, f"Datatype of feature column named {column_name} changed to datetime..")

        except Exception as e:
            self.logger.log(self.file_object, f"Exception occurred while changing the datatype of column {column_name} to datetime. Exception: {str(e)}")



    def splittingDatetimeColumnIntoThree(self,column_name):

        """
        Description: This method is used to create three new columns from the datetime column from the dataframe.
        Method also removes the original datetime column after creating three new columns.

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param column_name: Name of datetime column that needs to splitted into 3 columns
        :return: None

        """

        try:
            self.df['Day_of_Journey'] = self.df['Date_of_Journey'].dt.day
            self.df['Month_of_Journey'] = self.df['Date_of_Journey'].dt.month
            self.df['Year_of_Journey'] = self.df['Date_of_Journey'].dt.year
            self.logger.log(self.file_object, "Successfully split the datetime column into three newly created columns.")
            self.removeUnnecessaryFeatureColumn("Date_of_Journey")
            self.logger.log(self.file_object, f"Original datetime column removed successfully..")

        except Exception as e:
            self.logger.log(self.file_object, f"Exception occurred while splitting the datetime column into three newly created columns. Exception: {str(e)}")

    def convertDurationIntoMinutes(self):

        """
        Description: This method is used to create a new column named Flight_Duration which contains the flight durarion
        in minutes from the already present Duration column which has duration in the hours and minutes. Function also
        removes the original Duration column after removing the Flight_Duration column

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :return: None

        """

        try:
            pattern1 = re.compile(r"(\d+)(h|m)(\s)(\d*)(h|m)*")
            pattern2 = re.compile(r"(\s*)(\d+)(h)")
            pattern3 = re.compile(r"(\s*)(\d+)(m)")

            self.logger.log(self.file_object, "Created patterns which could match the values in the Duration column")

            min_lst = []
            for i in range(self.df.shape[0]):
                if 'h' in self.df.loc[i, "Duration"] and 'm' in self.df.loc[i, "Duration"]:
                    matchobj = re.match(pattern1, self.df.loc[i, "Duration"])
                    hour = int(matchobj.group(1))
                    minute = int(matchobj.group(4))
                    total_min = 60 * hour + minute
                    min_lst.append(total_min)
                elif 'h' in self.df.loc[i, "Duration"] and 'm' not in self.df.loc[i, "Duration"]:
                    matchobj = re.match(pattern2, self.df.loc[i, "Duration"])
                    hour = int(matchobj.group(2))
                    min_lst.append(60 * hour)
                elif 'h' not in self.df.loc[i, "Duration"] and 'm' in self.df.loc[i, "Duration"]:
                    matchobj = re.match(pattern3, self.df.loc[i, "Duration"])
                    minute = int(matchobj.group(2))
                    min_lst.append(minute)
                else:
                    min_lst.append(self.df.loc[i, "Duration"])

            self.logger.log(self.file_object, "Created a list which contains all the duration in minutes successfully..")

            self.removeUnnecessaryFeatureColumn("Duration")
            self.logger.log(self.file_object, "Removing the original Duratin column from the dataframe..")

            train_values = pd.Series(min_lst)
            self.df.insert(loc=7, column="Flight_Duration", value=train_values)
            self.logger.log(self.file_object, "Added a new column named Flight_Duration which contains the flight duration in minutes..")

        except Exception as e:
            self.logger.log(self.file_object, f"Exception occurred while converting the flight duration into minutes. Exception: {str(e)}")


    def makeTotalStopsInteger(self):

        """
        Description: This method is used to change the datatype of the Total_Stops from string to integer by removing
        strings such as stops from the data

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :return: None
        """

        try:
            dict1 = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}

            self.df['Total_Stops'] = self.df['Total_Stops'].map(dict1)
            self.logger.log(self.file_object, "Suceessfully changed the datatypes of the values from the Total_Stops columns..")

        except Exception as e:
            self.logger.log(self.file_object, f"Exception occurred while making values from column Total_Stops into integer. Exception: {str(e)}")


    def removingdDuplicateRows(self):

        """
        Description: This method is used to remove the duplicate rows from the data
        
        Written By: Shivam Shinder
        
        Version: 1.0
        
        Revision: None
        
        :return: None
        
        """
        
        try:
            self.df.drop_duplicates(inplace=True)
            self.logger.log(self.file_object, "Removing of duplicate rows successful..")
        except Exception as e:
            self.logger.log(self.file_object,f"Exception occurred while removing duplicate rows from the dataframe. Exception: {str(e)}")


    def splittingTheDataframeIntoXandy(self):

        """
        Description: This method is used to split the dataframe into independent and dependent features

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :return: X (dataframe with independent features) and y (dataframe with target feature)
        """

        try:
            X = self.df.drop(columns=['Price'], axis=1)
            y = self.df['Price']
            self.logger.log(self.file_object, "Successfully split the dataframe into independent and dependent features..")
            self.logger.log(self.file_object, "Returning X, y..")
            return X, y

        except Exception as e:
            self.logger.log(self.file_object, f"Exception occurred while splitting the dataframe into independent and dependent features. Exception: {str(e)}")


    # def findingNamesOfNumericalAndCategoricalColumns(self):
    #
    #
    #     """
    #
    #     Description: This method is used to identify the names of numerical and categories columns in the dataframe
    #
    #     Written By: Shivam Shinde
    #
    #     Version: 1.0
    #
    #     Revision: None
    #
    #     :return: Two lists each containing the names of numerical and categorical columns respectively
    #     """
    #
    #     X, y = self.splittingTheDataframeIntoXandy()
    #
    #     categorical_features = [feature for feature in X.columns if X[feature].dtypes == 'O']
    #     numerical_features = [feature for feature in X.columns if feature not in categorical_features]
    #
    #     return categorical_features, numerical_features

    def correctingTyposInAdditionalInfoColumn(self):

        """
        Description: In Additional_Info column, some values which should be 'No info' are misspelled as 'No Info'.
        This function will correct this typo

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :return: None

        """

        try:
            self.df['Additional_Info'] = np.where(self.df['Additional_Info'] == "No Info", "No info", self.df['Additional_Info'])
            self.logger.log(self.file_object, "Successfully corrected the typos in the Additional_Info column..")
        except Exception as e:
            self.logger.log(self.file_object, f"Exception occurred while correcting the typo in the Additional_Info column. Exception: {str(e)}")


    def replacingOutliersWithNan(self):

        """
        Description: This method is used to replace the outliers with the null values

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :return: None
        """

        try:
            X, y = self.splittingTheDataframeIntoXandy()

            for feature in X.columns:
                Q1 = self.df[feature].quantile(0.25)
                Q3 = self.df[feature].quantile(0.75)
                IQR = Q3 - Q1

                self.df[feature] = np.where((self.df[feature] > (Q3 + 1.45 * IQR) | self.df[feature] < (Q1 - 1.45 * IQR)), np.nan, self.df[feature])

            self.logger.log(self.file_object, "Successfully replaced the outliers in the columns with null..")

        except Exception as e:
            self.logger.log(self.file_object, F"Exception occurred while replacing outlier with the null values. Exception: {str(e)}")

    def fillingNullValues(self):

        """
        Description: This method is used to fill the null values in the dataframe using mean of the non-null data

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :return: None
        """
        try:
            X, y = self.splittingTheDataframeIntoXandy()

            for feature in X.columns:
                self.df[feature].fillna(self.df[feature].mean(), inplace=True)

            self.logger.log(self.file_object, "Successfully filled the null values from the columns with the mean of non-null values..")

        except  Exception as e:
            self.logger.log(self.file_object, f"Exception occurred while filling null values. Exception: {str(e)}")


    def importProcessedFile(self):

        """
        Description: This method is used to export the processed file  as csv file

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :return: None
        """

        try:
            self.df.to_csv('inputFileProcessed.csv',header=True,index=False)
            self.logger.log(self.file_object, "Successfully exported the preprocessed input file..")

        except Exception as e:
            self.logger.log(self.file_object, f"Exception occurred while exporting the preprocessed input file. Exception: {str(e)}")













