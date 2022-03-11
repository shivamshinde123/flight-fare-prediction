import numpy as np
import pandas as pd



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

    def removeUnnecessaryFeatureColumn(self,column_name):

        """
        Description: This method is used to remove any unnecessary columns from the dataframe

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param column_name: Name of column that needs to be removed from the dataframe
        :return: None

        """
        self.df.drop(columns=[column_name], inplace=True)


    def datatypeToDatetime(self,column_name):

        """
        Description: This method is used to change the datatype of a column (Date_of_Journey) to datetime

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param column_name: Name of the column for which the datatype needs to be changed to datetime
        :return: None

        """
        self.df[column_name] = pd.to_datetime(self.df[column_name], format="%d/%m/%Y")


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

        self.df['Day_of_Journey'] = self.df['Date_of_Journey'].dt.day
        self.df['Month_of_Journey'] = self.df['Date_of_Journey'].dt.month
        self.df['Year_of_Journey'] = self.df['Date_of_Journey'].dt.year
        self.df.drop(columns=['Date_of_Journey'], axis=1, inplace=True)
