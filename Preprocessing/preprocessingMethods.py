import os
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from Logging.logging import Logger

warnings.simplefilter(action='ignore', category=FutureWarning)

class PreprocessingMethods:
    """
    Description: This class will contain the methods which will be used for the data transformation
    before the date clustering and model training

    Written By: Shivam Shinde

    Version: 1.0

    Revision: None
    """

    def __init__(self):
        self.df = pd.read_csv("../fileFromDb/inputFile.csv")
        self.logger_obj = Logger()
        self.file_object = open("../TrainingLogs/preprocessingLogs.txt", "a+")

    def removeUnnecessaryFeatureColumn(self, column_name):

        """
        Description: This method is used to remove any unnecessary columns from the dataframe

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param column_name: Name of column that needs to be removed from the dataframe
        :return: None

        """
        try:
            # removing the given column from the dataframe
            self.df.drop(columns=[column_name], inplace=True)
            self.logger_obj.log(self.file_object, f"Feature column named {column_name} removed from the dataframe")
        except Exception as e:
            self.logger_obj.log(self.file_object,
                                f"Exception occurred while removing a feature column {column_name} from the "
                                f"dataframe. Exception: {str(e)} ")
            raise e

    def datatypeToDatetime(self, column_name):

        """
        Description: This method is used to change the datatype of a column (Date_of_Journey) to datetime

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param column_name: Name of the column for which the datatype needs to be changed to datetime
        :return: None

        """
        try:
            # converting the datatype of provided column into datetime
            self.df[column_name] = pd.to_datetime(self.df[column_name], format="%d/%m/%Y")
            self.logger_obj.log(self.file_object,
                                f"Datatype of feature column named {column_name} changed to datetime..")

        except Exception as e:
            self.logger_obj.log(self.file_object,
                                f"Exception occurred while changing the datatype of column {column_name} to datetime. "
                                f"Exception: {str(e)}")

            raise e

    def splittingDatetimeColumnIntoThree(self, column_name):

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
            # splitting the Date_of_Journey which contains the whole data into three new columns each of those showing
            # day, month and year of the date
            self.df['Day_of_Journey'] = self.df['Date_of_Journey'].dt.day
            self.df['Month_of_Journey'] = self.df['Date_of_Journey'].dt.month
            self.df['Year_of_Journey'] = self.df['Date_of_Journey'].dt.year
            self.logger_obj.log(self.file_object, "Successfully split the datetime column into three newly created "
                                                  "columns namely Day_of_Journey, Month_of_Journey and Year_of_Journey")

            # removing the original Date_of_Journey column from the dataframe
            self.removeUnnecessaryFeatureColumn("Date_of_Journey")

        except Exception as e:
            self.logger_obj.log(self.file_object,
                                f"Exception occurred while splitting the datetime column into three newly created "
                                f"columns. Exception: {str(e)}")
            raise e

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
            # creating the three possible pattern for the values in the featue column containing duration in hours and
            # minutes
            pattern1 = re.compile(r"(\d+)(h|m)(\s)(\d*)(h|m)*")
            pattern2 = re.compile(r"(\s*)(\d+)(h)")
            pattern3 = re.compile(r"(\s*)(\d+)(m)")

            # creating an empty list which will be used to store the flight duration in minute for every data
            # observation
            min_lst = []

            # calculating the flight duration in minutes for each data observation and then adding it to the created
            # empty list
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

            # removing the original Duration column which contains the flight duration in hours and minutes
            self.removeUnnecessaryFeatureColumn("Duration")

            # adding a new duration column which contains the flight duration in minutes to the dataframe
            train_values = pd.Series(min_lst)
            self.df.insert(loc=7, column="Flight_Duration", value=train_values)
            self.logger_obj.log(self.file_object,
                                "Added a new column named Flight_Duration which contains the flight duration in "
                                "minutes..")

        except Exception as e:
            self.logger_obj.log(self.file_object,
                                f"Exception occurred while converting the flight duration into minutes. Exception: {str(e)}")
            raise e

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
            # creating a dictionary which will be used to map against the Total_Stops column of the dataframe
            dict1 = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}

            # mapping the dict dictionary against the Total_Stops column of the dataframe
            self.df['Total_Stops'] = self.df['Total_Stops'].map(dict1)
            self.logger_obj.log(self.file_object,
                                "Successfully changed the datatypes of the values from the Total_Stops columns to "
                                "integer")

        except Exception as e:
            self.logger_obj.log(self.file_object,
                                f"Exception occurred while making values from column Total_Stops into integer. "
                                f"Exception: {str(e)}")
            raise e

    def removingdDuplicateRows(self):

        """
        Description: This method is used to remove the duplicate rows from the data
        
        Written By: Shivam Shinder
        
        Version: 1.0
        
        Revision: None
        
        :return: None
        
        """

        try:
            # removing the duplicate data observations from the dataframe
            self.df.drop_duplicates(inplace=True)
            self.logger_obj.log(self.file_object, "Removing of duplicate rows successful..")
        except Exception as e:
            self.logger_obj.log(self.file_object,
                                f"Exception occurred while removing duplicate rows from the dataframe. Exception: {str(e)}")
            raise e

    def splittingTheDataframeIntoXandy(self):

        """
        Description: This method is used to split the dataframe into independent and dependent features

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :return: X (dataframe with independent features) and y (dataframe with target feature)
        """

        try:
            # splitting the dataframe into the independent features (X) and dependent features (y)
            X = self.df.drop(columns=['Price'], axis=1)
            y = self.df['Price']
            return X, y

        except Exception as e:
            self.logger_obj.log(self.file_object,
                                f"Exception occurred while splitting the dataframe into independent and dependent "
                                f"features. Exception: {str(e)}")
            raise e

    def findingNamesOfNumericalAndCategoricalColumns(self):

        """

        Description: This method is used to identify the names of numerical and categories columns in the dataframe

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :return: Two lists each containing the names of categorical and numerical columns respectively
        """

        try:
            # splitting the dataframe into the independent features (X) and dependent features (y)
            X, y = self.splittingTheDataframeIntoXandy()

            # creating a list of names of the categorical features
            categorical_features = [feature for feature in X.columns if X[feature].dtypes == 'O']

            # creating a list of names of the numerical columns
            numerical_features = [feature for feature in X.columns if feature not in categorical_features]

            # returning the lists containing the names of categorical and numerical columns
            return categorical_features, numerical_features

        except Exception as e:
            self.logger_obj.log(self.file_object, f"Exception occurred while finding the names of the categorical and "
                                                  f"numerical columns in the data. Exception: {str(e)}")
            raise e

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
            # correcting the typos in the Additional_Info column of the dataframe
            self.df['Additional_Info'] = np.where(self.df['Additional_Info'] == "No Info", "No info",
                                                  self.df['Additional_Info'])
            self.logger_obj.log(self.file_object, "Successfully corrected the typos in the Additional_Info column..")
        except Exception as e:
            self.logger_obj.log(self.file_object,
                                f"Exception occurred while correcting the typo in the Additional_Info column. "
                                f"Exception: {str(e)}")
            raise e

    def replacingOutliersWithNan(self):

        """
        Description: This method is used to replace the outliers with the null values

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :return: None
        """

        try:
            # fetching the lists which contain the names of categorical and numerical columns
            cat_feat, num_feat = self.findingNamesOfNumericalAndCategoricalColumns()

            # finding the first and third quantile followed by the inter-quartile range for each of the numerical column
            for feature in num_feat:
                Q1 = self.df[feature].quantile(0.25)
                Q3 = self.df[feature].quantile(0.75)
                IQR = Q3 - Q1
                # replacing the outliers in the numerical column using the null value
                self.df[feature] = np.where(self.df[feature] > (Q3 + 1.45 * IQR), np.nan, self.df[feature])
                self.df[feature] = np.where(self.df[feature] < (Q1 - 1.45 * IQR), np.nan, self.df[feature])

            self.logger_obj.log(self.file_object, "Successfully replaced the outliers in the columns with null..")

        except Exception as e:
            self.logger_obj.log(self.file_object,
                                F"Exception occurred while replacing outlier with the null values. Exception: {str(e)}")
            raise e

    def removingColumnsWithZeroVariance(self):

        """
        Description: This method is used to remove the column from the dataframe which are having zero variance.

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None
        :return: None
        """

        try:
            # fetching the lists which contain the names of categorical and numerical columns
            cat_feat, num_feat = self.findingNamesOfNumericalAndCategoricalColumns()
            self.logger_obj.log(self.file_object, "Checking for the column with zero variance..")
            # finding and removing the column from the dataframe with the zero variance
            for feature in num_feat:
                if self.df[feature].var() == 0:
                    self.removeUnnecessaryFeatureColumn(feature)

        except Exception as e:
            self.logger_obj.log(self.file_object,
                                f"Exception occurred while removing the column with zero variance. Exception: {str(e)}")
            raise e

    def splittingTheDataIntoTrainAndTest(self):

        """
        Description: This method is used to split the dataframe into train and test dataframes

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        Returns
                 -  X_train (independent features data for training)

                 -  X_test (independent features data for testing)

                 -  y_train (dependent feature data for training)

                 -  y_test (dependent feature data for testing)
        """

        try:
            # splitting the dataframe into the independent features (X) and dependent features (y)
            X, y = self.splittingTheDataframeIntoXandy()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=199)
            self.logger_obj.log(self.file_object, "Split the data into training data and testing data")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            self.logger_obj.log(self.file_object,
                                "Exception occurred while splitting the data into training and testing data")
            raise e

    def transformPipeline(self):

        """
        Description:
                    This function is used to
                                            -   Impute the missing values in the dataframe
                                            -   Encode categorical columns
                                            -   Perform scaling on required numerical columns

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :return: Preprocessed dataframe
        """
        try:
            # splitting X and y into training and testing data
            X, y = self.splittingTheDataframeIntoXandy()

            # creating numerical pipeline for the already encoded categorical columns (Since they are encoded, they
            #  have been named as numeric for the pipeline)
            num_pipeline1 = Pipeline([
                ('num_most_frequent_imputation', SimpleImputer(strategy="most_frequent"))
            ])

            # creating a numerical pipeline for the numerical columns which needs the mean imputation and scaling
            num_pipeline2 = Pipeline([
                ('mean_imputation', SimpleImputer(strategy="mean")),
                # ('std_scaling', StandardScaler())
            ])

            # creating a categorical pipeline
            cat_pipeline = Pipeline([
                ('most_frequent_imputation', SimpleImputer(strategy="most_frequent")),
                ('label_encoding', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=20))
            ])

            # combining two numerical and one categorical pipelines using sklearn ColumnTransformer
            full_pipeline = ColumnTransformer([
                ('num1', num_pipeline1, ["Total_Stops", "Day_of_Journey", "Month_of_Journey"]),
                ('num2', num_pipeline2, ["Flight_Duration"]),
                ('cat', cat_pipeline, ["Airline", "Source", "Destination", "Additional_Info"])
            ])

            # fitting and transforming the X using full_pipeline and then returning it
            XPreprocessed = pd.DataFrame(full_pipeline.fit_transform(X),
                                         columns=["Total_Stops", "Day_of_Journey", "Month_of_Journey",
                                                  "Flight_Duration",
                                                  "Airline", "Source", "Destination", "Additional_Info"])

            yDataframe = pd.DataFrame(y,columns=['Price'])

            if not os.path.exists("../PreprocessedDara/"):
                os.makedirs("../PreprocessedDara/")

            XPreprocessed.to_csv("../PreprocessedDara/XPreprocessed.csv",header=True,index=False)
            y.to_csv("../PreprocessedDara/yDataframe.csv",header=True,index=False)

        except Exception as e:
            self.logger_obj.log(self.file_object, f"Exception occurred while implementing the data preprocessing "
                                                  f"pipeline. Exception: {str(e)}")
            self.file_object.close()
            raise e


    def getPreprocessedXAndy(self):

        """
        Description: This method is used to get the preprocessed X and y

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None
        :return:  None
        """
        try:
            self.logger_obj.log(self.file_object,"Exporting preprocessed X and y")
            X = pd.read_csv("../PreprocessedData/XPreprocessed.csv")
            y = pd.read_csv(("../PreprocessedData/yDataframe.csv"))

            return X, y
        except Exception as e:
            self.logger_obj.log(self.file_object, "Exception occurred while exporting the preprocessed X and y")