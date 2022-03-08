import sqlite3
from Logging.logging import Logger
import os
import csv
import shutil

class DBOperations:
    
    """
    Description: This class contains the methods that deal with the database operations.
    Written By: Shivam Shinde 
    Version: 1.0
    Revision: None
    
    """
    
    def __init__(self):
        self.path = "../Databases/"
        self.goodDataPath = "../Training_raw_data_validated/GoodData/"
        self.badDataPath = "../Training_raw_data_validated/BadData/"
        self.logger = Logger()

    def dbConnection(self,databaseName='goodRawDataDb'):

        """
        Description: This method is used to create a connection with the sqlite3 database
        On Failure: Raises exception
        Written By: Shivam Shinde
        Version: 1.0
        Revision: None
        :return: Databases connector

        """

        f = open('../TrainingLogs/DatabaseLogs.txt','a+')
        try:
            if not os.path.isdir(self.path):
                os.makedirs(self.path)
            conn = sqlite3.connect(self.path+databaseName+'.db')
            self.logger.log(f,f"Connection with the database {databaseName} made")
            f.close()
            return conn

        except ConnectionError as ce:
            self.logger.log(f,f"Connection error occurred while creating a connection to the database. Error: {str(ce)}")
            f.close()
            raise ce

    def createTableIntoDb(self,columnNamesDict,databaseName='goodRawDataDb'):

        """

        Description: This method is used create a table in the existing database
        Written By: Shivam Shinde
        On Failure: Raises exception
        Version: 1.0
        Revision: None
        :param databaseName: Name of the database into which table is to be added
        :param columnNamesDict: Dictionary having column names as keys and their datatype as values
        :return: None

        """

        conn = self.dbConnection(databaseName)
        cursor = conn.cursor()

        cursor.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='goodRawData' ''')

        f = open('../TrainingLogs/DatabaseLogs.txt', 'a+')
        try:
            if cursor.fetchone()[0] == 1:
                self.logger.log(f, "Table named goodRawData created in the database goodRawDataDb")
                f.close()
                conn.close()

            else:
                for key in columnNamesDict.keys():
                    datatype = columnNamesDict[key]

                    ## Here in try block we check if the table is existed or not and if it is then add the columns to it
                    ## In catch block, we will create a table
                    try:
                        command = f"""ALTER TABLE goodRawData ADD {key} {datatype}"""
                        cursor.execute(command)
                    except:
                        command = f"""CREATE TABLE goodRawData ({key} {datatype})"""
                        cursor.execute(command)

                conn.commit()
                self.logger.log(f,"Table named goodRawData created successfully in the database goodRawDataDb")
                f.close()
                conn.close()

        except Exception as e:
            self.logger.log(f,f"Exception occurred while creating the table inside the database named goodRawDataDb. Exception: {str(e)}")
            f.close()
            conn.close()
            raise e


    def insertGoodDataIntoTable(self,database='goodRawDataDb'):

        """

        Description: This method is used to add the data into the already created table.
        On Failure: Raises exception
        Written By: Shivam Shinde
        Version: 1.0
        Revision: None
        :param database: Name of the database into which the table is present.
        :return: None

        """

        conn = self.dbConnection(database)
        cursor = conn.cursor()
        f = open('../TrainingLogs/DatabaseLogs.txt', 'a+')
        for file in os.listdir(self.goodDataPath):
            try:
                with open(self.goodDataPath + file, 'r') as p:
                    next(p)
                    reader = csv.reader(p,delimiter='\n')
                    for line in enumerate(reader):
                        for list_ in line[1]:
                            try:
                                cursor.execute(f"INSERT INTO goodRawData values ({list_})")
                                self.logger.log(f, f"{file} File loaded successfully!!")
                                conn.commit()
                            except Exception as e:
                                raise e

            except Exception as e:
                conn.rollback()
                shutil.move(self.goodDataPath+file,self.badDataPath)
                self.logger.log(f,f"Error occurred while inserting the data into the table. Exception: {str(e)}")
                f.close()
                conn.close()
                raise e

            f.close()
            conn.close()



    def getDataFromDbTableIntoCSV(self,database='goodRawDataDb'):


        """

        Description: This method is used to fetch the data from the table inside the database and store it as csv file into some directory.
        Written By: Shivam Shinde
        Version: 1.0
        Revision: None
        :param database: The name of the database into which the table is present.
        :return: None

        """

        self.fileFromDb = "../fileFromDb/"
        self.fileName = "inputFile.csv"
        f = open('../TrainingLogs/DatabaseLogs.txt', 'a+')

        try:
            conn = self.dbConnection()
            cursor = conn.cursor()

            query = "SELECT * FROM goodRawData"
            cursor.execute(query)

            results = cursor.fetchall()

            ## getting the headers of the csv file
            headers = [i[0] for i in cursor.description]

            ## checking whether the output directory for the csv file is present or not. If not then creating one
            if not os.path.isdir(self.fileFromDb):
                os.makedirs(self.fileFromDb)

            ## checking file for writing
            p = open(self.fileFromDb + self.fileName, 'w', newline='')
            csvfile = csv.writer(p, delimiter=',', lineterminator='\r\n',quoting=csv.QUOTE_ALL, escapechar='\\')


            ## adding header and data to the csv file
            csvfile.writerow(headers)
            csvfile.writerow(results)

            self.logger.log(f,"File exported successfully!!")
            f.close()
            conn.close()

        except Exception as e:
            self.logger.log(f,f"Exception occurred while exporting the data file from the database. Exception: {str(e)}")
            f.close()



var = {
    "Airline": "TEXT",
    "Source": "TEXT",
    "Destination": "TEXT",
    "Flight_Duration": "REAL",
    "Total_Stops": "REAL",
    "Additional_Info": "TEXT",
    "Day_of_Journey": "INTEGER",
    "Month_of_Journey": "INTEGER",
    "Year_of_Journey": "INTEGER",
    "Price": "INTEGER"
}

d = DBOperations()
# d.dbConnection()
d.createTableIntoDb(var)
d.insertGoodDataIntoTable()
# d.getDataFromDbTableIntoCSV()