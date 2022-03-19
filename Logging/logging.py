from datetime import datetime

class Logger:
    """
                Description: This method is used for logging purpose in the code
                Written By: Shivam Shinde
                Version: 1.0
                Revision: None
    """

    def __init__(self):
             pass

    def log(self, file_object, log_message):


        """
            Description: This method is used for logging purpose in the code
            Written By: Shivam Shinde
            Version: 1.0
            Revision: None
        """
        try:
            self.now = datetime.now()
            self.date = self.now.date()
            self.time = self.now.strftime("%H:%M:%S")

            file_object.write(str(self.date) + " / " + str(self.time) + "  ----->  " + str(log_message) + "\n")

        except Exception as e:
            raise e