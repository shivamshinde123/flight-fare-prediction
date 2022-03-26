from flask import Flask, render_template, request, Response

from Predictions_using_trained_model import predictionsUsingTheTrainedModels
from modeltraining import modelTraining
from predictionValidationAndDBInsertion import \
    PredictionValidationAndDBInsertion
from trainingValidationAndDBInsertion import trainingValidationAndDBInsertion

app = Flask(__name__)


@app.route('/',methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/training', methods = ['POST'])
def training():

    try:
        if request.json['folderpath'] is not None:

            # requesting the path of file which will be used to train the models for each cluster
            path = request.json['folderpath']

            # checking if the data validation and the database insertion is successful or not.
            # If it is successful then moving on to the clustering followed by training a model for each of the cluster
            if trainingValidationAndDBInsertion(path).training_validation_and_db_insertion() == True:
                m = modelTraining()
                m.trainingModels()
                return Response("Training successfullly completed!!")

    except Exception as e:
        return Response(f"Exception occurred while training models. Exception: {str(e)}")



@app.route('/predictions', methods = ['POST'])
def prediction():

    try:
        if request.json['folderpath'] is not None:

            path = request.json['folderpath']

            if PredictionValidationAndDBInsertion(path).prediction_validation_and_db_insertion() == True:
                p = predictionsUsingTheTrainedModels(path)
                prediction_output_file_path = p.predictUsingModel()

                return Response("Prediction output file saved in the file folder Prediction_output_files/")

        if request.form['filepath'] is not None:

            path = request.form['filepath']

            if PredictionValidationAndDBInsertion(path).prediction_validation_and_db_insertion() == True:
                p = predictionsUsingTheTrainedModels(path)
                prediction_output_file_path = p.predictUsingModel()

                return Response(f"Prediction output file saved at the file location: {prediction_output_file_path}")


    except Exception as e:
        return Response(f"Exception occurred while predicting the flight fares using the saved models")




if __name__ == '__main__':
    app.run(debug=True)









