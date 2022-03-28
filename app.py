import pathlib
from flask import Flask, redirect, render_template, request, Response, url_for
from datetime import datetime, timedelta
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
        if request.json is not None:

            # requesting the path of file which will be used to train the models for each cluster
            path = request.json['folderpath']
            path = pathlib.Path(path)

            # checking if the data validation and the database insertion is successful or not.
            # If it is successful then moving on to the clustering followed by training a model for each of the cluster
            if trainingValidationAndDBInsertion(path).training_validation_and_db_insertion() == True:
                m = modelTraining()
                m.trainingModels()
                return Response("Training successfullly completed!!")

    except Exception as e:
        return Response(f"Exception occurred while training models. Exception: {str(e)}")


@app.route('/results')
def results():
    return render_template('results.html')


@app.route('/predictions', methods = ['POST'])
def prediction():

    try:
        if request.json is not None:

            path = request.json['folderpath']
            path = pathlib.Path(path)

            start_time = datetime.now()
            if PredictionValidationAndDBInsertion(path).prediction_validation_and_db_insertion() == True:
                p = predictionsUsingTheTrainedModels(path)
                p.predictUsingModel()
                finish_time = datetime.now()

                time_required = finish_time - start_time
                print(f"Time required for the predictions: {time_required}")

                return Response(f"Prediction output file saved in the file folder Prediction_output_files/")

        elif request.form is not None:

            path = request.form['folderpath']
            path = pathlib.Path(path)

            if PredictionValidationAndDBInsertion(path).prediction_validation_and_db_insertion() == True:
                p = predictionsUsingTheTrainedModels(path)
                p.predictUsingModel()

                return redirect(url_for('results'))

    except Exception as e:
        return Response(f"Exception occurred while predicting the flight fares using the saved models")




if __name__ == '__main__':
    app.run()









