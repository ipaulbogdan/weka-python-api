import os
from flask import Flask, request
from flask_restful import Resource, Api
from wekapy import Model

app = Flask(__name__)
api = Api(app)

file_name = 'loanPrediction.arff'
model = Model(classifier_type = 'trees.J48')

model_file_destination = model.model_dir + '/' + file_name

class DataSetUpload(Resource):
    def post(self):
        dataSet = request.files['dataSet']
        
        dataSet.save(model_file_destination)
        
        model.load_model(model_file_destination)

        return {'response': 'good job'}, 200

class Prediction(Resource):
    def post(self):
        print(model.training_instances)

        return None

api.add_resource(DataSetUpload, '/data-set')
api.add_resource(Prediction, '/predict')

if __name__ == '__main__':
    app.run(debug = True)
