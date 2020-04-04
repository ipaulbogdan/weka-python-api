from wekapy import *

model = Model(classifier_type = 'trees.J48')
model.train(training_file = 'loanPrediction.arff')
model.test(test_file = 'test.arff')

for prediction in model.predictions:
    print(prediction)