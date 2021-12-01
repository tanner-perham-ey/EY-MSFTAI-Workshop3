import json
import joblib
import numpy as np
from azureml.core.model import Model

# Called when the service is loaded
def init():
    global model
    # Get the path to the deployed model file and load it
    model_path = Model.get_model_path('porto_seguro_safe_driver_model.pkl')
    model = joblib.load(model_path)

# Called when a request is received
def run(raw_data):
    # Get the input data as a numpy array
    data = np.array(json.loads(raw_data)['data'])
    # Get a prediction from the model
    predictions = model.predict(data)
    # Get the corresponding classname for each prediction (0 or 1)
    classnames = ['no claim', 'claim']
    predicted_classes = []
    probabilities = [prediction for prediction in predictions]

    # TODO: 
    """Write a for-loop which appends classnames[0] to the predicted_classes
       list if the probability < 0.5, else to classnames[1]
    """
    for p in probabilities:
        if p < 0.5:
            predicted_classes.append(classnames[0])
        else:
            predicted_classes.append(classnames[1])
    
    # Return the predictions
    return (probabilities, predicted_classes)
