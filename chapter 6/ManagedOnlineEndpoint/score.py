
import os 
import json
import joblib
from pandas import json_normalize
import pandas as pd
import logging

# Called when the service is loaded
def init():
    global model
    # Get the path to the deployed model file and load it
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    model = joblib.load(model_path)
    logging.info("Init complete")

# Called when a request is received
def run(raw_data):
    dict= json.loads(raw_data)
    df = json_normalize(dict['raw_data']) 
    y_pred = model.predict(df)
    print(type(y_pred))
    
    result = {"result": y_pred.tolist()}
    return result
