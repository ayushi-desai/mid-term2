import numpy as np
import pickle
import pandas as pd
from flask import Flask, request
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_obj = open("LogisticRegression.pkl","rb")
classifier=pickle.load(pickle_obj)


@app.route('/')
def hello():
    return "Welcome"



@app.route('/predict_test', methods=["POST"])
def predict_test_class():
    
    """Let's predict the class for iris
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: file
        in: formData
        type: file
        required: true
    responses:
        200:
            description: The output values
        
    """
    cancer_df=pd.read_csv(request.files.get("file"))
    prediction=classifier.predict(cancer_df)
    return " The Predicated Class for the TestFile is"+ str(list(prediction))


if __name__=='__main__':
    app.run()