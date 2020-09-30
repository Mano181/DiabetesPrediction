# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 19:31:53 2020

@author: admin
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model=pickle.load(open('prediction.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    
    
    -------
    None.

    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction=model.predict(final_features)
    
    output = prediction[0]
    if(output==1):
        return render_template('index.html',prediction_text='sorry you have diabetes')
    else:
        return render_template('index.html',prediction_text='Hurrry you not have diabetes')
    #return render_template('index.html',prediction_text='Employee Salary should be $ {}'.format(output))

if __name__=="__main__":
    app.run(debug=True)
