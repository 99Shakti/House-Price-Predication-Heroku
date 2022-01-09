# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 21:57:27 2022

@author: rash0007
"""
import numpy as np
import pickle
from flask import Flask,render_template,request

app=Flask(__name__)
model=pickle.load(open('HP','rb'))

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    predication=model.predict(final_features)

    output=round(predication[0],2) 
    
    return render_template('index.html',predication_text="The price of house should be $ {}".format(output))

if __name__=='main':
    app.run()