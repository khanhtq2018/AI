# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 18:35:38 2018
@author: KhanhTQ7
"""

from flask import Flask, render_template, request
import numpy as np
from keras.preprocessing import image
import perReModel
import os
from werkzeug.utils import secure_filename
import base64

UPLOAD_FOLDER = os.path.basename('upload_image')

app = Flask(__name__)
app.config["DEBUG"] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/personRecognition")
def main():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    
    # get Model
    model = perReModel.getModel()
    
    # read the posted values from the UI
    file = request.files['img']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    pathImg = 'upload_image/'+filename
    #ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    with open(pathImg, "rb") as image_file:
        encoded_srcImg = base64.b64encode(image_file.read()).decode("ascii")
    
    # Making new predictions
    test_image = image.load_img(pathImg, target_size = (224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    
    if result[0][0] == 1:
        prediction = 'Person'
    else:
        prediction = 'NoPerson'
    
    return render_template('index.html', srcImg=encoded_srcImg, result=prediction)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
