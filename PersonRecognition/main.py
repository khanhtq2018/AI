# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 18:35:38 2018

@author: KhanhTQ7
"""

from flask import Flask, render_template, json, request
app = Flask(__name__)

@app.route("/personRecognition")
def main():
    return render_template('index.html')
if __name__ == "__main__":
    app.run()

@app.route('/classify', methods=['POST'])
def classify():
    # read the posted values from the UI
    test_image = request.form['img']
 
    # validate the received values
    if test_image:
        return json.dumps({'html':'<span>All fields good !!</span>'})
    else:
        return json.dumps({'html':'<span>Enter the required fields</span>'})