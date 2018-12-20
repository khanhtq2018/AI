# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 18:35:38 2018
@author: KhanhTQ7
"""

from flask import Flask, render_template, json, request
app = Flask(__name__)
app.config["DEBUG"] = False

@app.route("/personRecognition")
def main():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    
    # read the posted values from the UI
    #test_image = request.form['img']
    test_image = json.loads(request.data)
    print(test_image)
 
    # validate the received values
    if test_image:
        return 'khanh'
    else:
        return json.dumps({'html':'<span>All fields good !!</span>'})

if __name__ == "__main__":
    app.run()