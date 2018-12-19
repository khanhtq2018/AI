# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 18:35:38 2018

@author: KhanhTQ7
"""

from flask import Flask, render_template
app = Flask(__name__)

@app.route("/")
def main():
    return render_template('index.html')

if __name__ == "__main__":
    app.run()