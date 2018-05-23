# FVK
import tensorflow as tf

from flask import Flask
application = Flask(__name__)

@application.route("/")
def hello():
    return "Hello World from " + tf.__version__ 

if __name__ == "__main__":
    application.run()
