from flask import Flask
from flask import render_template
import os

app = Flask(__name__,static_folder="static")
port = 9999

@app.route("/")
def index():
    return app.send_static_file("index.html")




if __name__ == "__main__":
    app.run(debug=True, port=port)