from flask import Flask
from flask import request
from Backend.processing import imageProcess as img



app = Flask(__name__,static_folder="static")
port = 9999



# Route for root
@app.route("/")
def index():
    return app.send_static_file("index.html")


# Route for image dropper
# Should get an image in the request.
# Send that image to imageProcess
@app.route("/imageSend",methods=["POST"])
def image_send():
    print("fired")
    return "Yes"



if __name__ == "__main__":
    app.run(debug=True, port=port)