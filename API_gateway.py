from flask import Flask, request, Response
from werkzeug.utils import secure_filename
from Backend.processing import imageProcess as img
from os import path


app = Flask(__name__,static_folder="static")
port = 9999

app.config["UPLOAD_FOLDER"] = "uploads"



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
    f = request.files["file"]
    filename = secure_filename(f.filename)
    the_path = path.join(app.config['UPLOAD_FOLDER'])
    # f.save(the_path, filename)
    img.imageprocess("uploads/"+filename)
    return "Ok"


if __name__ == "__main__":
    app.run(debug=True, port=port)