from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from Backend.processing import imageProcess as img
from os import path, remove

app = Flask(__name__,static_folder="static")
port = 9999

app.config["UPLOAD_FOLDER"] = "uploads"



# Route for root
@app.route("/")
def index():
    return app.send_static_file("index.html")


# Route for image dropper
@app.route("/imageSend",methods=["POST"])
def image_send():
    print("fired")
    file = request.files["file"]
    filename = secure_filename(file.filename)
    file.save(path.join(app.config['UPLOAD_FOLDER'], filename))
    pred_to_front,conf = img.predict_image("uploads/{}".format(filename))
    remove("uploads/{}".format(filename))
    return jsonify(prediction = pred_to_front,confidence = "{:.2f}".format(conf) )


if __name__ == "__main__":
    app.run(debug=True, port=port)
