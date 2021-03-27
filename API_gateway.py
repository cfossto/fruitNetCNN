from flask import Flask
from Backend.processing import imageProcess as img



app = Flask(__name__,static_folder="static")
port = 9999

@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/imageSend",methods=["POST"])
def image_send():
    hello = img.greeting()
    product = img.process(4)
    return str(product) + hello
        




if __name__ == "__main__":
    app.run(debug=True, port=port)