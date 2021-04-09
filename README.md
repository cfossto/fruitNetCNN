# LeScapeNet

LeScapeNetCNN is an app that uses a vanilla JS/HTML/CSS frontend and a Python backend built in Flask to classify images with an advanced CNN. The CNN is used with a costumized Tensorflow-model that is built from the ground up by the team.


## Techniques used:

* JS/HTML/CSS
* AJAX
* Python
* Tensorflow/Keras
* Flask

## Team members
Emil Lagerstedt, Filip Aldenhov, Kevin Andersson, Christopher Fossto



# How to use the app

1. Download this repository
2. Navigate to the project folder in a terminal
3. Run: python3 API_gateway
4. Open a browser and write localhost:9999
5. Done. Have fun!

(Ask any of us if you need the .h5 model. It is not available on Github, since it's too large.
Drop the .h5-file into the "Backend"-folder. The program should find it there.
If not, specify your own path in the imageProcess.py-file. Under the load_file-method,
you will find the variable "model_path". Drop the path-name there.)
