# mnist_tensorflow_ocv_demo
## Description
This is a demo of recognizing handwritten digits using Tensorflow.  
You will train your own model and apply it interactively with a simple GUI.  
Refer to https://www.tensorflow.org/tutorials/quickstart/beginner for more information.

## Preparing VENV
> Make sure you have python3-venv installed
```
python3 -m venv env
source env/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

## Training & Using The Model
To perform the demo without CNN

> python3 MNIST_Digit.py

To perform the demo with CNN

> MNIST_Digit_CNN.py

## Using the model
When training is finished, a new window will pop up and you can draw with a mouse

Result will be written to stdout

![alt text](https://github.com/hyfung/mnist_tensorflow_ocv_demo/blob/white/images/01.png "")

## TODO
* ~~Save the model~~
* Let the script to read a saved model instead of training everytime
* ~~Also upload the CNN version~~
