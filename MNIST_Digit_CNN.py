'''
https://www.tensorflow.org/tutorials/quickstart/beginner
'''
import tensorflow as tf

import numpy as np
import cv2

# Added to fix CUDA 11.2 & Tensorflow 2.4.0
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# ---------

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_rows, img_cols = 28, 28

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('X_train shape:', x_train.shape) #X_train shape: (60000, 28, 28, 1)

##model building
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28, 28, 1)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

predictions = model(x_train[:1]).numpy()

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
    loss=loss_fn,
    metrics=['accuracy'],
    )

model.fit(x_train, y_train, epochs=5)

model.save('mnist_conv')

probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
    ])

#--------------------------------------------------------------#

draw = False
mat = np.zeros((28,28), dtype=np.uint8)

def mouse_cb(event, x, y, flags, param):
    """Mouse callback function to modify the mat"""
    global draw
    global mat

    if x > 27 or y > 27 or x < 0 or y < 0:
        return

    if event == cv2.EVENT_LBUTTONDOWN:        
        draw = True
        print('drawing')

    elif event == cv2.EVENT_LBUTTONUP:
        draw = False
        print('stop drawing')

    elif event == cv2.EVENT_MOUSEMOVE:
        if draw:
            mat[y-1][x-1] = 128
            mat[y-1][x+1] = 128
            mat[y+1][x-1] = 128
            mat[y+1][x+1] = 128
            mat[y][x] = 255

    elif event == cv2.EVENT_LBUTTONDBLCLK:
        mat = np.zeros((28,28), dtype=np.uint8)
        print("Mat cleared")
    

cv2.namedWindow('Drawboard')
cv2.setMouseCallback("Drawboard", mouse_cb)

while True:
    cv2.imshow('Drawboard', mat)

    # new_mat = np.reshape(mat,(1,784))
    new_mat = np.reshape(mat,(1, 28, 28, 1))
    print(np.argmax(probability_model(new_mat)))
    
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
