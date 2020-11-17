'''
https://www.tensorflow.org/tutorials/quickstart/beginner
'''
import tensorflow as tf

import numpy as np
import cv2

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10),
    ])

predictions = model(x_train[:1]).numpy()

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
    loss=loss_fn,
    metrics=['accuracy'],
    )

model.fit(x_train, y_train, epochs=1)

# model.evaluate(x_test,  y_test, verbose=2)

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

    if event == cv2.EVENT_LBUTTONDOWN:        
        draw = True
        print('drawing')

    elif event == cv2.EVENT_LBUTTONUP:
        draw = False
        print('stop drawing')
        # print(mat)

    elif event == cv2.EVENT_MOUSEMOVE:
        if draw:
            # print(x,y)
            mat[y][x] = 255

    elif event == cv2.EVENT_LBUTTONDBLCLK:
        mat = np.zeros((28,28), dtype=np.uint8)
        print("Mat cleared")
    

cv2.namedWindow('Drawboard')
cv2.setMouseCallback("Drawboard", mouse_cb)

while True:
    cv2.imshow('Drawboard', mat)

    new_mat = np.reshape(mat,(1,784))
    print(np.argmax(probability_model(new_mat)))
    
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
