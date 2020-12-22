from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import random

(x_train, y_train), (x_test, y_test) = mnist.load_data()

plt.imshow(x_test[0])

input_img_row = x_train[0].shape[0]
input_img_cols = x_train[0].shape[1]

x_test = x_test.reshape(x_test.shape[0], input_img_row, input_img_cols, 1)

x_test = x_test.astype("float32")

x_test = x_test / 255

# one hot encoder of the labels
y_test = np_utils.to_categorical(y_test)

model_file_path = input('Enter a file path (with .h5 extension) which contains a saved model:')
loaded_model = load_model(model_file_path)

predictions = loaded_model.predict([x_test])

# Check the effectiveness of the model:
while True:
	index = random.randint(0, 10000)

	# Print out the number
	print('This is probably', np.argmax(predictions[index]))

	# Import the image
	plt.imshow(x_test[index], cmap="gray")

	# Show the image:
	plt.show()
