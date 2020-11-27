print('Due TensorFlow library it may take a few second before it starts...')

from webcam_sudoku_solver import *

# import cv2 as cv
# import numpy as np
# import tensorflow as tf
# import keras
# from keras.models import load_model
# import matplotlib.pyplot as plt

WEBCAM_WIDTH, WEBCAM_HEIGHT = 1920, 1080


def main():
	webcam = cv.VideoCapture(0)
	webcam.set(cv.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
	webcam.set(cv.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)

	# model = load_model('cnn model.h5')
	model = None

	webcam_sudoku_solver = WebcamSudokuSolver(model)

	while webcam.isOpened():
		successful_frame_read, frame = webcam.read()

		if not successful_frame_read:
			break

		output_frame = webcam_sudoku_solver.solve(frame)

		cv.imshow('Webcam Sudoku Solver', output_frame)
		key = cv.waitKey(1)
		if key == 27:
			break

	cv.destroyAllWindows()
	webcam.release()


if __name__ == "__main__":
	main()

print('Code is done, so everything works fine!')
