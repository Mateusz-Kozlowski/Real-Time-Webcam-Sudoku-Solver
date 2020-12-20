print('Importing code and libraries from other files...')

from webcam_sudoku_solver import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def main():
	# TODO remove all comments or clean code at all
	# TODO dfs all code and ADD interesting print-logs
	# TODO also add some comments
	# TODO add "log" to print statements which are logs
	webcam_width, webcam_height = 1920, 1080
	webcam = cv.VideoCapture(0)
	webcam.set(cv.CAP_PROP_FRAME_WIDTH, webcam_width)
	webcam.set(cv.CAP_PROP_FRAME_HEIGHT, webcam_height)

	model = tf.keras.models.load_model('Models/cnn model.h5', custom_objects=None, compile=True)

	webcam_sudoku_solver = WebcamSudokuSolver(model)

	while webcam.isOpened():
		successful_frame_read, frame = webcam.read()

		if not successful_frame_read:
			break

		output_frame = webcam_sudoku_solver.solve(frame)

		cv.imshow('Webcam Sudoku Solver', output_frame)

		if cv.waitKey(1) >= 0:
			break

	cv.destroyAllWindows()
	webcam.release()


if __name__ == "__main__":
	main()

print('Code is done, so everything works fine!')
