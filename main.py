from webcam_sudoku_solver import *

import tensorflow as tf


def main():
	webcam_width, webcam_height = 1920, 1080
	webcam = cv.VideoCapture(0)
	webcam.set(cv.CAP_PROP_FRAME_WIDTH, webcam_width)
	webcam.set(cv.CAP_PROP_FRAME_HEIGHT, webcam_height)

	model = tf.keras.models.load_model('Models/cnn model.h5', custom_objects=None, compile=True)

	webcam_sudoku_solver = WebcamSudokuSolver(model)

	while webcam.isOpened():
		# successful_frame_read, frame = webcam.read()

		# if not successful_frame_read:
		# 	break

		frame = cv.imread('Fake webcam/1.jpg')

		output_frame = webcam_sudoku_solver.solve(frame)

		cv.imshow('Webcam Sudoku Solver', output_frame)
		if cv.waitKey(1):
			break

	cv.destroyAllWindows()
	webcam.release()


if __name__ == "__main__":
	main()

print('Code is done, so everything works fine!')
