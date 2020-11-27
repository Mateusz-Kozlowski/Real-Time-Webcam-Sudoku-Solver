from webcam_sudoku_solver import *

import tensorflow as tf


def main():
	webcam_width, webcam_height = 1920, 1080
	webcam = cv.VideoCapture(0)
	webcam.set(cv.CAP_PROP_FRAME_WIDTH, webcam_width)
	webcam.set(cv.CAP_PROP_FRAME_HEIGHT, webcam_height)

	model = tf.keras.models.load_model('cnn model.h5', custom_objects=None, compile=True)

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
