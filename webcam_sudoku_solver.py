import sudoku_solver
import cv2 as cv
import numpy as np
from scipy import ndimage


def warp_sudoku_board(frame):
	gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	blur_gray_frame = cv.GaussianBlur(gray_frame, (7, 7), 0)
	blur_threshold_frame = cv.adaptiveThreshold(blur_gray_frame, 255, 1, 1, 11, 2)
	contours, hierarchy = cv.findContours(blur_threshold_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	biggest = np.array([])
	max_area = 0
	for i in contours:
		area = cv.contourArea(i)
		peri = cv.arcLength(i, True)
		approx = cv.approxPolyDP(i, 0.02 * peri, True)
		if area > max_area and len(approx) == 4:
			biggest = approx
			max_area = area
	blur_threshold_warp_sudoku_board = None
	variable2undo_warp = None
	if biggest.size > 0:
		biggest = biggest.reshape((4, 2))
		biggest_new = np.zeros((4, 1, 2), dtype=np.int32)
		add = biggest.sum(1)
		biggest_new[0] = biggest[np.argmin(add)]
		biggest_new[3] = biggest[np.argmax(add)]
		diff = np.diff(biggest, axis=1)
		biggest_new[1] = biggest[np.argmin(diff)]
		biggest_new[2] = biggest[np.argmax(diff)]
		biggest = biggest_new
		cv.drawContours(frame, biggest, -1, (0, 0, 255), 25)
		pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
		warp_sudoku_board_width = 2 * 9 * 28
		warp_sudoku_board_height = 2 * 9 * 28
		# PREPARE POINTS FOR WARP
		pts2 = np.float32(
			[
				[0, 0],
				[warp_sudoku_board_width, 0],
				[0, warp_sudoku_board_height],
				[warp_sudoku_board_width, warp_sudoku_board_height]
			]
		)
		matrix = cv.getPerspectiveTransform(pts1, pts2)  # GER
		variable2undo_warp = cv.getPerspectiveTransform(pts2, pts1)
		blur_threshold_warp_sudoku_board = cv.warpPerspective(
			blur_threshold_frame, matrix, (warp_sudoku_board_width, warp_sudoku_board_height)
		)
	return blur_threshold_warp_sudoku_board, variable2undo_warp


def split_boxes(blur_threshold_warp_sudoku_board):
	squares = list()
	rows = np.vsplit(blur_threshold_warp_sudoku_board, 9)
	for row in rows:
		cols = np.hsplit(row, 9)
		for square in cols:
			squares.append(square)
	return squares


class WebcamSudokuSolver:
	def __init__(self, model):
		self.model = model
		self.last_sudoku_solution = None

	def solve(self, frame):
		blur_threshold_warp_sudoku_board, variable2undo_warp = warp_sudoku_board(frame)
		# if blur_threshold_warp_sudoku_board is not None and blur_threshold_warp_sudoku_board[0] is not None:
		# 	if len(blur_threshold_warp_sudoku_board) > 0 and len(blur_threshold_warp_sudoku_board[0]):
		# 		cv.imshow('gray_warp_sudoku_board', blur_threshold_warp_sudoku_board)
		# else:
		# 	return frame
		if blur_threshold_warp_sudoku_board is None or blur_threshold_warp_sudoku_board[0] is None:
			return frame
		# if len(blur_threshold_warp_sudoku_board) > 0 and len(blur_threshold_warp_sudoku_board[0]):
		cv.imshow('gray_warp_sudoku_board', blur_threshold_warp_sudoku_board)
		cv.waitKey(0)

		squares = split_boxes(blur_threshold_warp_sudoku_board)
		digits = [-1 for x in range(81)]
		index = -1

		height = blur_threshold_warp_sudoku_board.shape[0] // 9
		width = blur_threshold_warp_sudoku_board.shape[1] // 9

		offset_width = width // 10
		offset_height = height // 10

		for square in squares:
			crop_image = warp[height * i + offset_height:height * (i + 1) - offset_height,
						 width * j + offset_width:width * (j + 1) - offset_width]



			cnts, _ = cv.findContours(square, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
			biggest = max(cnts, key=cv.contourArea)
			x, y, w, h, = cv.boundingRect(biggest)
			cv.imshow(str(index), square)
			square = cv.cvtColor(square, cv.COLOR_GRAY2BGR)
			cv.rectangle(square, (x, y), (x + w, y + h), (0, 255, 0), 3)
			cv.imshow(str(index) + '+rect', square)
			cv.waitKey(0)

			index += 1
			cropped = square[6:len(square)-6, 6:len(square)-6]
			contours, hierarchy = cv.findContours(cropped, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
			if len(contours) == 0:
				digits[index] = 0
				continue
			biggest = max(contours, key=cv.contourArea)
			if cv.contourArea(biggest) < 32:
				# Why exactly 32? Whenever I have to choose any number, I choose a power of two
				digits[index] = 0
				continue
			square = cv.bitwise_not(square)
			x, y, w, h = cv.boundingRect(biggest)
			# the biggest contour was founded in cropped image, not in original square
			x += 6
			y += 6
			# width and height of the bounding square don't change

			digit_rectangle = square
			cy, cx = ndimage.measurements.center_of_mass(square)
			rows, cols = square.shape
			shiftx = np.round(cols / 2.0 - cx).astype(int)
			shifty = np.round(rows / 2.0 - cy).astype(int)

			rows, cols = square.shape
			M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
			square = cv.warpAffine(square, M, (cols, rows))

			square = cv.bitwise_not(square)

			# nie wiem co jest na jakich bitwisach, ale na pewno trzeba cyfre do 20 zmniejszyc
			# a ostatecznie ma byc wycentrowane masowo 28x28
			# no ale pamietaj o tych kolorkach czy pracujesz na black-white czy na white-black
			# bo za ta petla squary traktujesz juz kolektywnie
			# gosciu ma zajebiscie chyba ogarniete to co wyzej libertarianskie i
			# a poza tym sprawdz z czego on nauczyl siec, a wg plik train jego
			# moze ma fajny model
			# jezeli nie bedzie dzialac cos to zawsze mozna sprobowac zrobic siec na fontach komputerowych
			# gosciu ma w repo plik trenujacy, wiec jak cos bedzie latwo
			# na koncu repo masz jak wytrenowac wow, teamwork i opensource, to sie szanuje

			square = cv.resize(square, (28, 28), interpolation=cv.INTER_AREA)

		cv.waitKey(0)

		squares = squares.reshape(squares.shape[0], 28, 28, 1)
		squares = squares.astype("float32")
		squares = squares / 255
		predictions = self.model.predict([squares])
		index = 0
		for prediction in predictions:

			index += 1
		return frame
