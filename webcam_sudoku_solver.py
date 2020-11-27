from sudoku_solver import *

import cv2 as cv
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


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
		warp_sudoku_board_width = 3 * 9 * 28
		warp_sudoku_board_height = 3 * 9 * 28
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


def get_best_shift(img):
	cy, cx = ndimage.measurements.center_of_mass(img)
	rows, cols = img.shape
	shift_x = np.round(cols/2.0-cx).astype(int)
	shift_y = np.round(rows/2.0-cy).astype(int)
	return shift_x, shift_y


def shift(img, sx, sy):
	rows, cols = img.shape
	m = np.float32([[1, 0, sx], [0, 1, sy]])
	shifted = cv.warpAffine(img, m, (cols, rows))
	return shifted


class WebcamSudokuSolver:
	def __init__(self, model):
		self.model = model
		self.last_sudoku_solution = None

	def solve(self, frame):
		blur_threshold_warp_sudoku_board, variable2undo_warp = warp_sudoku_board(frame)

		if blur_threshold_warp_sudoku_board is None or blur_threshold_warp_sudoku_board[0] is None:
			return frame

		cv.imshow('gray_warp_sudoku_board', blur_threshold_warp_sudoku_board)
		cv.waitKey(0)

		board = blur_threshold_warp_sudoku_board

		board_height = board.shape[0]
		board_width = board.shape[1]

		temp = [False for x in range(9)]
		digit_exists = list()
		for x in range(9):
			digit_exists.append(temp.copy())

		cropped_squares = list()
		digits_count = 0

		# check for all squares if a square contain a digit and save cropped square if so
		for i in range(9):
			for j in range(9):
				x1 = j * board_width // 9
				x2 = (j + 1) * board_width // 9
				y1 = i * board_height // 9
				y2 = (i + 1) * board_height // 9
				square = board[y1:y2, x1:x2]

				# cv.imshow('square', square)
				# cv.resizeWindow('square', 200, 200)
				# cv.waitKey(0)

				delta_y = y2 - y1
				delta_x = x2 - x1

				strongly_cropped_square = square[3 * delta_y // 20:int(0.85 * delta_y), 3 * delta_x // 20:int(0.85 * delta_x)]

				# cv.imshow('strongly_cropped_square', strongly_cropped_square)
				# cv.resizeWindow('strongly_cropped_square', 200, 200)
				# cv.waitKey(0)

				contours, hierarchy = cv.findContours(strongly_cropped_square, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
				if not contours:
					# print('No contours')
					# cv.waitKey(0)
					continue
				biggest = max(contours, key=cv.contourArea)
				area = cv.contourArea(biggest)
				if area < 2 * delta_y:
					# print('Area to small')
					# cv.waitKey(0)
					continue
				x, y, w, h, = cv.boundingRect(biggest)
				if h < 0.75 * strongly_cropped_square.shape[0]:
					# cv.imshow('square', square)
					# cv.imshow('cropped', strongly_cropped_square)
					# print('To short:')
					# print('if h < 0.75 * delta_y: == True:')
					# print('if ' + str(h) + '< 0.75 * ' + str(delta_y) + ': == True:')
					# print(str(h) + '<' + str(0.75 * delta_y))
					# cv.waitKey(0)
					continue
				# now we are sure that there is a digit
				digit_exists[i][j] = True
				digits_count += 1
				cropped_square = square[delta_y // 10:int(0.95 * delta_y), delta_x // 10:int(0.95 * delta_x)]
				cropped_squares.append(cropped_square)

		# prepare data from cropped squares and put it into inputs array
		inputs = np.ones((digits_count, 28, 28, 1), dtype='float32')
		digit_number = 0
		for cropped_square in cropped_squares:
			contours, hierarchy = cv.findContours(cropped_square, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
			biggest = max(contours, key=cv.contourArea)
			x, y, w, h = cv.boundingRect(biggest)
			digit = cropped_square[y:y+h, x:x+w]
			if h > w:
				factor = 20.0 / h
				h = 20
				w = int(round(w * factor))
				digit = cv.resize(digit, (w, h), interpolation=cv.INTER_AREA)
			else:
				factor = 20.0 / w
				w = 20
				h = int(round(h * factor))
				digit = cv.resize(digit, (w, h), interpolation=cv.INTER_AREA)
			horizontal_margin = 28 - w
			vertical_margin = 28 - h
			left_margin = horizontal_margin // 2
			right_margin = horizontal_margin - left_margin
			top_margin = vertical_margin // 2
			bottom_margin = vertical_margin - top_margin
			digit = cv.copyMakeBorder(
				digit, top_margin, bottom_margin, left_margin, right_margin, borderType=cv.BORDER_CONSTANT
			)
			shift_x, shift_y = get_best_shift(digit)
			digit = shift(digit, shift_x, shift_y)
			inputs[digit_number] = digit.reshape(28, 28, 1)
			digit_number += 1

		inputs = inputs / 255

		# now it's time to check if inputs array contains correct data:
		# for i in range(digits_count):
		# 	print(i)
		# 	plt.imshow(inputs[i], cmap="gray")
		# 	plt.show()

		predictions = self.model.predict([inputs])

		for i in range(digits_count):
			print('This is probably', np.argmax(predictions[i]))
			plt.imshow(inputs[i], cmap="gray")
			plt.show()

		input('And those were all digits...')










		for i in range(9):
			for j in range(9):
				# square = blur_threshold_warp_sudoku_board[
				# 				offset_height + i * 9 * offset_height:(i+1) * 9 * offset_height,
				# 				offset_width + j * 9 * offset_width:(j+1) * 9 * offset_width]

				square = blur_threshold_warp_sudoku_board[
						 temp_height * i + temp_offset_height:temp_height * (i + 1) - temp_offset_height,
						 temp_width * j + temp_offset_width:temp_width * (j + 1) - temp_offset_width
						 ]

				# cv.imshow(str(j) + ':' + str(i) + 'square', square)
				# cv.waitKey(0)

				strongly_cropped_square = square[
										  int(1.5 * offset_height):int(6.5 * offset_height),
										  int(1.5 * offset_width):int(6.5 * offset_width)]

				# cv.imshow(str(j) + ':' + str(i) + 'strongly cropped', strongly_cropped_square)
				# cv.waitKey(0)

				contours, hierarchy = cv.findContours(strongly_cropped_square, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
				if len(contours) == 0:
					digits[i][j] = 0
					print(0, end=' ')
					# print(digits[i][j], end=' ')
					# print('ni ma kontur w strongly cropped')
					# cv.waitKey(0)
					# cv.destroyWindow(str(j) + ':' + str(i) + 'square')
					# cv.destroyWindow(str(j) + ':' + str(i) + 'strongly cropped')
					continue
				biggest = max(contours, key=cv.contourArea)
				x, y, w, h, = cv.boundingRect(biggest)

				if h < 2 * offset_height:
					digits[i][j] = 0
					print(0, end=' ')
					# print(digits[i][j], end=' ')
					# print('jest kontura, ale jest za mala')
					# cv.waitKey(0)
					# cv.destroyWindow(str(j) + ':' + str(i) + 'square')
					# cv.destroyWindow(str(j) + ':' + str(i) + 'strongly cropped')
					continue

				contours, hierarchy = cv.findContours(square, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
				biggest = max(contours, key=cv.contourArea)
				x, y, w, h, = cv.boundingRect(biggest)

				digit = square[y:y + h, x:x + w]

				print('D', end=' ')
			# print('Jest ladnie cyferka')
			# cv.imshow(str(j) + ':' + str(i) + 'digit', digit)
			# cv.waitKey(0)
			#
			# cv.destroyWindow(str(j) + ':' + str(i) + 'square')
			# cv.destroyWindow(str(j) + ':' + str(i) + 'strongly cropped')
			# cv.destroyWindow(str(j) + ':' + str(i) + 'digit')

			print()

		# input('For loops has ended, press any key to continue: ')

		print()
		for y in range(9):
			for x in range(9):
				if digits[y][x] is None:
					print('D ', end='')
				else:
					print(digits[y][x], end=' ')
			print()
		cv.waitKey(0)

		# squares = squares.reshape(squares.shape[0], 28, 28, 1)
		# squares = squares.astype("float32")
		# squares = squares / 255
		# predictions = self.model.predict([squares])
		# index = 0
		# for prediction in predictions:
		#
		# 	index += 1
		return frame
