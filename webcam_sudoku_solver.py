import sudoku_solver

from copy import deepcopy

import numpy as np
import cv2 as cv
from scipy import ndimage
import matplotlib.pyplot as plt


class WebcamSudokuSolver:
	def __init__(self, model):
		self.model = model
		self.last_sudoku_solution = None
		self.last_solved_sudoku_rotation = 0

	def solve(self, frame):
		"""
		:param frame:
			OpenCV image (3D numpy array (rows, columns, color channels)).
			It has to be either an BGR or a grayscale image.
			Otherwise an error may occur or the function won't find a board.

		:return:
		"""
		warp_matrix, warp_sudoku_board = get_biggest_quadrangle(frame)

		if warp_sudoku_board is None:
			return frame

		squares = get_squares(warp_sudoku_board)

		digits_occurrence = check_digits_occurrence(squares)

		inputs = prepare_inputs(squares, digits_occurrence)

		current_attempt = 1

		while current_attempt <= 4:
			rotation_angle = self.last_solved_sudoku_rotation + 90 * (current_attempt - 1)
			rotated_inputs = rotate_inputs(inputs, rotation_angle)

			predictions = self.model.predict([rotated_inputs])

			print('Predictions:')
			for i in predictions:
				print(np.argmax(i), end=' ')

			input('Dziala? Taktyczne X D jak cos...')

			###

			if not probabilities_are_good(predictions):  # check only average, separately maybe in the future (in digits = get_digits_grid(predictions, digits_occurrence))
				current_attempt += 1
				continue

			# digits is correctly rotated grid, does not matter how rotated is sheet of paper, always look in this way:

			'''
			4 0 0 0 0 0 0 1 0
			0 0 0 0 0 2 0 0 3
			0 0 0 4 0 0 0 0 0
			0 0 0 0 0 0 5 0 0
			6 0 1 7 0 0 0 0 0
			0 0 4 1 0 0 0 0 0
			0 5 0 0 0 0 2 0 0
			0 0 0 0 8 0 0 6 0
			0 3 0 9 1 0 0 0 0
			'''

			# so this is still the same sudoku, even if it was rotated in during last frame
			# the only one problem may be unwarping, it's kinda confusing

			digits_grid = get_digits_grid(predictions, digits_occurrence, rotation_angle)

			if not is_solvable(digits_grid):
				current_attempt += 1
				continue

			if self.new_sudoku_solution_may_be_last_solution(digits_grid):
				unwarp_digits_on_frame(self.last_sudoku_solution, frame, warp_matrix, rotation_angle)
				return frame

			solution_digits_grid = sudoku_solver.solve_sudoku(digits_grid)
			if solution_digits_grid is None:
				current_attempt += 1
				continue

			self.last_sudoku_solution = solution_digits_grid

			unwarp_digits_on_frame(solution_digits_grid, frame, warp_matrix, rotation_angle)
			return frame

		return frame

	def new_sudoku_solution_may_be_last_solution(self, digits_grid):
		"""
		:param digits_grid:
		:return:
		"""
		if self.last_sudoku_solution is None:
			return False

		for y in range(9):
			for x in range(9):
				if digits_grid[y, x] != 0:
					if digits_grid[y, x] != self.last_sudoku_solution[y, x]:
						return False
		return True


def get_biggest_quadrangle(frame, draw_vertices_on_frame=True):
	"""
	:param frame:
	:param draw_vertices_on_frame:
	:return:
	"""
	# if the frame contains 3 color channels then it's converted into grayscale
	if len(frame.shape) == 3:
		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	else:
		gray = frame

	blur_gray_frame = cv.GaussianBlur(gray, (7, 7), 0)
	threshold_frame = cv.adaptiveThreshold(blur_gray_frame, 255, 1, 1, 11, 2)

	contours, hierarchy = cv.findContours(threshold_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

	if len(contours) == 0:
		return None, None

	vertices = np.array([])  # coordinates of vertices of the biggest quadrangle
	max_area = 0

	for contour in contours:
		area = cv.contourArea(contour)
		perimeter = cv.arcLength(contour, True)
		approx_epsilon = 0.02
		approx = cv.approxPolyDP(contour, approx_epsilon * perimeter, True)
		if area > max_area and len(approx) == 4:
			vertices = approx
			max_area = area

	if vertices.size == 0:
		return None, None

	vertices = reorder_quadrangle_vertices(vertices)
	warp_width, warp_height = get_quadrangle_dimensions(vertices)

	if draw_vertices_on_frame:
		cv.drawContours(frame, vertices, -1, (0, 0, 255), warp_width // 32)

	pts1 = np.float32(vertices)
	pts2 = np.float32(
		[
			[0, 0],
			[warp_width, 0],
			[0, warp_height],
			[warp_width, warp_height]
		]
	)

	warp_matrix = cv.getPerspectiveTransform(pts1, pts2)
	warp_sudoku_board = cv.warpPerspective(threshold_frame, warp_matrix, (warp_width, warp_height))

	return warp_matrix, warp_sudoku_board


def reorder_quadrangle_vertices(vertices):
	"""
	:param vertices:
	:return:
	"""
	vertices = vertices.reshape((4, 2))
	reordered_vertices = np.zeros((4, 1, 2), dtype=np.int32)
	add = vertices.sum(1)
	reordered_vertices[0] = vertices[np.argmin(add)]
	reordered_vertices[3] = vertices[np.argmax(add)]
	diff = np.diff(vertices, axis=1)
	reordered_vertices[1] = vertices[np.argmin(diff)]
	reordered_vertices[2] = vertices[np.argmax(diff)]
	return reordered_vertices


def get_quadrangle_dimensions(vertices):
	"""
	:param vertices:
	:return:
	"""
	temp = np.zeros((4, 2), dtype=int)
	for i in range(4):
		temp[i] = vertices[i, 0]

	delta_x = temp[0, 0]-temp[1, 0]
	delta_y = temp[0, 1]-temp[1, 1]
	width1 = int((delta_x**2 + delta_y**2)**0.5)

	delta_x = temp[1, 0] - temp[2, 0]
	delta_y = temp[1, 1] - temp[2, 1]
	width2 = int((delta_x**2 + delta_y**2)**0.5)

	delta_x = temp[2, 0] - temp[3, 0]
	delta_y = temp[2, 1] - temp[3, 1]
	height1 = int((delta_x**2 + delta_y**2)**0.5)

	delta_x = temp[3, 0] - temp[0, 0]
	delta_y = temp[3, 1] - temp[0, 1]
	height2 = int((delta_x**2 + delta_y**2)**0.5)

	width = max(width1, width2)
	height = max(height1, height2)

	return width, height


def get_squares(warp_sudoku_board):
	"""
	:param warp_sudoku_board:
	:return:
	"""
	temp = [None for i in range(9)]
	squares = [temp.copy() for i in range(9)]

	board_height = warp_sudoku_board.shape[0]
	board_width = warp_sudoku_board.shape[1]

	for y in range(9):
		for x in range(9):
			x1 = x * board_width // 9
			x2 = (x + 1) * board_width // 9
			y1 = y * board_height // 9
			y2 = (y + 1) * board_height // 9
			squares[y][x] = warp_sudoku_board[y1:y2, x1:x2]

	return squares


def check_digits_occurrence(squares):
	"""
	:param squares:
	:return:
	"""
	digits_occurrence = np.zeros((9, 9), dtype=bool)

	for y in range(9):
		for x in range(9):
			square = squares[y][x]

			height = square.shape[0]
			width = square.shape[1]

			strongly_cropped_square = square[3 * height // 20:int(0.85 * height), 3 * width // 20:int(0.85 * width)]

			contours, hierarchy = cv.findContours(strongly_cropped_square, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
			if not contours:
				continue

			biggest = max(contours, key=cv.contourArea)
			area = cv.contourArea(biggest)
			if area < 2 * height:
				continue

			x_, y_, w_, h_, = cv.boundingRect(biggest)
			if h_ < 0.75 * strongly_cropped_square.shape[0] and w_ < 0.75 * strongly_cropped_square.shape[1]:
				continue

			digits_occurrence[y, x] = True

	print('Print from check occurrence function')
	for y in range(9):
		for x in range(9):
			print(int(digits_occurrence[y, x]), end=' ')
		print()
	print()

	return digits_occurrence


def prepare_inputs(squares, digits_occurrence):
	"""
	:param squares:
	:param digits_occurrence:
	:return:
	"""
	digits_count = 0
	for y in digits_occurrence:
		for x in y:
			digits_count += int(x)

	print('digits_count = ', digits_count)
	print()

	cropped_squares_with_digits = get_cropped_squares_with_digits(squares, digits_occurrence)

	digits = get_cropped_digits(cropped_squares_with_digits)

	resize(digits)

	digits = add_margins(digits, 28, 28)

	center_using_mass_centers(digits)

	digits = digits.reshape(digits.shape[0], 28, 28, 1)

	digits = digits / 255

	# print('Take a look how inputs look (from prepare inputs function):')
	#
	# for i in digits:
	# 	plt.imshow(i, cmap='gray')
	# 	plt.show()
	#
	# input('Now it is time to go back from prepare inputs function')

	return digits


def get_cropped_squares_with_digits(squares, digits_occurrence):
	"""
	Prepares squares that contains digits to find the biggest EXTERNAL contours by removing white lines from sudoku grid
	:param squares:
	:param digits_occurrence:
	:return:
	"""
	cropped_squares_with_digits = list()

	for y in range(9):
		for x in range(9):
			if digits_occurrence[y, x]:
				height = squares[y][x].shape[0]
				width = squares[y][x].shape[1]

				cropped_squares_with_digits.append(
					squares[y][x][
						int(0.05 * height):int(0.95 * height),
						int(0.05 * width):int(0.95 * width)
					]
				)

				binary = deepcopy(cropped_squares_with_digits[-1])
				_, binary = cv.threshold(binary, 0, 255, cv.THRESH_BINARY)

				while np.sum(binary[0]) >= int(0.9 * 255 * len(binary[0])):
					binary = binary[1:]
					cropped_squares_with_digits[-1] = cropped_squares_with_digits[-1][1:]

				while np.sum(binary[:, 0]) >= int(0.9 * 255 * len(binary[:, 0])):
					binary = np.delete(binary, 0, 1)
					cropped_squares_with_digits[-1] = np.delete(cropped_squares_with_digits[-1], 0, 1)

				while np.sum(binary[-1]) >= int(0.9 * 255 * len(binary[-1])):
					binary = binary[:-1]
					cropped_squares_with_digits[-1] = cropped_squares_with_digits[-1][:-1]

				while np.sum(binary[:, -1]) >= int(0.9 * 255 * len(binary[:, -1])):
					binary = np.delete(binary, -1, 1)
					cropped_squares_with_digits[-1] = np.delete(cropped_squares_with_digits[-1], -1, 1)

	return cropped_squares_with_digits


def get_cropped_digits(cropped_squares_with_digits, remove_noise=True):
	"""
	Crops digits to their bounding rectangles. Also can remove noise.
	:param cropped_squares_with_digits:
	:param remove_noise:
	:return:
	"""
	digits = list()
	for i in cropped_squares_with_digits:
		contours, _ = cv.findContours(i, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
		biggest = max(contours, key=cv.contourArea)
		digit = deepcopy(i)
		if remove_noise:
			mask = np.zeros(i.shape, np.uint8)
			cv.drawContours(mask, [biggest], -1, (255, 255, 255), -1)
			digit = cv.bitwise_and(i, mask)
		x, y, w, h = cv.boundingRect(biggest)
		digit = digit[y:y+h, x:x+w]
		digits.append(digit)
	return digits


def resize(digits):
	"""
	:param digits:
	:return:
	"""
	for index in range(len(digits)):
		h = digits[index].shape[0]  # height
		w = digits[index].shape[1]  # width

		if h > w:
			factor = 20.0 / h
			h = 20
			w = int(round(w * factor))
		else:
			factor = 20.0 / w
			w = 20
			h = int(round(h * factor))

		digits[index] = cv.resize(digits[index], (w, h), interpolation=cv.INTER_AREA)


def add_margins(digits, new_width, new_height):
	"""
	:param digits:
	:param new_width:
	:param new_height:
	:return:
	"""
	digits_array = np.zeros((len(digits), new_width, new_height), dtype='float32')

	i = 0
	for digit in digits:
		h = digit.shape[0]  # height
		w = digit.shape[1]  # width

		horizontal_margin = new_width - w
		vertical_margin = new_height - h

		left_margin = horizontal_margin // 2
		right_margin = horizontal_margin - left_margin

		top_margin = vertical_margin // 2
		bottom_margin = vertical_margin - top_margin

		digits_array[i] = cv.copyMakeBorder(
			digit, top_margin, bottom_margin, left_margin, right_margin, borderType=cv.BORDER_CONSTANT
		)

		i += 1

	return digits_array


def center_using_mass_centers(digits):
	"""
	:param digits:
	:return:
	"""
	for i in range(len(digits)):
		shift_x, shift_y = get_best_shift(digits[i])
		rows, cols = digits[i].shape
		m = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
		digits[i] = cv.warpAffine(digits[i], m, (cols, rows))


def get_best_shift(img):
	"""
	:param img:
	:return:
	"""
	cy, cx = ndimage.measurements.center_of_mass(img)
	rows, cols = img.shape
	shift_x = np.round(cols/2.0-cx).astype(int)
	shift_y = np.round(rows/2.0-cy).astype(int)
	return shift_x, shift_y


def rotate_inputs(inputs, rotation_angle):
	"""
	:param inputs:
	:param rotation_angle:
	:return:
	"""
	return inputs


def probabilities_are_good(predictions):
	"""
	:param predictions:
	:return:
	"""
	return True


def get_digits_grid(predictions, digits_occurrence, rotation_angle):
	"""
	:param predictions:
	:param digits_occurrence:
	:param rotation_angle:
	:return:
	"""
	return digits_occurrence


def is_solvable(digits_grid):
	"""
	:param digits_grid:
	:return:
	"""
	return True


def unwarp_digits_on_frame(solution_digits_grid, frame, warp_matrix, rotation_angle):
	"""
	:param solution_digits_grid:
	:param frame:
	:param warp_matrix:
	:param rotation_angle:
	:return:
	"""
	return frame
