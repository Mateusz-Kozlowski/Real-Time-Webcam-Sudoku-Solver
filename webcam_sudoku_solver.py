# The core of the program.
# Contains WebcamSudokuSolver class which contains solve function.
# The function takes as an argument an image from webcam and returns it with some modifications.
# On the returned image there are marked 4 vertices of the biggest quadrangle.
# If the quadrangle is a sudoku board then the function tries to solve it and if the process is successful then
# the solution is drawn on the returned image.

import sudoku_solver

from copy import deepcopy

import numpy as np
import cv2 as cv
from scipy import ndimage


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
			Otherwise an error may occur or the function won't find any board.
		:return:
			A copy of a frame with some modifications - vertices of the biggest quadrangle and
			solution if any is found.
		"""
		if frame is None:
			return frame

		frame = deepcopy(frame)

		warp_sudoku_board, warp_matrix = get_biggest_quadrangle(frame)

		if warp_sudoku_board is None:
			return frame

		boxes = get_boxes(warp_sudoku_board)

		digits_occurrence = check_digits_occurrence(boxes)

		inputs = prepare_inputs(boxes, digits_occurrence)
		if inputs is None:
			return frame

		# try to solve a sudoku in every rotation (0, 90, 180 and 270 degrees)
		current_attempt = 1
		while current_attempt <= 4:
			rotation_angle = self.last_solved_sudoku_rotation + 90 * (current_attempt - 1)

			rotated_inputs = rotate_inputs(inputs, rotation_angle)

			predictions = self.model.predict([rotated_inputs])

			if not probabilities_are_good(predictions):
				current_attempt += 1
				continue

			digits_grid = get_digits_grid(predictions, digits_occurrence, rotation_angle)

			if self.new_sudoku_solution_may_be_last_solution(digits_grid):
				self.last_solved_sudoku_rotation = rotation_angle

				result = inverse_warp_digits_on_frame(
					digits_grid, self.last_sudoku_solution, frame, warp_sudoku_board.shape, warp_matrix, rotation_angle
				)

				return result

			solved_digits_grid = sudoku_solver.solve_sudoku(digits_grid)
			if solved_digits_grid is None:
				current_attempt += 1
				continue

			self.last_sudoku_solution = solved_digits_grid
			self.last_solved_sudoku_rotation = rotation_angle

			result = inverse_warp_digits_on_frame(
				digits_grid, solved_digits_grid, frame, warp_sudoku_board.shape, warp_matrix, rotation_angle
			)

			return result

		return frame

	def new_sudoku_solution_may_be_last_solution(self, digits_grid):
		"""
		:param digits_grid:
			2D numpy array which contains a sudoku puzzle; if a field is empty then it should contain 0
		:return:
			True or False
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
		OpenCV image (3D numpy array (rows, columns, color channels)).
		It has to be either an BGR or a grayscale image.
		Otherwise an error may occur or the function won't find the biggest quadrangle.
		The argument may be modified depending on the value of second argument.
	:param draw_vertices_on_frame:
		Allows to mark vertices of the biggest quadrangle as red circles/dots.
	:return:
		warp_matrix which will allow you to "unwarp" cropped sudoku board and
		warp_sudoku_board which is thresholded, gray, cropped and warped sudoku board;
		the function may return None, None if there is no external contours or
		if there is no quadrangle with positive size or
		if the size of the board is too small (width or height is smaller than 9 * 28 pixels)
	"""
	# if the frame contains 3 color channels then it's converted into grayscale
	if len(frame.shape) == 3:
		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	else:
		gray = frame

	blur_gray_frame = cv.GaussianBlur(gray, (7, 7), 0)
	threshold_frame = cv.adaptiveThreshold(blur_gray_frame, 255, 1, 1, 11, 2)

	contours, _ = cv.findContours(threshold_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

	if len(contours) == 0:
		return None, None

	# coordinates of vertices of the biggest quadrangle
	vertices = np.array([])

	# find the biggest contour
	max_area = 0
	for contour in contours:
		area = cv.contourArea(contour)
		perimeter = cv.arcLength(contour, True)
		approx_epsilon = 0.02
		approx = cv.approxPolyDP(contour, approx_epsilon * perimeter, True)
		if area > max_area and len(approx) == 4:
			vertices = approx
			max_area = area

	# there is no quadrangle with positive size
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

	# the size of the board is too small
	if warp_sudoku_board.shape[0] < 28 * 9 or warp_sudoku_board.shape[1] < 28 * 9:
		return None, None

	return warp_sudoku_board, warp_matrix


def reorder_quadrangle_vertices(vertices):
	"""
	:param vertices:
		A 3D numpy array which contains a coordinates of a quadrangle, it should look like this:
		[ [[x1, y1]], [[x2, y2]], [[x3, y3]], [[x4, y4]] ].
		Of course there don't have to be sorted in any order, because it the task of the function to reorder them.
	:return:
		Reordered vertices, they'll look like this:
		D---C
		|   |
		A---B
		[ [[Dx, Dy]], [[Cx, Cy]], [[Bx, By]], [[Ax, Ay]] ].
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
		A 3D numpy array which contains a coordinates of a quadrangle, it should look like this:
		D---C
		|   |
		A---B
		[ [[Dx, Dy]], [[Cx, Cy]], [[Bx, By]], [[Ax, Ay]] ].
	:return:
		width, height (which are integers)
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


def get_boxes(warp_sudoku_board):
	"""
	Splits image into 81 small boxes.

	:param warp_sudoku_board:
		OpenCV image
	:return:
		9x9 2D list; each cell contains 2D numpy array
	"""
	temp = [None for i in range(9)]
	boxes = [temp.copy() for i in range(9)]

	board_height = warp_sudoku_board.shape[0]
	board_width = warp_sudoku_board.shape[1]

	for y in range(9):
		for x in range(9):
			x1 = x * board_width // 9
			x2 = (x + 1) * board_width // 9
			y1 = y * board_height // 9
			y2 = (y + 1) * board_height // 9
			boxes[y][x] = warp_sudoku_board[y1:y2, x1:x2]

	return boxes


def check_digits_occurrence(boxes):
	"""
	:param boxes:
		2D list of 81 gray OpenCV images (2D numpy arrays)
	:return:
		2D numpy array that contains True or False values that represent occurrence of digits
	"""
	digits_occurrence = np.zeros((9, 9), dtype=bool)

	for y in range(9):
		for x in range(9):
			height = boxes[y][x].shape[0]
			width = boxes[y][x].shape[1]

			strongly_cropped_box = boxes[y][x][3 * height // 20:int(0.85 * height), 3 * width // 20:int(0.85 * width)]

			contours, hierarchy = cv.findContours(strongly_cropped_box, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
			if not contours:
				continue

			biggest = max(contours, key=cv.contourArea)
			area = cv.contourArea(biggest)
			if area < 2 * height:
				continue

			x_, y_, w_, h_, = cv.boundingRect(biggest)
			if h_ < 0.75 * strongly_cropped_box.shape[0] and w_ < 0.75 * strongly_cropped_box.shape[1]:
				continue

			digits_occurrence[y, x] = True

	return digits_occurrence


def prepare_inputs(boxes, digits_occurrence):
	"""
	:param boxes:
		2D list of 81 gray OpenCV images (2D numpy arrays)
	:param digits_occurrence:
		2D numpy array that contains True or False values that represent occurrence of digits
	:return:
		if no digit was found returns None;
		otherwise returns 4D numpy array with shape = (digits count, 28, 28, 1) that
		contains cropped, scaled and centered digits that are perfectly prepared for a cnn model
		(at least for this model I created)
	"""
	digits_count = 0
	for y in digits_occurrence:
		for x in y:
			digits_count += int(x)

	if digits_count == 0:
		return None

	cropped_boxes_with_digits = get_cropped_boxes_with_digits(boxes, digits_occurrence)

	digits = get_cropped_digits(cropped_boxes_with_digits)

	if digits is None:
		return None

	resize(digits)

	digits = add_margins(digits, 28, 28)

	center_using_mass_centers(digits)

	digits = digits.reshape((digits.shape[0], 28, 28, 1))

	digits = digits / 255

	return digits


def get_cropped_boxes_with_digits(boxes, digits_occurrence):
	"""
	Prepares boxes that contains digits to find the biggest EXTERNAL contours by removing white lines from sudoku grid.

	:param boxes:
		2D list of 81 gray OpenCV images (2D numpy arrays)
	:param digits_occurrence:
		2D numpy array that contains True or False values that represent occurrence of digits
	:return:
		list of 2D numpy arrays that are cropped boxes that contains digits
	"""
	cropped_boxes_with_digits = list()

	for y in range(9):
		for x in range(9):
			if digits_occurrence[y, x]:
				height = boxes[y][x].shape[0]
				width = boxes[y][x].shape[1]

				cropped_boxes_with_digits.append(
					boxes[y][x][
						int(0.05 * height):int(0.95 * height),
						int(0.05 * width):int(0.95 * width)
					]
				)

				# there may be some artifacts (remains of a sudoku grid) that are easy to remove using while loops
				# (grid and digits are white, background is black)

				binary = deepcopy(cropped_boxes_with_digits[-1])
				_, binary = cv.threshold(binary, 0, 255, cv.THRESH_BINARY)

				while len(binary) > 0 and len(binary[0]) > 0 and np.sum(binary[0]) >= int(0.9 * 255 * len(binary[0])):
					binary = binary[1:]
					cropped_boxes_with_digits[-1] = cropped_boxes_with_digits[-1][1:]

				while len(binary) > 0 and len(binary[0]) > 0 and np.sum(binary[:, 0]) >= int(0.9 * 255 * len(binary[:, 0])):
					binary = np.delete(binary, 0, 1)
					cropped_boxes_with_digits[-1] = np.delete(cropped_boxes_with_digits[-1], 0, 1)

				while len(binary) > 0 and len(binary[0]) > 0 and np.sum(binary[-1]) >= int(0.9 * 255 * len(binary[-1])):
					binary = binary[:-1]
					cropped_boxes_with_digits[-1] = cropped_boxes_with_digits[-1][:-1]

				while len(binary) > 0 and len(binary[0]) > 0 and np.sum(binary[:, -1]) >= int(0.9 * 255 * len(binary[:, -1])):
					binary = np.delete(binary, -1, 1)
					cropped_boxes_with_digits[-1] = np.delete(cropped_boxes_with_digits[-1], -1, 1)

	return cropped_boxes_with_digits


def get_cropped_digits(cropped_boxes_with_digits, remove_noise=True):
	"""
	Crops digits to their bounding rectangles. Also can remove noise.

	:param cropped_boxes_with_digits:
		list of 2D numpy arrays that are cropped boxes that contain digits
	:param remove_noise:
		bool variable that allows to remove noises around digits
	:return:
		list of 2D numpy arrays that contain perfectly cropped digits (to their bounding rectangles)
	"""
	digits = list()
	for i in cropped_boxes_with_digits:
		contours, _ = cv.findContours(i, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
		if len(contours) == 0:
			return None
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
	Normalizes digits to fit them in a 20x20 pixel boxes while preserving their aspect ratio.

	:param digits:
		list of 2D numpy arrays that contain perfectly cropped digits (to their bounding rectangles);
		original list will be modified
	:return:
		None (original list will be modified)
	"""
	for index, digit in enumerate(digits):
		h = digit.shape[0]  # height
		w = digit.shape[1]  # width

		if h > w:
			factor = 20.0 / h
			h = 20
			w = int(round(w * factor))
		else:
			factor = 20.0 / w
			w = 20
			h = int(round(h * factor))

		digits[index] = cv.resize(digit, (w, h), interpolation=cv.INTER_AREA)


def add_margins(digits, new_width, new_height):
	"""
	:param digits:
		list of 2D numpy arrays that contain perfectly cropped digits (to their bounding rectangles)
	:param new_width:
		total new width
	:param new_height:
		total new height
	:return:
		3D numpy array with shape = (digits count, new width, new height) and dtype='float32'
	"""
	digits_array = np.zeros((len(digits), new_width, new_height), dtype='float32')

	for i, digit in enumerate(digits):
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

	return digits_array


def center_using_mass_centers(digits):
	"""
	:param digits:
		list of 2D numpy arrays that contain perfectly cropped digits (to their bounding rectangles);
		original list will be modified
	:return:
		None (original list will be modified)
	"""
	for i, digit in enumerate(digits):
		shift_x, shift_y = get_best_shift(digit)
		rows, cols = digit.shape
		m = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
		digits[i] = cv.warpAffine(digit, m, (cols, rows))


def get_best_shift(img):
	"""
	:param img:
		2D numpy array
	:return:
		shift_x, shift_y that are integers
	"""
	cy, cx = ndimage.measurements.center_of_mass(img)
	rows, cols = img.shape
	shift_x = np.round(cols/2.0-cx).astype(int)
	shift_y = np.round(rows/2.0-cy).astype(int)
	return shift_x, shift_y


def rotate_inputs(inputs, rotation_angle):
	"""
	:param inputs:
		Perfectly prepared inputs for a cnn model (at least for this model I created)
	:param rotation_angle:
		90 * k, k e Z;
		inputs will be rotated clockwise
	:return:
		rotated inputs copies
	"""
	rotation_angle = rotation_angle % 360

	if rotation_angle == 0:
		return deepcopy(inputs)

	rotated_inputs = np.zeros((inputs.shape[0], 28, 28))

	if rotation_angle == 90:
		for i, single_input in enumerate(inputs):
			rotated_inputs[i] = cv.rotate(single_input, cv.ROTATE_90_CLOCKWISE)
	elif rotation_angle == 180:
		for i, single_input in enumerate(inputs):
			rotated_inputs[i] = cv.rotate(single_input, cv.ROTATE_180)
	elif rotation_angle == 270:
		for i, single_input in enumerate(inputs):
			rotated_inputs[i] = cv.rotate(single_input, cv.ROTATE_90_COUNTERCLOCKWISE)

	return rotated_inputs.reshape((inputs.shape[0], 28, 28, 1))


def probabilities_are_good(predictions):
	"""
	Returns False if average probability < 90%, otherwise True.

	:param predictions:
		a variable returned by keras models
	:return:
		True or False
	"""
	average = 0
	for prediction in predictions:
		average += prediction[np.argmax(prediction)]
	average = average / len(predictions)
	if average < 0.9:
		return False
	return True


def get_digits_grid(predictions, digits_occurrence, rotation_angle):
	"""
	:param predictions:
		a variable returned by keras models
	:param digits_occurrence:
		2D numpy array that contains True or False values that represent occurrence of digits
	:param rotation_angle:
		90 * k, k e Z;
		inputs are rotated clockwise
	:return:
		2D numpy array with shape = (9, 9) with dtype=np.uint8 that contains "vertically normalized" digits grid -
		even if a sudoku in the real life is rotated by 90, 180 or 270 degrees - digits grid won't be rotated;
		in other words:
		no matter how a sudoku is rotated, the function will always return a normalized grid;
		marks empty boxes as 0
	"""
	digits_grid = np.zeros((9, 9), np.uint8)

	rotation_angle = rotation_angle % 360

	i = 0
	for y in range(9):
		for x in range(9):
			if digits_occurrence[y, x]:
				if predictions[i][np.argmax(predictions[i])] > 0.5:
					digits_grid[y, x] = np.argmax(predictions[i])
				else:
					print('A digit is strange, its probability =', predictions[i][np.argmax(predictions[i])])
					digits_grid[y, x] = 0
				i += 1

	if rotation_angle != 0:
		digits_grid = np.rot90(digits_grid, (360 - rotation_angle) / 90)

	return digits_grid


def inverse_warp_digits_on_frame(digits_grid, solution_digits_grid, frame, warp_dimensions, warp_matrix, rotation_angle):
	"""
	:param digits_grid:
		2D numpy array with "vertically normalized" content, requires empty boxes marked as 0
	:param solution_digits_grid:
		2D numpy array with "vertically normalized" content
	:param frame:
		results will be drawn on the copy of frame
	:param warp_dimensions:
		height and width of warped sudoku board
	:param warp_matrix:
		an argument that was used to extract warped board from frame
	:param rotation_angle:
		90 * k, k e Z;
		inputs are rotated clockwise
	:return:
		result - a copy of a frame with a drawn solution
	"""
	# green digits form solution drawn on a black background
	only_digits = get_only_digits_img(digits_grid, solution_digits_grid, warp_dimensions, rotation_angle)

	# "unwarped" digits
	inverted_warped_only_digits = cv.warpPerspective(
		only_digits, warp_matrix, (frame.shape[1], frame.shape[0]), flags=cv.WARP_INVERSE_MAP
	)

	# merge with frame
	result = np.where(inverted_warped_only_digits.sum(axis=-1, keepdims=True) == 200, inverted_warped_only_digits, frame)

	return result


def get_only_digits_img(digits_grid, solution_digits_grid, warp_dimensions, rotation_angle):
	"""
	:param digits_grid:
		2D numpy array with "vertically normalized" content, requires empty boxes marked as 0
	:param solution_digits_grid:
		2D numpy array with "vertically normalized" content
	:param warp_dimensions:
		height and width of warped sudoku board
	:param rotation_angle:
		90 * k, k e Z;
		inputs are rotated clockwise
	:return:
		green digits from solution on a black background
	"""

	blank = np.zeros((warp_dimensions[0], warp_dimensions[1], 3), dtype='uint8')

	rotation_angle = rotation_angle % 360
	digits_grid = np.rot90(digits_grid, rotation_angle / 90)
	solution_digits_grid = np.rot90(solution_digits_grid, rotation_angle / 90)

	box_height, box_width = warp_dimensions[0] // 9, warp_dimensions[1] // 9
	dimension = min(box_width, box_height)

	digits = np.zeros((9, 9, dimension, dimension, 3), dtype='uint8')

	font = cv.FONT_HERSHEY_DUPLEX

	for y in range(9):
		for x in range(9):
			if digits_grid[y, x] != 0:
				continue
			text = str(solution_digits_grid[y, x])

			scale = dimension / 41

			(text_height, text_width), _ = cv.getTextSize(text, font, fontScale=scale, thickness=3)

			bottom_left_x = box_width // 2 - text_width // 2
			bottom_left_y = box_height // 2 + text_height // 2

			digits[y, x] = cv.putText(
				digits[y, x], text, (bottom_left_x, bottom_left_y), font, scale, (0, 200, 0),
				thickness=3, lineType=cv.LINE_AA
			)

			start_y = y * box_height
			start_x = x * box_width

			# rotate each digit individually
			if rotation_angle == 0:
				temp = digits[y, x]
				blank[start_y:start_y + dimension, start_x:start_x + dimension] = temp
			elif rotation_angle == 90:
				temp = cv.rotate(digits[y, x], cv.ROTATE_90_COUNTERCLOCKWISE)
				blank[start_y:start_y + dimension, start_x:start_x + dimension] = temp
			elif rotation_angle == 180:
				temp = cv.rotate(digits[y, x], cv.ROTATE_180)
				blank[start_y:start_y + dimension, start_x:start_x + dimension] = temp
			elif rotation_angle == 270:
				temp = cv.rotate(digits[y, x], cv.ROTATE_90_CLOCKWISE)
				blank[start_y:start_y + dimension, start_x:start_x + dimension] = temp

	return blank
