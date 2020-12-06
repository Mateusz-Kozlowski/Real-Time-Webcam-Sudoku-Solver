import sudoku_solver

import cv2 as cv
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


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


def shift(img, sx, sy):
	"""

	:param img:
	:param sx:
	:param sy:
	:return:
	"""
	rows, cols = img.shape
	m = np.float32([[1, 0, sx], [0, 1, sy]])
	shifted = cv.warpAffine(img, m, (cols, rows))
	return shifted


class WebcamSudokuSolver:
	def __init__(self, model):
		self.model = model
		self.last_sudoku_solution = None

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

		h, w = warp_sudoku_board.shape[:2]
		center = w / 2, h / 2

		current_attempt = 1
		solved = False

		while current_attempt <= 4 and not solved:
			if current_attempt == 1:
				rotated_sudoku_board = warp_sudoku_board
			else:
				rotation_matrix = cv.getRotationMatrix2D(center, 90 * (current_attempt - 1), 1.0)
				rotated_sudoku_board = cv.warpAffine(warp_sudoku_board, rotation_matrix, (w, h))

			rotated_sudoku_board = remove_rotation_artifacts(rotated_sudoku_board)

			# print('Before bug')
			# print(rotated_sudoku_board)
			# print(rotated_sudoku_board.shape[0])
			# print(rotated_sudoku_board.shape[1])

			if rotated_sudoku_board.shape[0] == 0 or rotated_sudoku_board.shape[1] == 0:
				# print('after deleting artifacts width or height is 0, so return frame & end function at all')
				# cv.waitKey(0)
				return frame

			cv.imshow('current_attempt=' + str(current_attempt), rotated_sudoku_board)

			squares = get_squares(rotated_sudoku_board)

			is_there_a_digit_in_the_square = np.zeros((9, 9), dtype=bool)

			for y in range(9):
				for x in range(9):
					is_there_a_digit_in_the_square[y, x] = contains_a_digit(squares[y][x])

			for y in range(9):
				for x in range(9):
					print(int(is_there_a_digit_in_the_square[y, x]), end=' ')
				print()
			print()

			cv.waitKey(0)

			cv.destroyWindow('current_attempt=' + str(current_attempt))

			###

			solved = True

			# inputs = prepare_inputs_for_model(strongly_cropped_squares_with_digits)

			current_attempt += 1

		return frame

		# digits = get_digits(warp_image)










		# find board (4 points from biggest quadrangle contour)
		# get warped board from 4 points
		# divide into 81 squares (requires binary warped board)
		# try 4 times with 4 rotations everything what left
		# if a try is successful draw unwarped digits and return (requires matrix)

		# blur_threshold_warp_sudoku_board, variable2undo_warp = find_sudoku_board(frame)

		# cv.imshow('gray_warp_sudoku_board', blur_threshold_warp_sudoku_board)
		# cv.waitKey(0)

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

		if len(inputs) == 0:
			return frame
		predictions = self.model.predict([inputs])

		# for i in range(digits_count):
		# 	print('This is probably', np.argmax(predictions[i]))
		# 	plt.imshow(inputs[i], cmap="gray")
		# 	plt.show()
		# input('And those were all digits...')

		temp = [0 for x in range(9)]
		new_sudoku = [temp.copy() for x in range(9)]

		i = 0
		for y in range(9):
			for x in range(9):
				if digit_exists[y][x]:
					new_sudoku[y][x] = np.argmax(predictions[i])
					i += 1

		# for row in new_sudoku:
		# 	for square in row:
		# 		print(square, end=' ')
		# 	print()

		# cv.waitKey(0)

		for y in range(9):
			for x in range(9):
				print(new_sudoku[y][x], end=' ')
			print()

		cv.waitKey(0)

		# now it's time to check if solution of new sudoku can be last sudoku solution
		solve_sudoku_again = False
		if self.last_sudoku_solution is None:
			solve_sudoku_again = True
		else:
			for y in range(9):
				for x in range(9):
					if new_sudoku[y][x] != 0:
						if new_sudoku[y][x] is not self.last_sudoku_solution[y][x]:
							solve_sudoku_again = True

		if solve_sudoku_again:
			sudoku_solver.solve_sudoku(new_sudoku)  # Solve it

		# cv.waitKey(0)
		# draw unwarp

		# for i in range(9):
		# 	for j in range(9):
		# 		# square = blur_threshold_warp_sudoku_board[
		# 		# 				offset_height + i * 9 * offset_height:(i+1) * 9 * offset_height,
		# 		# 				offset_width + j * 9 * offset_width:(j+1) * 9 * offset_width]
		#
		# 		square = blur_threshold_warp_sudoku_board[
		# 				 temp_height * i + temp_offset_height:temp_height * (i + 1) - temp_offset_height,
		# 				 temp_width * j + temp_offset_width:temp_width * (j + 1) - temp_offset_width
		# 				 ]
		#
		# 		# cv.imshow(str(j) + ':' + str(i) + 'square', square)
		# 		# cv.waitKey(0)
		#
		# 		strongly_cropped_square = square[
		# 								  int(1.5 * offset_height):int(6.5 * offset_height),
		# 								  int(1.5 * offset_width):int(6.5 * offset_width)]
		#
		# 		# cv.imshow(str(j) + ':' + str(i) + 'strongly cropped', strongly_cropped_square)
		# 		# cv.waitKey(0)
		#
		# 		contours, hierarchy = cv.findContours(strongly_cropped_square, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
		# 		if len(contours) == 0:
		# 			digits[i][j] = 0
		# 			print(0, end=' ')
		# 			# print(digits[i][j], end=' ')
		# 			# print('ni ma kontur w strongly cropped')
		# 			# cv.waitKey(0)
		# 			# cv.destroyWindow(str(j) + ':' + str(i) + 'square')
		# 			# cv.destroyWindow(str(j) + ':' + str(i) + 'strongly cropped')
		# 			continue
		# 		biggest = max(contours, key=cv.contourArea)
		# 		x, y, w, h, = cv.boundingRect(biggest)
		#
		# 		if h < 2 * offset_height:
		# 			digits[i][j] = 0
		# 			print(0, end=' ')
		# 			# print(digits[i][j], end=' ')
		# 			# print('jest kontura, ale jest za mala')
		# 			# cv.waitKey(0)
		# 			# cv.destroyWindow(str(j) + ':' + str(i) + 'square')
		# 			# cv.destroyWindow(str(j) + ':' + str(i) + 'strongly cropped')
		# 			continue
		#
		# 		contours, hierarchy = cv.findContours(square, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
		# 		biggest = max(contours, key=cv.contourArea)
		# 		x, y, w, h, = cv.boundingRect(biggest)
		#
		# 		digit = square[y:y + h, x:x + w]
		#
		# 		print('D', end=' ')
		# 	# print('Jest ladnie cyferka')
		# 	# cv.imshow(str(j) + ':' + str(i) + 'digit', digit)
		# 	# cv.waitKey(0)
		# 	#
		# 	# cv.destroyWindow(str(j) + ':' + str(i) + 'square')
		# 	# cv.destroyWindow(str(j) + ':' + str(i) + 'strongly cropped')
		# 	# cv.destroyWindow(str(j) + ':' + str(i) + 'digit')
		#
		# 	print()
		#
		# # input('For loops has ended, press any key to continue: ')
		#
		# print()
		# for y in range(9):
		# 	for x in range(9):
		# 		if digits[y][x] is None:
		# 			print('D ', end='')
		# 		else:
		# 			print(digits[y][x], end=' ')
		# 	print()
		# cv.waitKey(0)
		#
		# # squares = squares.reshape(squares.shape[0], 28, 28, 1)
		# # squares = squares.astype("float32")
		# # squares = squares / 255
		# # predictions = self.model.predict([squares])
		# # index = 0
		# # for prediction in predictions:
		# #
		# # 	index += 1
		return frame


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


def get_biggest_quadrangle(frame, draw_vertices_on_frame=True):
	"""

	:param frame:
	:param draw_vertices_on_frame:
	:return:
	"""
	if frame is None:
		return None, None

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


def remove_rotation_artifacts(warp_sudoku_board):
	while len(warp_sudoku_board) > 0 and np.sum(warp_sudoku_board[0]) == 0:
		warp_sudoku_board = warp_sudoku_board[1:]

	while len(warp_sudoku_board) > 0 and np.sum(warp_sudoku_board[:, 0]) == 0:
		warp_sudoku_board = np.delete(warp_sudoku_board, 0, 1)

	while len(warp_sudoku_board) > 0 and np.sum(warp_sudoku_board[-1]) == 0:
		warp_sudoku_board = warp_sudoku_board[:-1]

	while len(warp_sudoku_board) > 0 and np.sum(warp_sudoku_board[:, -1]) == 0:
		warp_sudoku_board = np.delete(warp_sudoku_board, -1, 1)

	return warp_sudoku_board


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


def contains_a_digit(square):
	"""

	:param square:
	:return:
	"""
	delta_y = square.shape[0]
	delta_x = square.shape[1]

	strongly_cropped_square = square[3 * delta_y // 20:int(0.85 * delta_y), 3 * delta_x // 20:int(0.85 * delta_x)]

	contours, hierarchy = cv.findContours(strongly_cropped_square, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	if not contours:
		return False

	biggest = max(contours, key=cv.contourArea)
	area = cv.contourArea(biggest)
	if area < 2 * delta_y:
		return False

	x, y, w, h, = cv.boundingRect(biggest)
	if h < 0.75 * strongly_cropped_square.shape[0]:
		return False

	return True


# def get_digits(warp_image):
# 	"""
#
# 	:param warp_image:
# 	:return:
# 	"""
# 	board_height = warp_image.shape[0]
# 	board_width = warp_image.shape[1]
#
# 	digits = np.zeros((9, 9), dtype=bool)
#
# 	for y in range(9):
# 		for x in range(9):
# 			pass
#
# 	# input = self.
#
# 	cv.waitKey(0)
#
# 	return None
