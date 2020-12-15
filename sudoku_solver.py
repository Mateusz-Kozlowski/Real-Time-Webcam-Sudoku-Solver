import numpy as np


def solve_sudoku(digits_grid):
	"""
	:param digits_grid:
	:return:
	"""
	if not is_solvable(digits_grid):
		print('A sudoku is unsolvable')
		return False

	human_notes = get_full_human_notes(digits_grid)

	while True:
		sth_has_changed1 = remove_orphans_technique(digits_grid, human_notes)

		sth_has_changed2 = single_appearances_technique(digits_grid, human_notes)

		if not sth_has_changed1 and not sth_has_changed2:
			break

	return is_solved_correctly(digits_grid)


def is_solvable(digits_grid):
	"""
	:param digits_grid:
	:return:
	"""
	for y in range(9):
		for x in range(9):
			if digits_grid[y, x]:
				if not check_row(x, y, digits_grid):
					return False
				if not check_col(x, y, digits_grid):
					return False
				if not check_square(x, y, digits_grid):
					return False
	return True


def check_row(x, y, digits_grid):
	"""
	:param x:
	:param y:
	:param digits_grid:
	:return:
	"""
	for i in range(9):
		if i != x and digits_grid[y, i] == digits_grid[y, x]:
			return False
	return True


def check_col(x, y, digits_grid):
	"""
	:param x:
	:param y:
	:param digits_grid:
	:return:
	"""
	for i in range(9):
		if i != y and digits_grid[i, x] == digits_grid[y, x]:
			return False
	return True


def check_square(x, y, digits_grid):
	"""
	:param x:
	:param y:
	:param digits_grid:
	:return:
	"""
	x_square = x // 3
	y_square = y // 3
	for i in range(3):
		for j in range(3):
			if 3 * y_square + i != y or 3 * x_square + j != x:
				if digits_grid[3 * y_square + i, 3 * x_square + j] == digits_grid[y, x]:
					return False
	return True


def get_full_human_notes(digits_grid):
	"""
	:param digits_grid:
	:return:
	"""
	full_human_notes = np.zeros((9, 9), dtype=set)
	for y in range(9):
		for x in range(9):
			if digits_grid[y, x] == 0:
				full_human_notes[y, x] = find_all_candidates(digits_grid, x, y)
			else:
				full_human_notes[y, x] = set()
	return full_human_notes


def find_all_candidates(digits_grid, x, y):
	"""
	:param digits_grid:
	:param x:
	:param y:
	:return:
	"""
	dg = digits_grid
	candidates = set()
	for i in range(1, 10):
		if fits_in_a_row(dg, y, i) and fits_in_a_col(dg, x, i):
			x_square = x // 3
			y_square = y // 3
			if fits_in_a_square(dg, x_square, y_square, i):
				candidates.add(i)
	return candidates


def fits_in_a_row(digits_grid, y, digit):
	"""
	:param digits_grid:
	:param y:
	:param digit:
	:return:
	"""
	for i in range(9):
		if digits_grid[y, i] == digit:
			return False
	return True


def fits_in_a_col(digits_grid, x, digit):
	"""
	:param digits_grid:
	:param x:
	:param digit:
	:return:
	"""
	for i in range(9):
		if digits_grid[i, x] == digit:
			return False
	return True


def fits_in_a_square(digits_grid, x_square, y_square, digit):
	"""
	:param digits_grid:
	:param x_square:
	:param y_square:
	:param digit:
	:return:
	"""
	for i in range(3):
		for j in range(3):
			if digits_grid[3 * y_square + i, 3 * x_square + j] == digit:
				return False
	return True


def remove_orphans_technique(digits_grid, human_notes):
	"""
	:param digits_grid:
	:param human_notes:
	:return:
	"""

	sth_has_changed = False
	for y in range(9):
		for x in range(9):
			if len(human_notes[y, x]) == 1:
				sth_has_changed = True

				# get first element of a set
				digit = 0  # it isn't necessarily but removes a warning from PyCharm
				for digit in human_notes[y, x]:
					break

				digits_grid[y, x] = digit
				human_notes[y, x] = set()

				implications_of_removing_an_orphan(human_notes, x, y, digit)
	return sth_has_changed


def implications_of_removing_an_orphan(candidates, x, y, digit):
	"""
	:param candidates:
	:param x:
	:param y:
	:param digit:
	:return:
	"""
	for i in range(9):
		candidates[y, i].discard(digit)

	for i in range(9):
		candidates[i, x].discard(digit)

	x_square = x // 3
	y_square = y // 3
	for i in range(3):
		for j in range(3):
			candidates[y_square * 3 + i, x_square * 3 + j].discard(digit)


def single_appearances_technique(digits_grid, human_notes):
	"""
	:param digits_grid:
	:param human_notes:
	:return:
	"""
	sth_has_changed = False

	for y in range(9):
		for digit in range(1, 10):
			appearances = 0
			appearance_index = -1
			for x in range(9):
				if digit in human_notes[y, x]:
					appearances += 1
					if appearances == 2:
						break
					appearance_index = x
			if appearances == 1:
				sth_has_changed = True
				digits_grid[y, appearance_index] = digit
				human_notes[y, appearance_index] = set()
				implications_of_removing_an_orphan(human_notes, appearance_index, y, digit)

	for x in range(9):
		for digit in range(1, 10):
			appearances = 0
			appearance_index = -1
			for y in range(9):
				if digit in human_notes[y, x]:
					appearances += 1
					if appearances == 2:
						break
					appearance_index = y
			if appearances == 1:
				sth_has_changed = True
				digits_grid[appearance_index, x] = digit
				human_notes[appearance_index, x] = set()
				implications_of_removing_an_orphan(human_notes, x, appearance_index, digit)

	for i in range(3):
		for j in range(3):
			for digit in range(1, 10):
				appearances = 0
				appearance_x_index = -1
				appearance_y_index = -1
				for y in range(3):
					for x in range(3):
						if digit in human_notes[i * 3 + y, j * 3 + x]:
							appearances += 1
							if appearances == 2:
								break
							appearance_x_index = j * 3 + x
							appearance_y_index = i * 3 + y
				if appearances == 1:
					sth_has_changed = True
					digits_grid[appearance_y_index, appearance_x_index] = digit
					human_notes[appearance_y_index, appearance_x_index] = set()
					implications_of_removing_an_orphan(human_notes, appearance_x_index, appearance_y_index, digit)

	return sth_has_changed


def is_solved_correctly(digits_grid):
	"""
	:param digits_grid:
	:return:
	"""
	for y in digits_grid:
		for x in y:
			if not x:
				return False

	return is_solvable(digits_grid)


# sudoku1 = [
# 	[4, 6, 5, 3, 7, 8, 9, 1, 2],
# 	[8, 1, 9, 5, 6, 2, 4, 7, 3],
# 	[3, 7, 2, 4, 9, 1, 8, 5, 6],
# 	[7, 9, 3, 8, 2, 6, 5, 0, 1],
# 	[6, 2, 1, 7, 5, 4, 3, 9, 0],
# 	[5, 8, 4, 1, 3, 9, 6, 2, 7],
# 	[1, 5, 8, 6, 4, 7, 2, 3, 9],
# 	[9, 4, 7, 2, 8, 3, 1, 6, 5],
# 	[2, 3, 6, 9, 1, 5, 7, 0, 0]
# ]
#
# sudoku2 = [
# 	[0, 1, 0, 6, 0, 4, 3, 0, 7],
# 	[3, 5, 6, 0, 0, 0, 0, 0, 0],
# 	[0, 0, 0, 0, 5, 3, 6, 9, 0],
# 	[0, 8, 3, 2, 6, 0, 4, 0, 9],
# 	[0, 0, 0, 0, 0, 0, 0, 0, 0],
# 	[4, 0, 5, 0, 7, 8, 2, 6, 0],
# 	[0, 4, 2, 5, 3, 0, 0, 0, 0],
# 	[0, 0, 0, 0, 0, 0, 7, 2, 4],
# 	[7, 0, 9, 4, 0, 2, 0, 8, 0]
# ]
#
# sudoku3 = [
# 	[4, 0, 0, 0, 0, 0, 0, 1, 0],
# 	[0, 0, 0, 0, 0, 2, 0, 0, 3],
# 	[0, 0, 0, 4, 0, 0, 0, 0, 0],
# 	[0, 0, 0, 0, 0, 0, 5, 0, 0],
# 	[6, 0, 1, 7, 0, 0, 0, 0, 0],
# 	[0, 0, 4, 1, 0, 0, 0, 0, 0],
# 	[0, 5, 0, 0, 0, 0, 2, 0, 0],
# 	[0, 0, 0, 0, 8, 0, 0, 6, 0],
# 	[0, 3, 0, 9, 1, 0, 0, 0, 0]
# ]
#
# sudoku0 = np.array(sudoku4)
#
# if not solve_sudoku(sudoku0):
# 	print('Not solved :(')
# print(sudoku0)
