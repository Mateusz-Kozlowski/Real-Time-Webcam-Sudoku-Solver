# This file provides an algorithm to solve very quickly every sudoku.
# But a puzzle have to have only one solution.
# If a puzzle has more than one correct solution then it won't be solved at all.
# The algorithm is in solve_sudoku function.

from copy import deepcopy

import numpy as np


def solve_sudoku(digits_grid):
	"""
	:param digits_grid:
		2D numpy array that contains digits. 0 <==> there is no digit, blank box
	:return:
		2D numpy array - solved sudoku or None if a puzzle isn't solvable or has more than one solution
	"""
	if not is_solvable(digits_grid):
		return None

	digits_grid = deepcopy(digits_grid)

	human_notes = get_full_human_notes(digits_grid)

	while True:
		sth_has_changed1 = remove_orphans_technique(digits_grid, human_notes)

		sth_has_changed2 = single_appearances_technique(digits_grid, human_notes)

		if not sth_has_changed1 and not sth_has_changed2:
			break

	if is_solved_correctly(digits_grid):
		return digits_grid
	return None


def is_solvable(digits_grid):
	"""
	:param digits_grid:
		2D numpy array that contains digits. 0 <==> there is no digit, blank box
	:return:
		True or False
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
	Checks if a digit in a box with coordinates y, x fits to its row.
	Useful for checking if sudoku is solvable at all.

	:param x:
		a coordinate counted from 0
	:param y:
		a coordinate counted from 0
	:param digits_grid:
		2D numpy array that contains digits. 0 <==> there is no digit, blank box
	:return:
		True or False
	"""
	for i in range(9):
		if i != x and digits_grid[y, i] == digits_grid[y, x]:
			return False
	return True


def check_col(x, y, digits_grid):
	"""
	Checks if a digit in a box with coordinates y, x fits to its column.
	Useful for checking if sudoku is solvable at all.

	:param x:
		a coordinate counted from 0
	:param y:
		a coordinate counted from 0
	:param digits_grid:
		2D numpy array that contains digits. 0 <==> there is no digit, blank box
	:return:
		True or False
	"""
	for i in range(9):
		if i != y and digits_grid[i, x] == digits_grid[y, x]:
			return False
	return True


def check_square(x, y, digits_grid):
	"""
	Checks if a digit in a box with coordinates y, x fits to its big box.
	Useful for checking if sudoku is solvable at all.

	:param x:
		a coordinate counted from 0
	:param y:
		a coordinate counted from 0
	:param digits_grid:
		2D numpy array that contains digits. 0 <==> there is no digit, blank box
	:return:
		True or False
	"""
	x_big_box = x // 3
	y_big_box = y // 3
	for i in range(3):
		for j in range(3):
			if 3 * y_big_box + i != y or 3 * x_big_box + j != x:
				if digits_grid[3 * y_big_box + i, 3 * x_big_box + j] == digits_grid[y, x]:
					return False
	return True


def get_full_human_notes(digits_grid):
	"""
	:param digits_grid:
		2D numpy array that contains digits. 0 <==> there is no digit, blank box
	:return:
		2D numpy array of Python sets; each set contains digits that can match the box <==>
		if a digits for sure doesn't match a box then it isn't in a set related with that box
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
		2D numpy array that contains digits. 0 <==> there is no digit, blank box
	:param x:
		a coordinate counted from 0
	:param y:
		a coordinate counted from 0
	:return:
		Python set that contains digits that can match the box <==>
		if a digits for sure doesn't match a box then it isn't in a set related with that box
	"""
	dg = digits_grid
	candidates = set()
	for i in range(1, 10):
		if fits_in_row(dg, y, i) and fits_in_col(dg, x, i):
			x_square = x // 3
			y_square = y // 3
			if fits_in_a_square(dg, x_square, y_square, i):
				candidates.add(i)
	return candidates


def fits_in_row(digits_grid, y, digit):
	"""
	:param digits_grid:
		2D numpy array that contains digits. 0 <==> there is no digit, blank box
	:param y:
		a coordinate counted from 0
	:param digit:
		a number in range <1; 9>
	:return:
		True or False
	"""
	for i in range(9):
		if digits_grid[y, i] == digit:
			return False
	return True


def fits_in_col(digits_grid, x, digit):
	"""
	:param digits_grid:
		2D numpy array that contains digits. 0 <==> there is no digit, blank box
	:param x:
		a coordinate counted from 0
	:param digit:
		a number in range <1; 9>
	:return:
		True or False
	"""
	for i in range(9):
		if digits_grid[i, x] == digit:
			return False
	return True


def fits_in_a_square(digits_grid, x_square, y_square, digit):
	"""
	:param digits_grid:
		2D numpy array that contains digits. 0 <==> there is no digit, blank box
	:param x_square:
		a coordinate of big 3x3 box in range <0; 2>
	:param y_square:
		a coordinate of big 3x3 box in range <0; 2>
	:param digit:
		a number in range <1; 9>
	:return:
		True or False
	"""
	for i in range(3):
		for j in range(3):
			if digits_grid[3 * y_square + i, 3 * x_square + j] == digit:
				return False
	return True


def remove_orphans_technique(digits_grid, human_notes):
	"""
	Finds boxes with only one digit, types that digits into them and do implications.

	:param digits_grid:
		2D numpy array that contains digits. 0 <==> there is no digit, blank box;
		this function will change this argument
	:param human_notes:
		2D numpy array of Python sets; each set contains digits that can match the box <==>
		if a digits for sure doesn't match a box then it isn't in a set related with that box
		this function will change this argument
	:return:
		True or False depending on whether something has changed
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
		a set of digits that represents human notes from a box
	:param x:
		a coordinate counted from 0
	:param y:
		a coordinate counted from 0
	:param digit:
		a number in range <1; 9>
	:return:
		None (original set will be modified)
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
	Finds digits that appear only once in a row/col/big 3x3 box of human notes,
	types them into a box and do implications.

	:param digits_grid:
		2D numpy array that contains digits. 0 <==> there is no digit, blank box;
		this function will change this argument
	:param human_notes:
		2D numpy array of Python sets; each set contains digits that can match the box <==>
		if a digits for sure doesn't match a box then it isn't in a set related with that box
		this function will change this argument
	:return:
		True or False depending on whether something has changed
	"""
	sth_has_changed = False

	# rows
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

	# columns
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

	# 3x3 boxes:
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
	Checks if all sudoku is filled with digits greater than 0 and if is solved correctly.

	:param digits_grid:
		2D numpy array that contains digits. 0 <==> there is no digit, blank box;
		this function will change this argument
	:return:
		True or False
	"""
	for y in digits_grid:
		for x in y:
			if not x:
				return False

	return is_solvable(digits_grid)
