-> [(COMING SOON!) LINK TO A VIDEO SHOWING HOW THE PROGRAM EXACTLY WORKS]() <-

# Real-Time-Webcam-Sudoku-Solver

## Table of contents
* [What is Real Time Webcam Sudoku Solver?](#What-is-Real-Time-Webcam-Sudoku-Solver?)
* [Code requirements](#Code-requirements)
* [Instalation](#Instalation)
* [Usage](#Usage)
* [How it works? Precise explanation](#How-it-works?-Precise-explanation)
* [Status](#Status)
* [Contributing](#Contributing)
* [Bibliography, inspiration and sources](#Bibliography,-inspiration-and-sources)
* [License](#License)

## What is Real Time Webcam Sudoku Solver?
This is a program written in Python that connects with your webcam and tries to solve a popular puzzle called [sudoku](https://en.wikipedia.org/wiki/Sudoku). Of course I'm not the first person which write such a program. 
I was inspired by [this project](https://github.com/murtazahassan/OpenCV-Sudoku-Solver) that 
I came across thanks to [this YouTube video](https://youtu.be/qOXDoYUgNlU).
I recognized that this type of project has a great potential and decided to write my own version.
At the end I will mention more about the sources I used.

## Code requirements
Python 3.8 with following modules installed:
* NumPy 1.18 
* TensorFlow 2.3 
* Keras 2.4
* Matplotlib 3.3 (if you want to train a model that recognizes digits by your own)
* OpenCV 4.4

But other versions of that libraries can also work.
If you already have any of those libraries installed first try with your version.
If sth doesn't work, then install the version of that library that I proposed.

## Instalation
Simply download the project as a compressed folder or clone it.
Then you have to make sure that [Code requirements](#Code-requirements) are met.
To check for yourself how the program works you don't have to train your CNN model. 
Already trained is saved in Models folder. 
Using Terminal/Command Prompt navigate to the correct directory and run main_file.py using the following command: python main_file.py

## Usage
After running main_file.py you should see a window that shows live feed from your webcam.
Now place a sudoku in the webcam's field of view.
And that's all. In the window should appear a solution.
If the solution doesn't appear, or the program doesn't even locate the sudoku, try to move it closer/further to the webcam. If it doesn't help, you may need to improve the lighting quality.

## How it works? Precise explanation
Algorithm:
* read a frame from a webcam
* convert the frame into grayscale
* binarize the frame
* find all external contours
* get the biggest quadrangle from that contours
* apply warp transform (bird eye view) on the quadrangle
* split the quadrangle into 81 small boxes
* check which boxes contain digits
* extract digits from the boxes that aren't empty
* prepare the digits for a CNN model
* while sudoku isn't solved and iterations of the loop <= 4:
	* classify the digits using the CNN model
	* compare the digits with a previous solution
	* if the digits are part of the previous solution then we don't need to solve sudoku again
	* solve sudoku if it is necessary
	* draw solution
* return a copy of the frame (with a solution if any were found)

Explanation with code analysis:  
The program runs in main loop in main_file.py. In every iteration of the loop a frame is read from a webcam.
That frame is passed as an argument to solve function, where everything interesting happens.
The function returns a copy of that frame with a drawn solution. 
```python
output_frame = webcam_sudoku_solver.solve(frame)
```
output_frame is shown and iteration of main loop ends.
Everything interesting happens in solve function. It is a heart of the program.
To check how it works we need to move to webcam_sudoku_solver.py.  

First task of the function is to extract a sudoku board. How to do it? 
We can treat sudoku boards as the biggest quadrangles in the frame.
I'm not going to explain how get_biggest_quadrangle function works, but
if you are curious about this, you can check this out by your own,
get_biggest_quadrangle and other functions that I won't discuss in detail are defined below WebcamSudokuSolver class.
```python
if frame is None:
	return frame

frame = deepcopy(frame)

warp_sudoku_board, warp_matrix = get_biggest_quadrangle(frame)

if warp_sudoku_board is None:
	return frame
```
As you can see if the function won't solve sudoku it will return an unchaged copy of a frame.
Next step is to split the board into 81 boxes.
```python
boxes = get_boxes(warp_sudoku_board)
```
If you want to know the exact way this function works then you need to find its declaration in webcam_sudoku_solver.py file.  
When a box is empty and when a box contains a digit? This is a very good question.
Using the trial and error technique, I developed the following algorithm:
* copy a box
* crop the copy on each side by 15%
* find all external contours
* if there is no contours it means there is no digit
* if there is at least one external contour then only the biggest could be a digit
* if an area is too small it means there is no digit
* get bounding rectangle of the biggest contour
* if width and height of this rectangle is too small it means there is no digit
* otherwise there is a digit
The algorithm is implemented in check_digits_occurrence function
```python
digits_occurrence = check_digits_occurrence(boxes)
```
Now it's time to get inputs for CNN model from boxes that contain digits
```python
inputs = prepare_inputs(boxes, digits_occurrence)
if inputs is None:
	return frame
```
The program works with sudoku rotated in every way. It means we need to try to solve it even 4 times.
```python
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
```
Let's analyze this loop step by step.
Why it has at most 4 iterations? A cropped and warped sudoku board that is returned by get_biggest_quadrangle function may be rotated only in 4 ways - by 0, 90, 180 or 270 degrees. We have to try to solve that board in every rotation,
because we don't know which is correct.  
First an angle is calculated and inputs for a CNN model are rotated. 
```python
rotation_angle = self.last_solved_sudoku_rotation + 90 * (current_attempt - 1)

rotated_inputs = rotate_inputs(inputs, rotation_angle)
```
Now the CNN model can predict.
```python
predictions = self.model.predict([rotated_inputs])
```
If an average probability isn't good it means current rotation isn't correct. We can skip to the next iteration. Notice that if it is 4th iteration then the function won't solve sudoku and will return a copy of a frame without any changes.
If the average probability is high enought we can get a grid with recognized digits.
```python
digits_grid = get_digits_grid(predictions, digits_occurrence, rotation_angle)
```
Regardless of the rotation of the sudoku in real life (in front of the webcam), this function will return a vertically oriented grid ("head up"). It allows to compare it with a previous solution, no matter how it was rotated.
```python
if self.new_sudoku_solution_may_be_last_solution(digits_grid):
```
If this sudoku solution can be equal to the previous we don't have to solve the sudoku at all.
```python
if self.new_sudoku_solution_may_be_last_solution(digits_grid):
	self.last_solved_sudoku_rotation = rotation_angle

	result = inverse_warp_digits_on_frame(
		digits_grid, self.last_sudoku_solution, frame, warp_sudoku_board.shape, warp_matrix, rotation_angle
	)

	return result 
```
Otherwise solve function will try to solve the sudoku.
```python
solved_digits_grid = sudoku_solver.solve_sudoku(digits_grid)
```
If the sudoku is unsolvable it means that the current rotation isn't correct after all.
```python
if solved_digits_grid is None:
	current_attempt += 1
	continue
```
But if the sudoku has been solved we overwrite the previous solution and draw it on a copy of the current frame.
```python
self.last_sudoku_solution = solved_digits_grid
self.last_solved_sudoku_rotation = rotation_angle

result = inverse_warp_digits_on_frame(
	digits_grid, solved_digits_grid, frame, warp_sudoku_board.shape, warp_matrix, rotation_angle
)

return result
```
If we couldn't find a solution to the sudoku in any rotation, we return the image without a solution.
```python
return frame
```
And this is how solve function works, if you are curious about how the utilities-functions that were called and
did a lot of nice things then check their definitions and descriptions which are located in webcam_sudoku_solver.py file below WebcamSudokuSolver class.

But there is also one more point to be discussed. How does solve function solve a sudoku puzzle?
```python
solved_digits_grid = sudoku_solver.solve_sudoku(digits_grid)
```
To check how it works we need to move to sudoku_solver.py.
This file will be discussed in exactly the same way as the previous one -
I will discuss in detail only an externally called function.
The whole algorithm starts and ends in solve_sudoku function.
First we need to check if a sudoku is solvable at all. 
```python
if not is_solvable(digits_grid):
	return None
```
The algorithm is based on pencilmarks that we use to help ourself solve sudoku in real life.  
I called it human_notes.
```python
human_notes = get_full_human_notes(digits_grid)
```
Sudoku is solved in a loop.
```python
while True:
	sth_has_changed1 = remove_orphans_technique(digits_grid, human_notes)

	sth_has_changed2 = single_appearances_technique(digits_grid, human_notes)

	if not sth_has_changed1 and not sth_has_changed2:
		break
```
Each iteration of the loop calls two functions: remove_orphans_technique and single_appearances_technique.
Their task is to successively delete unnecessary notes and complete the sudoku.
The loops ends when the functions doesn't change anything anymore. It means the sudoku is solved or can't be solved using this technique.  
After the loop we check if the sudoku is solved correctly (so we check also if is solved at all).
```python
if is_solved_correctly(digits_grid):
	return digits_grid
return None
```
Note that a very popular technique for solving sudoku is backtracking algorithm, but the program didn't use it because it works too slowly on more difficult puzzles. However it has one advantage - can solve sudoku with more than one solution. My algorithm can't do it, because it is an ambigous case. The loop will break because those 2 functions inside it can change only obvious digits.

## Status
Project is _finished_, but there are still things that can be improved:
* a new way of locating a sudoku board that can be described by the following algorithm:
  - create a binarized image that contains only linear segments that can be found using probabilistic Hough transform;
  - apply earlier algorithm (finding the biggest quadrangle) on just created image.
  This approach probably greatly increases the chance of finding a board because
  in each picture there are fewer quadrangles that are also grids than any quadrangles.
  Sudoku board will almost always be the only one grid on an image.
* Second thing that can be improved is a cnn model that classifies digits.  
  The current model is trained on [MNIST dataset](http://yann.lecun.com/exdb/mnist/) but
  if want the program to work correctly on printed sudoku you have to use a model trained on 
  [Chars74K dataset](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/).

## Contributing
Feel free to contribute to this project.
How to do it?
(Even I'm not sure because I have never contributed to any project),
but I guess the best way to do it is:
* Fork the repo on GitHub
* Clone the project to your own machine
* Commit changes to your own branch
* Push your work back up to your fork
* Submit a Pull request so that I can review your changes
NOTE: Be sure to merge the latest from "upstream" before making a pull request!

## Bibliography, inspiration and sources
As I wrote before, I was inspired by [this project](https://github.com/murtazahassan/OpenCV-Sudoku-Solver).
To master the basics of OpenCV library I watched a [tutorial](https://youtu.be/WQeoO7MI0Bs) 
that is also made by that guy.
Then I started looking for other augmented reality/webcam sudoku solving projects.
Many of them had very nice features, but none were perfect.
The following deserve special mention:
https://youtu.be/QR66rMS_ZfA  

https://youtu.be/uUtw6Syic6A
https://github.com/anhminhtran235/real_time_sudoku_solver

Even though my project may seem at first glance just a combination of the best features of those mentioned above, creating it was not a trivial task. It took me more than 1 month (at least 100 hours).
The most engaging was to come up with a better way to extract digits from a board for the neural network that would work for any sudoku - both those with large and small digits and both those with thick and thin grids.
But the most important thing is that while writing this project I learned LOTS OF new things.

## License
At least mention me in your README.
