-> [LINK TO A VIDEO SHOWING HOW THE PROGRAM EXACTLY WORKS](https://drive.google.com/file/d/1hQUiYHpJQpClfQFXCh4waUWeijJky7jv/view?usp=sharing) <-

# Real-Time-Webcam-Sudoku-Solver

## Table of contents
* [What is Real Time Webcam Sudoku Solver?](#What-is-Real-Time-Webcam-Sudoku-Solver?)
* [Code requirements](#Code-requirements)
* [Instalation](#Instalation)
* [Usage](#Usage)
* [How it works?](#How-it-works?)
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

## How it works?
Short explanation - algorithm:
* read a frame from a webcam
* convert that frame into grayscale
* binarize that frame
* find all external contours
* get the biggest quadrangle from that contours
* apply warp transform (bird eye view) on the biggest quadrangle
* split that quadrangle into 81 small boxes
* check which boxes contain digits
* extract digits from boxes that aren't empty
* prepare that digits for a CNN model
* while not solved and iterations of the loop <= 4:
	* rotate the digits by (90 * current iteration) degrees
	* classify the digits using a CNN model
	* if an average probability is too low go to the next iteration of the loop
	* compare the digits with a previous solution
	* if the digits are part of the previous solution then we don't need to solve the sudoku again - break the loop
	* try to solve the sudoku
	* if solved correctly break the loop
* return a copy of the frame (with a solution if any were found)

Precise explanation - code analysis:  

As I said before, the program starts in main_file.py.  
First of all we have to import a source code and libraries from other files.
```python
print('Importing a source code and libraries from other files...')

from webcam_sudoku_solver import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide tf warnings
import tensorflow as tf
```

Then main function can start.  
First task of the function is to prepare a CNN model and a webcam.
```python
model = tf.keras.models.load_model('Models/handwritten_cnn.h5')

webcam_width, webcam_height = 1920, 1080
webcam = cv.VideoCapture(0)
webcam.set(cv.CAP_PROP_FRAME_WIDTH, webcam_width)
webcam.set(cv.CAP_PROP_FRAME_HEIGHT, webcam_height)
```

Now main loop of the program can start.  
We'll use there a object of WebcamSudokuSolver class - the core of the program.  
```python
# create the core of the program
webcam_sudoku_solver = WebcamSudokuSolver(model)
```

At the beginning of each iteration of main loop a frame is read from a webcam.  
Then that frame is passed as an argument to the object of WebcamSudokuSolver class using solve function.  
The function returns a copy of that frame (with a drawn solution if any has been found).  
How does solve function convert a webcam frame into a frame with solution? I'll explain it in a moment.
But now let's see what happens with that returned frame.  
That frame is just displayed.
We also check if a user has pressed a key (if so, the program is closed).

```python
print('Logs:')
while webcam.isOpened():
	successful_frame_read, frame = webcam.read()

	if not successful_frame_read:
		break

	# run the core of the program
	output_frame = webcam_sudoku_solver.solve(frame)

	# output results
	cv.imshow('Webcam Sudoku Solver', output_frame)

	# check if a user has pressed a key, if so, close the program
	if cv.waitKey(1) >= 0:
		break

cv.destroyAllWindows()
webcam.release()
```

If there are no errors, the following information will be displayed at the very end of the program:  
"Code is done, so everything works fine!".  

But how does solve function convert a webcam frame into a frame with solution?

To answer this question we have to move to webcam_sudoku_solver.py file. 

First task of the function is to extract a sudoku board.  
We can treat a sudoku board as the biggest quadrangle in a frame.
I'm not going to explain how get_biggest_quadrangle function exactly works,  
but if you are curious about this, you can check this out by your own.
All functions that I won't discuss in detail are defined under WebcamSudokuSolver class.
```python
if frame is None:
	return frame

frame = deepcopy(frame)

warp_sudoku_board, warp_matrix = get_biggest_quadrangle(frame)

if warp_sudoku_board is None:
	return frame
```
As you can see if the function won't solve a sudoku it will return an unchaged copy of a frame.  

Next step is to split that board into 81 boxes.
```python
boxes = get_boxes(warp_sudoku_board)
```

When is a box empty and when does a box contain a digit?  
This is a very good question.
Using trial and error technique, I developed the following algorithm:
* copy a box
* crop that copy on each side by 15%
* find all external contours
* if there are no contours it means there is no digit - return False
* if there is at least one external contour get the biggest (only the biggest could be a digit)
* if an area of that contour is too small it means there is no digit - return False
* get a bounding rectangle of the biggest contour
* if width and height of that rectangle is too small it means there is no digit - return False
* return True - there is a digit

The algorithm is implemented in check_digits_occurrence function
```python
digits_occurrence = check_digits_occurrence(boxes)
```

Now it's time to get inputs for a CNN model from boxes that contain digits
```python
inputs = prepare_inputs(boxes, digits_occurrence)
if inputs is None:
	return frame
```

The program works with sudoku rotated in every way,  
but cropped and warped boards which are returned by get_biggest_quadrangle function may be rotated only in 4 ways - by 0, 90, 180 or 270 degrees.  
That's just how get_biggest_quadrangle function works.

We don't know which rotation is correct, so we need to try solve it even 4 times.
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

First an angle is calculated and inputs for a CNN model are rotated. 
```python
rotation_angle = self.last_solved_sudoku_rotation + 90 * (current_attempt - 1)

rotated_inputs = rotate_inputs(inputs, rotation_angle)
```

Now a CNN model can predict.
```python
predictions = self.model.predict([rotated_inputs])
```

If an average probability isn't good it means the current rotation isn't correct. We can skip to the next iteration.  
Notice that if it is 4th iteration then the function won't solve a sudoku and will return a copy of a frame without any changes.  
```python
if not probabilities_are_good(predictions):
	current_attempt += 1
	continue
```

If an average probability is high enought we can get a grid with recognized digits.
```python
digits_grid = get_digits_grid(predictions, digits_occurrence, rotation_angle)
```

This function always returns a "vertically normalized" grid - it always can be compared with the previous solution, regardless of their rotation.  

Comparing the current grid with the previous solution:
```python
if self.new_sudoku_solution_may_be_last_solution(digits_grid):
```

If a solution of the current grid can be equal to the previous solution we don't have to solve the current sudoku at all.
```python
if self.new_sudoku_solution_may_be_last_solution(digits_grid):
	self.last_solved_sudoku_rotation = rotation_angle

	result = inverse_warp_digits_on_frame(
		digits_grid, self.last_sudoku_solution, frame, warp_sudoku_board.shape, warp_matrix, rotation_angle
	)

	return result 
```

Otherwise solve function will try to solve the current sudoku.
```python
solved_digits_grid = sudoku_solver.solve_sudoku(digits_grid)
```

If that sudoku is unsolvable it means that the current rotation isn't correct after all.
```python
if solved_digits_grid is None:
	current_attempt += 1
	continue
```

But if that sudoku has been solved correctly we overwrite the previous solution.
```python
self.last_sudoku_solution = solved_digits_grid
self.last_solved_sudoku_rotation = rotation_angle
```

Draw the current solution on a copy of the current frame and return it.
```python
result = inverse_warp_digits_on_frame(
	digits_grid, solved_digits_grid, frame, warp_sudoku_board.shape, warp_matrix, rotation_angle
)

return result
```

If we couldn't find any solution of the sudoku in any rotation, we return the image without a solution.
```python
return frame
```
And this is how solve function works.  
If you are curious about utilities-functions that are called by solve function and does a nice job then check their definitions and descriptions which are located in webcam_sudoku_solver.py file below WebcamSudokuSolver class.

But there is also one more point to be discussed:  
How does solve_sudoku function solve a sudoku puzzle?
```python
solved_digits_grid = sudoku_solver.solve_sudoku(digits_grid)
```

To check how it works we need to move to sudoku_solver.py.
This file will be discussed in exactly the same way as the previous one -
I will discuss in detail only an externally called function
where the whole algorithm starts and ends.  

First we need to check if a sudoku is solvable at all. 
```python
if not is_solvable(digits_grid):
	return None
```

The algorithm is based on pencilmarks that we use to help ourself solve sudoku in real life.  
I called them human_notes.
```python
human_notes = get_full_human_notes(digits_grid)
```
A sudoku is solved in a loop.
```python
while True:
	sth_has_changed1 = remove_orphans_technique(digits_grid, human_notes)

	sth_has_changed2 = single_appearances_technique(digits_grid, human_notes)

	if not sth_has_changed1 and not sth_has_changed2:
		break
```
Each iteration of the loop calls two functions: remove_orphans_technique and single_appearances_technique.
Their task is to successively delete unnecessary notes and complete a sudoku.
The loop ends when the functions doesn't change anything anymore. It means the sudoku is solved or can't be solved using this technique.  
After the loop we check if that sudoku is solved correctly (so we check also if is solved at all).
```python
if is_solved_correctly(digits_grid):
	return digits_grid
return None
```

There is a very popular technique for solving sudoku called backtracking algorithm. I didn't choose that technique because it works too slowly on more difficult puzzles. However, it has one advantage - can solve sudoku with more than one solution. My algorithm can't do it, because it is an ambigous case.

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
(Even I am not sure because I have never contributed to any project),
but I guess the best way to do it is:
* Fork this repo on GitHub
* Clone the project to your own machine
* Commit changes to your own branch
* Push your work back up to your fork
* Submit pull request so that I can review your changes  

NOTE: Be sure to merge the latest from "upstream" before making a pull request!

## Bibliography, inspiration and sources
As I wrote before, I was inspired by [this project](https://github.com/murtazahassan/OpenCV-Sudoku-Solver).
To master the basics of OpenCV library I watched a [tutorial](https://youtu.be/WQeoO7MI0Bs) 
that is also made by that guy.
Then I started looking for other augmented reality/webcam sudoku solving projects.
Many of them had very nice features, but none was perfect.
The following projects deserve a special mention:  
* https://youtu.be/QR66rMS_ZfA  

* https://youtu.be/uUtw6Syic6A  
https://github.com/anhminhtran235/real_time_sudoku_solver

Even though my project may seem at first glance just a combination of the best features of those mentioned above, creating it was not a trivial task. It took me more than 1 month (at least 100 hours).
The most engaging was to come up with a better way to extract digits from a board for the neural network that would work for any sudoku - both those with large and small digits and both those with thick and thin grids.
Another difficult task was the implementation of solving rotated sudoku.  
But what I am most proud of is something that is imperceptible at first glance - my own sudoku puzzle algorithm, which I have unconsciously developed over the past few years. Its implementation was a very interesting experience.  

But THE MOST IMPORTANT thing is that while writing this project I learned A LOT OF new things.

## License
I would be very grateful if you mention my project in your readme, if it was an inspiration for you.

##
Mateusz Kozłowski 2020
