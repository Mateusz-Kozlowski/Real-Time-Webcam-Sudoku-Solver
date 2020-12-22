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
This is a program written in Python that conects with your webcam and tries to solve a popular puzzle called [sudoku](https://en.wikipedia.org/wiki/Sudoku).



-> [(COMING SOON!) LINK TO A VIDEO SHOWING HOW THE PROGRAM EXACTLY WORKS]() <-

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


## Usage


## How it works? Precise explanation


## Status
Project is _finished_, but there are still things that can be improved:
* a way of locating a sudoku board:

  currently program tries to find the biggest quadrangle and treats it as a board:
  ```python
  warp_sudoku_board, warp_matrix = get_biggest_quadrangle(frame)
  ```
  but there is a better way to do it - first find linear segments using probabilistic Hough transform
  and then find the biggest quadrangle on an image created from that segments.
  This approach causes the program to find only quadrangles that are also grids.
* a model that classifies the digits:

  the current model is trained 
  
  Adding a new CNN model trained on Chars74K dataset (Computer Fonts) instead of MNIST for handwritten digits.
  http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

## Contributing


## Bibliography, inspiration and sources


## License
If you want to use my project, please try to contact me somehow.
