# Real-Time-Webcam-Sudoku-Solver

## Table of contents
* [What is Real Time Webcam Sudoku Solver?](#What-is-Real-Time-Webcam-Sudoku-Solver?)
* [Screenshots](#Screenshots)
* [Code requirements](#Code-requirements)
* [Instalation](#Instalation)
* [Usage](#Usage)
* [How it works? Precise explanation](#How-it-works?-Precise-explanation)
* [Status](#Status)
* [Contributing](#Contributing)
* [Bibliography, inspiration and sources](#Bibliography,-inspiration-and-sources)
* [License](#License)

## What is Real Time Webcam Sudoku Solver?
Purposes, motivation ect.
and sth

-> [(COMING SOON!) LINK TO A VIDEO SHOWING HOW THE PROGRAM EXACTLY WORKS]() <-

## Screenshots


## Code requirements
Python 3.8 with following modules installed:
* Keras 2.4
* Matplotlib 3.3
* NumPy 1.18
* OpenCV 4.4
* TensorFlow 2.3

But others versions of that libraries can also work.
If you already have any of those libraries installed first try with your version.
If sth doesn't work, then install the version of that library that I proposed.

## Instalation


## Usage


## How it works? Precise explanation


## Status
Project is: _finished_, but there are still many things that can be improved:
* changing the way of locating a sudoku board:

  currently program tries to find the biggest quadrangle and treats it as a board:
  ```python
  warp_matrix, warp_sudoku_board = get_biggest_quadrangle(frame)
  ```
  but there is a better way to do it - first find linear segments using probabilistic Hough transform
  and then find the biggest quadrangle on an image created from that segments.
  This approach causes the program to find only quadrangles that are also grids. 


## Contributing


## Bibliography, inspiration and sources


## License
If you want to use my project, please try to contact me somehow.
