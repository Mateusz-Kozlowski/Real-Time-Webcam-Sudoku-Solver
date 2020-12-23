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
* a new way of locating a sudoku board that can be described by the following algorithm:
  - create a binarized image that contains only linear segments that can be found using probabilistic Hough transform;
  - apply earlier algorithm (finding the biggest quadrangle) on just created image.
  This approach probably greatly increases the chance of finding a board because
  in each picture there are fewer quadrangles that are also grids than any quadrangles.
  Sudoku boards will almost always be the only grids on the image.
* Second thing that can be improved is a cnn model that classifies digits.  
  The current model is trained on [MNIST dataset](http://yann.lecun.com/exdb/mnist/) but
  if want the program to work correctly on printed sudoku you have to use a model trained on 
  [Chars74K dataset](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/).

## Contributing


## Bibliography, inspiration and sources


## License
If you want to use my project, please try to contact me somehow.
