print('Due TensorFlow library it may take a few seconds before it starts...')

import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

MODEL = tf.keras.models.load_model('cnn model.h5', custom_objects=None, compile=True)
WIN_WIDTH, WIN_HEIGHT = 1920, 1080


def preprocess_image(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred_gray_img = cv.GaussianBlur(gray_img, (5, 5), 1)
    img_threshold = cv.adaptiveThreshold(blurred_gray_img, 255, 1, 1, 11, 2)
    return gray_img, img_threshold


def find_biggest_quadrangle(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv.contourArea(i)
        peri = cv.arcLength(i, True)
        approx = cv.approxPolyDP(i, 0.02 * peri, True)
        if area > max_area and len(approx) == 4:
            biggest = approx
            max_area = area
    return biggest


def reorder(my_points):
    my_points = my_points.reshape((4, 2))
    my_points_new = np.zeros((4, 1, 2), dtype=np.int32)
    add = my_points.sum(1)
    my_points_new[0] = my_points[np.argmin(add)]
    my_points_new[3] = my_points[np.argmax(add)]
    diff = np.diff(my_points, axis=1)
    my_points_new[1] = my_points[np.argmin(diff)]
    my_points_new[2] = my_points[np.argmax(diff)]
    return my_points_new


def split_squares(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)
    return boxes


def get_predictions(squares):
    probabilities = list()
    digits = list()
    index = 0
    for square in squares:
        index += 1
        img = square.copy()
        cv.imshow('img', cv.resize(img, (400, 400), cv.INTER_LINEAR))
        img = img[7:img.shape[0] - 7, 7:img.shape[1] - 7]
        cv.imshow('cutted', cv.resize(img, (400, 400), cv.INTER_LINEAR))
        img = cv.bitwise_not(img)
        threshold, img = cv.threshold(img, 150, 255, cv.THRESH_BINARY)
        img = cv.resize(img, (28, 28), interpolation=cv.INTER_AREA)
        img = img.reshape(1, 28, 28, 1)
        img = img / 255.0
        predictions = MODEL.predict([img])[0]
        probability = int(max(predictions) * 100)
        probabilities.append(probability)
        if probability < 80:
            digits.append(0)
        else:
            digits.append(np.argmax(predictions))
        print(digits[-1])
        plt.imshow(img[0], cmap="gray")
        plt.show()
    return digits, probabilities


def main():
    webcam = cv.VideoCapture(0)
    webcam.set(cv.CAP_PROP_FRAME_WIDTH, WIN_WIDTH)
    webcam.set(cv.CAP_PROP_FRAME_HEIGHT, WIN_HEIGHT)

    while True:
        # STEP 1 - GET A WEBCAM FRAME
        (successful_frame_read, frame) = webcam.read()

        if not successful_frame_read:
            break

        # STEP 2 - PREPROCESS
        gray_img, img_threshold = preprocess_image(frame)

        cv.imshow('gray_img', gray_img)

        # STEP 3 - FIND CONTOURS
        contours, hierarchy = cv.findContours(img_threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # STEP 4 - FIND THE BIGGEST QUADRANGLE
        biggest_quadrangle = find_biggest_quadrangle(contours)

        if biggest_quadrangle.size > 0:
            # STEP 5 - APPLY WARP PERSPECTIVE
            width_img, height_img = 504, 504
            biggest_quadrangle = reorder(biggest_quadrangle)
            pts1 = np.float32(biggest_quadrangle)  # PREPARE POINTS FOR WARP
            pts2 = np.float32([[0, 0], [width_img, 0], [0, height_img], [width_img, height_img]])  # PREPARE POINTS FOR WARP
            matrix = cv.getPerspectiveTransform(pts1, pts2)  # GER
            warp_img = cv.warpPerspective(frame, matrix, (width_img, height_img))
            gray_warp_img = cv.cvtColor(warp_img, cv.COLOR_BGR2GRAY)

            cv.imshow('gray warp', gray_warp_img)

            blur = cv.GaussianBlur(gray_warp_img, (13, 13), 0)
            cv.imshow("blur1", blur)

            thresh = cv.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
            cv.imshow("thresh1", thresh)

            cv.waitKey(0)


            img_detected_digits = np.zeros((height_img, width_img, 3), np.uint8)

            failed_approaches, solved, average_effectiveness_is_fine = 0, False, False
            while failed_approaches < 4:
                # STEP 6 - SPLIT WARP SUDOKU BOARD IMAGE INTO 81 SMALL SQUARES
                squares = split_squares(gray_warp_img)
                digits, probabilities = get_predictions(squares)
                for i in range(9):
                    for j in range(9):
                        print(str(digits[9 * i + j]) + ' ', end="")
                    print('')
                cv.waitKey(0)
                failed_approaches += 1

            # cv.imshow('Webcam Sudoku Solver', frame)
            # cv.imshow('Warp', warp_img)

        key = cv.waitKey(16)  # One frame last at least 16 ms

        if key == 27:  # Esc key has been pressed
            pass
            # break


if __name__ == '__main__':
    main()

print('Code is done, so everything works fine!')
