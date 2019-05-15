import numpy as np
import cv2
import random as rng
import matplotlib.pyplot as plt
import sys


def analyze_image(image):
    image = image.astype(np.uint8)

    # Crop the image, such that the person is basically the only thing in focus
    y = 30
    h = 80
    x = 60
    w = 60
    res = 1.5/w  # store resolution of image for converting pixels to meters
    image = image[y:y + h, x:x + w]

    # Set all values in the image greater than 3.0 to 255, and less than 3.0 to 0. This puts the person in focus
    ret, thresh = cv2.threshold(image, 3.0, 255, cv2.THRESH_BINARY)

    # Make a copy of the image, as findContours changes it
    im_copy = image.copy()

    # Find all contours in the image
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Approximate contours to polygons + get bounding rects and circles
    contours_poly = [None] * len(contours)
    bound_rect = [None] * len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        bound_rect[i] = cv2.boundingRect(contours_poly[i])

    drawing = np.zeros((im_copy.shape[0], im_copy.shape[1], 3), dtype=np.uint8)

    # Draw polygonal contour + bonding rects
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) < 100:
            continue
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv2.drawContours(drawing, contours_poly, i, color)
        cv2.rectangle(drawing, (int(bound_rect[i][0]), int(bound_rect[i][1])),
                      (int(bound_rect[i][0] + bound_rect[i][2]), int(bound_rect[i][1] + bound_rect[i][3])), color, 2)

    # Iterates from left of image until human is detected
    collision_row = 76
    curr_pixel = np.array([0, 0, 0])
    curr_pixel_index = np.array([collision_row, 3])
    while curr_pixel.sum() == 0:
        curr_pixel_index[1] += 1
        curr_pixel = drawing[curr_pixel_index[0], curr_pixel_index[1]]
    left_person = curr_pixel_index.copy()

    # Iterates from right of image until human is detected.
    curr_pixel = np.array([0, 0, 0])
    curr_pixel_index = np.array([collision_row, 57])
    while curr_pixel.sum() == 0:
        curr_pixel_index[1] -= 1
        curr_pixel = drawing[curr_pixel_index[0], curr_pixel_index[1]]
    right_person = curr_pixel_index.copy()

    # Calculates the overall location of the person relative to the ends of the hallways
    left = (left_person[1])*res
    right = (w - right_person[1])*res
    # pdb.set_trace()

    # Figures out which side is larger
    if left >= right:
        print("left " + str(left))
    elif right > left:
        print("right" + str(right))

    # plt.imshow(drawing)
    # plt.show()


def load_image(filename):
    return np.loadtxt(filename)


if __name__ == "__main__":
    input_name = str(sys.argv[1])
    curr_image = load_image(input_name)
    analyze_image(curr_image)

