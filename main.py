import cv2
import matplotlib.pyplot as plt
import numpy as np
import math as m
import sys

refPt = []  # selected points of line to analize
measurePt = []  # selected points of line to measure
PIX_TO_SM = 1.0


def set_line(event, x, y, flags, param):
    global img, clone, measure_clone, measurePt, PIX_TO_SM

    if event == cv2.EVENT_MBUTTONUP:
        print((x, y), img[x, y])

    elif event == cv2.EVENT_RBUTTONDOWN:
        if measurePt:
            measurePt.clear()
            img = clone.copy()
        measurePt.append((x, y))
    elif event == cv2.EVENT_RBUTTONUP:
        if (x, y) != measurePt[0]:
            measurePt.append((x, y))
            cv2.line(img, measurePt[0], measurePt[1], 127, 3)
            print(img[measurePt[0]])
            # print("Measure Points: ", measurePt)
            cv2.imshow("image", img)
            PIX_TO_SM = ((measurePt[0][0] - x) ** 2 + (measurePt[0][1] - y) ** 2) ** 0.5
            measure_clone = img.copy()
            print(PIX_TO_SM)
        else:
            print(x, y, average(img, x, y))

    elif event == cv2.EVENT_LBUTTONDOWN:
        plt.close()
        if refPt:
            refPt.clear()
            img = measure_clone.copy()
        refPt.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        if (x, y) != refPt[0]:
            refPt.append((x, y))
            # print("Points: ", refPt)
            cv2.imshow("image", img)
            make_plot()
        else:
            print(x, y, img[x, y])


def make_plot():
    n_points = 70
    x = np.linspace(refPt[0][0], refPt[1][0], n_points, dtype=int)
    y = np.linspace(refPt[0][1], refPt[1][1], n_points, dtype=int)
    pix = np.zeros(n_points)
    for i in range(n_points):
        pix[i] = average(img, x[i], y[i], 3, 3)
        cv2.circle(img, (x[i], y[i]), 1, 255, -1)

    plt.plot(((x - x[0]) ** 2 + (y - y[0]) ** 2) ** 0.5 / PIX_TO_SM, pix)
    # plt.scatter(((x - x[0]) ** 2 + (y - y[0]) ** 2) ** 0.5 / PIX_TO_SM, img[x, y])
    # plt.scatter(x, y, c=img[x, y])
    plt.xlabel('Расстояние от центра, см')
    plt.ylabel('Яркость вдоль линии, у.е.')
    # xmin, xmax, ymin, ymax = plt.axis()
    # plt.axis([xmin, xmax, 0, 255])
    plt.grid(True)
    cv2.imshow("image", img)
    plt.show()


def average(pixels, y, x, dx=4, dy=4):
    area = pixels[max(x - dx, 0):min(x + dx + 1, pixels.shape[0]),
                  max(y - dy, 0):min(y + dy + 1, pixels.shape[1])]
    return np.average(area)


def maximaze(pixels, x, y, dx=2, dy=2):
    area = pixels[max(x - dx, 0):min(x + dx + 1, pixels.shape[0]),
                  max(y - dy, 0):min(y + dy + 1, pixels.shape[1])]
    return np.amax(area)


def median(pixels, x, y, dx=2, dy=2):
    area = pixels[max(x - dx, 0):min(x + dx + 1, pixels.shape[0]),
                  max(y - dy, 0):min(y + dy + 1, pixels.shape[1])]
    return np.median(area)


def main():
    args = sys.argv[1:]
    global img, clone
    np.set_printoptions(threshold=sys.maxsize)
    cv2.setUseOptimized(True)

    # Изображение должно быть квадратным!!!
    if args:
        img = cv2.imread(args[0])
    else:
        img = cv2.imread("test_image.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clone = img.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", set_line)

    # Блюр для усреднения
    # kernel = np.ones((5, 5), np.float32) / 25
    # img = cv2.filter2D(img, -1, kernel)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    while True:
        cv2.imshow("image", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c") or key == 27:  # ESCAPE button
            break


if __name__ == '__main__':
    main()
