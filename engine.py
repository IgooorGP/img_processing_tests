"""
Module with img pre-processing functions for the OCR imgs.
"""
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pyocr
from PIL import Image
import cv2
from statistics import mean
import pytesseract
import argparse

TMP_RECEIPTS_FILE = "./tmp_receipts/"
INPUT_RECEIPT = "./sample_receipts/bad_receipt2.png"


def parse_args():
    """
    Parses arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--image", required=True)
    parser.add_argument("-p", "--preprocess", type=str, default="black")
    parser.add_argument("-b", "--blur", default=True, nargs="?", type=bool)
    parser.add_argument("-r", "--resize", type=int)
    parser.add_argument("-s", "--save", type=str)
    # parser.add_argument("-rot", "--rotate")

    args = vars(parser.parse_args())

    return args


def rotate(im):
    coords = np.column_stack(np.where(im > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)

    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle

    # rotate the image to deskew it
    (h, w) = im.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        im, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


def main():
    """
    Starts all the img pre-processing steps.
    """
    args = parse_args()
    im = cv2.imread(args["image"])
    im_original = im.copy()
    rotated = ""

    if args.get("preprocess") == "black":
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    if args.get("blur"):
        im = cv2.medianBlur(im, 3)

    if args.get("resize"):
        size = args.get("resize")
        im = cv2.resize(im, (size, size))

    if args.get("save"):
        cv2.imwrite(args["save"], im)

    else:
        cv2.imshow("Input image", im_original)
        cv2.imshow("Output image", im)
        # cv2.imshow("Rotated image", rotated)

        cv2.waitKey()


if __name__ == "__main__":
    main()
