"""
Module with img pre-processing functions for the OCR imgs.
"""
import numpy as np
import PIL
from PIL import Image

import pyocr

TMP_RECEIPTS_FILE = "/home/igor/Documents/others/ocr/ocr/tmp_receipts/"
INPUT_RECEIPT = (
    "/home/igor/Documents/others/ocr/ocr/sample_receipts/bad_receipt2.png"
)


def set_image_dpi(file_path):
    """
    Sets a fixed DPI fore the OCR engine. Returns the img.
    """
    im = Image.open(fp=file_path)

    length_x, width_y = im.size
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)

    resized_img = im.resize(size, Image.ANTIALIAS)
    resized_img.save(TMP_RECEIPTS_FILE + "resized.png", dpi=(300, 300))

    return resized_img


def norm_image(img):
    """
    Normalize PIL image

    Normalizes luminance to (mean,std)=(0,1),
    and applies a [1%, 99%] contrast stretch
    """
    img_y, img_b, img_r = img.convert("YCbCr").split()

    img_y_np = np.asarray(img_y).astype(float)

    img_y_np /= 255
    img_y_np -= img_y_np.mean()
    img_y_np /= img_y_np.std()
    scale = np.max(
        [
            np.abs(np.percentile(img_y_np, 1.0)),
            np.abs(np.percentile(img_y_np, 99.0)),
        ]
    )
    img_y_np = img_y_np / scale
    img_y_np = np.clip(img_y_np, -1.0, 1.0)
    img_y_np = (img_y_np + 1.0) / 2.0

    img_y_np = (img_y_np * 255 + 0.5).astype(np.uint8)

    img_y = Image.fromarray(img_y_np)

    img_ybr = Image.merge("YCbCr", (img_y, img_b, img_r))

    img_nrm = img_ybr.convert("RGB")

    return img_nrm


def resize_image(img, size):
    """
    Resize PIL image

    Resizes image to be square with sidelength size. Pads with black if needed.
    """
    # Resize
    n_x, n_y = img.size
    if n_y > n_x:
        n_y_new = size
        n_x_new = int(size * n_x / n_y + 0.5)
    else:
        n_x_new = size
        n_y_new = int(size * n_y / n_x + 0.5)

    img_res = img.resize((n_x_new, n_y_new), resample=PIL.Image.BICUBIC)

    # Pad the borders to create a square image
    img_pad = Image.new("RGB", (size, size), (128, 128, 128))
    ulc = ((size - n_x_new) // 2, (size - n_y_new) // 2)
    img_pad.paste(img_res, ulc)

    return img_pad


def prep_images(path, out_dir=TMP_RECEIPTS_FILE):
    """
    Preprocess images

    Reads images in paths, and writes to out_dir

    """
    img = Image.open(path)
    img_nrm = norm_image(img)
    img_res = resize_image(img_nrm, 800)
    # img_res.save(TMP_RECEIPTS_FILE + "test.png")

    return img_res


def main():
    """
    Starts all the img pre-processing steps.
    """
    tools = pyocr.get_available_tools()[0]

    # print("no pre processing:")
    # text = tools.image_to_string(
    #     Image.open(INPUT_RECEIPT), builder=pyocr.builders.TextBuilder()
    # )
    # print(text)

    print("pre processed:")
    img_resized = prep_images(INPUT_RECEIPT)
    img_resized.draw()

    text = tools.image_to_string(
        img_resized, builder=pyocr.builders.DigitBuilder()
    )

    print(text)


if __name__ == "__main__":
    main()
