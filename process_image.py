import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from histogram import hog as invoke_hog

a, b = int(250 / 4), int(250 * 3 / 4)


def get_image(file_name):
    return Image.open("data/" + file_name)


def process_image(input_image, crop=True, hog=True):
    img = input_image.convert("L")
    img = np.array(img)
    if crop:
        img = img[a:b, a:b]
    if hog:
        img = invoke_hog(img)
    return img


if __name__ == "__main__":
    img = process_image(get_image("lfw_funneled_normalized/Aaron_Eckhart_0001.jpg"), hog=False)
    img = process_image(get_image("my_image/Kasia_0001.jpg"), crop=False, hog=False)
    plt.imshow(img)
    plt.show()
