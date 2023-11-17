import numpy as np


def add_impulse_noise(image: np.ndarray,ratio: float):
    x = image.shape[0]
    y = image.shape[1]

    # get list of pixel coordinates
    coords = list()
    for i in range(x):
        for j in range(y):
            coords.append((i, j))

    #shuffle the list
    np.random.shuffle(coords)

    # get number of pixels to be changed
    num_pixels = int(ratio * x * y)
    for i in range(num_pixels):
        x, y = coords[i]
        image[x, y] = np.random.choice([0, 255])

    return image

