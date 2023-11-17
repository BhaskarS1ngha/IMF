import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


deltaMax = None
deltaMin = None


def get_constrained_mean(window: np.ndarray) -> float:
    rows, cols = window.shape
    noiseFreeIntensities = []
    for i in range(0,rows):
        for j in range(0,cols):
            if window[i,j] != deltaMin and window[i,j] != deltaMax:
                noiseFreeIntensities.append(window[i,j])
    if len(noiseFreeIntensities)>0:
        return np.mean(noiseFreeIntensities)
    else:
        return window[rows//2, cols//2]


def get_manhattan_distance(image1: np.ndarray, image2: np.ndarray) -> float:
    return np.sum(abs(image1-image2))


def get_noise_ratio(image: np.ndarray):
    flatImage = image.flatten()
    salts = np.where(flatImage==deltaMax)
    peppers = np.where(flatImage==deltaMin)
    ratio = (len(salts)+len(peppers))/len(flatImage)
    return ratio


def imf(image: np.ndarray):
    """
    Iterative mean filter for removal of impulse noies
    :param image: 2d numpy array of noisy image
    :return: 2d numpy array of filtered image
    """

    global  deltaMax, deltaMin
    imtype = image.dtype
    deltaMax = np.max(image)
    deltaMin = np.min(image)
    r = 1
    k=0
    epsilon = 0
    temp_img1 = np.pad(image,r,constant_values=0)

    rows, cols = temp_img1.shape
    while(1):
        temp_img2 = np.zeros(temp_img1.shape)
        ratio = get_noise_ratio(temp_img1[r:rows-r,r:cols-r])
        # print(f'noise ratio at iteration {k} = {ratio}')
        for i in range(r,rows-r):
            for j in range(r,cols-r):
                if deltaMin==temp_img1[i,j] or deltaMax==temp_img1[i,j]:
                    window = temp_img1[i-r:i+r+1,j-1:j+r+1]
                    cMean = get_constrained_mean(window)
                    temp_img2[i,j] = cMean
                else:
                    temp_img2[i,j] = temp_img1[i,j]
        lDistance = get_manhattan_distance(temp_img1,temp_img2)
        if lDistance <=epsilon:
            break
        else:
            temp_img1 = temp_img2
        k+=1

    filteredImage = temp_img2[r:rows-r,r:cols-r].astype(imtype)
    return filteredImage
