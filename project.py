# %% [markdown]
# Student IDs: 1043258, XXXXXX
# Student Names: Jun Li Chen, XXX

# %%
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm.auto import *
from random import randint
from multiprocessing import Process

# %%


def sum_square_diff(val1, val2):
    if val1.shape != val2.shape:
        return 0
    return np.sum(np.power(np.diff(val1 - val2), 2))


def sum_abs_diff(val1, val2):
    if val1.shape != val2.shape:
        return 0
    return np.sum(np.absolute(val1 - val2))


def norm_cross_cor(val1, val2):
    if val1.shape != val2.shape:
        return 0
    denominator = np.multiply(
        np.sqrt(np.sum(np.power(val1, 2))), np.sqrt(np.sum(np.power(val2, 2))))
    if (denominator == 0):
        return 0
    return np.divide(np.sum(np.multiply(val1, val2)), denominator)


def zero_mean_sum_abs_diff(val1, val2):
    if val1.shape != val2.shape:
        return 0
    avg_val1 = np.mean(val1)
    avg_val2 = np.mean(val2)
    return np.sum(np.absolute(val1 - avg_val1 - val2 + avg_val2))

# %%


def compare_blocks(y, x, l_block, r_img, block_size, search_size, func):
    x_min = max(0, x - search_size)
    x_max = min(r_img.shape[1], x + search_size)
    first = True
    min_val = None
    min_idx = None
    for x in range(x_min, x_max):
        r_block = r_img[y: y+block_size, x: x+block_size]
        val = func(l_block, r_block)
        if first:
            min_val = val
            min_idx = (y, x)
            first = False
        else:
            if val <= min_val:
                min_val = val
                min_idx = (y, x)
    return min_idx
# %%


BLOCK_SIZE = 3
SEARCH_SIZE = 12


def calc(h, w, l_img, r_img, func, compare_func, pos):
    disp_map = np.zeros((h, w))
    for y in tqdm(range(BLOCK_SIZE, h-BLOCK_SIZE), position=pos, desc=func.__name__, leave=True):
        for x in range(BLOCK_SIZE, w-BLOCK_SIZE):
            l_block = l_img[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]
            min_index = compare_func(
                y, x, l_block, r_img, BLOCK_SIZE, SEARCH_SIZE, func)
            disp_map[y, x] = abs(min_index[1] - x)
    disp_map = (disp_map - disp_map.min()) / (disp_map.max() - disp_map.min())
    plt.axis(False)
    plt.imshow(disp_map)
    plt.savefig("./image/" + func.__name__ + ".png")


# %%
if __name__ == '__main__':
    path = "Dataset\\"
    files = os.listdir(path)
    files = [files[i:i + 3] for i in range(0, len(files), 3)]
    img = randint(0, len(files)-1)
    a_img = cv2.imread(path + files[img][0])
    l_img = cv2.imread(path + files[img][1])
    r_img = cv2.imread(path + files[img][2])
    try:
        os.remove("./image/groundtruth.png")
        os.remove("./image/disparitySGBM.png")
        os.remove("./image/sum_square_diff.png")
        os.remove("./image/zero_mean_sum_abs_diff.png")
    except OSError:
        pass

    plt.axis(False)
    plt.imshow(a_img)
    plt.savefig("./image/groundtruth.png")
    print(files[img][0])
    print(l_img.shape, r_img.shape)
    h, w, ch = l_img.shape

    stereoSGBM = cv2.StereoSGBM_create(
        numDisparities=SEARCH_SIZE, blockSize=BLOCK_SIZE)
    disparitySGBM = stereoSGBM.compute(l_img, r_img)
    disparitySGBM_min = disparitySGBM.min()
    disparitySGBM_max = disparitySGBM.max()
    disparitySGBM = ((disparitySGBM - disparitySGBM_min) /
                     (disparitySGBM_max - disparitySGBM_min)*255).astype(np.uint8)
    plt.axis(False)
    plt.imshow(disparitySGBM)
    plt.savefig("disparitySGBM.png")

    workers = [Process(target=calc, args=(
        h, w, l_img, r_img, sum_square_diff, compare_blocks, 0,)),
        Process(target=calc, args=(h, w, l_img, r_img,
                                   zero_mean_sum_abs_diff, compare_blocks, 1))]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
