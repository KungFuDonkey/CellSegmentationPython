import numpy as np


def calculate(result_image, ground_truth):
    intersection = np.logical_and(result_image > 1, ground_truth > 1)
    union = np.logical_or(result_image > 1, ground_truth > 1)
    intersection_num = np.sum(intersection)
    union_num = np.sum(union)
    return intersection_num / union_num
