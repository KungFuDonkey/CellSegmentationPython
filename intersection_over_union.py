import numpy as np

import statistics


def calculate_iou(result_image, ground_truth):
    intersection = np.logical_and(result_image, ground_truth)
    union = np.logical_or(result_image, ground_truth)
    intersection_num = np.sum(intersection)
    union_num = np.sum(union)
    return intersection_num / union_num

def calculate(result_images, ground_truths):
    values = []
    for i in range(len(result_images)):
        values.append(calculate_iou(result_images[i], ground_truths[i]))

    return statistics.get_stats(values)
