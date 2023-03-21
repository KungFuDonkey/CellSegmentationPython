import sys

import numpy as np
import tqdm

import statistics


def calculate_iou(result_image, ground_truth):
    intersection = np.logical_and(result_image, ground_truth)
    union = np.logical_or(result_image, ground_truth)
    intersection_num = np.sum(intersection)
    union_num = np.sum(union)
    return intersection_num / union_num

def calculate(result_images, ground_truths):
    values = []

    print('calculating iou')
    for i in tqdm.tqdm(range(len(result_images)), file=sys.stdout):
        values.append(calculate_iou(result_images[i], ground_truths[i]))

    return statistics.get_stats(values)
