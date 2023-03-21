# This is a sample Python script.
import os.path

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2 as cv
import watershed
import numpy as np
import opencv_tools
import intersection_over_union
from benchmarker import BenchMarker
import tqdm
import sys
from colorama import Fore
from prettytable import PrettyTable


# pretty prints the results in one table for all benches
def pretty_print_results(results):
    print(Fore.GREEN + 'results:' + Fore.RESET)
    print('')
    table = PrettyTable(['method', 'time_avg', 'time_mean', 'time_mean_dev', 'iou_avg', 'iou_mean', 'iou_mean_dev'])
    for bench, (iou_avg, iou_mean, iou_mean_dev) in results:
        name = bench.name
        avg, mean, mean_dev = bench.get_benchmark_results()
        table.add_row([name, avg, mean, mean_dev, iou_avg, iou_mean, iou_mean_dev])
    print(table)


# generates output images for this method
def generate_output_images(method, method_name, cv_images):
    print('generating output images for {}'.format(method_name))
    output_images = []
    for i in tqdm.tqdm(range(len(cv_images)), file=sys.stdout):
        output = method(cv_images[i])
        opencv_tools.export_image(output, method_name, str(i))
        output_images.append(output)

    return output_images


# runs all tests for one method
def run_tests_for_method(method, method_name, cv_images, ground_truths):
    print(Fore.GREEN + 'running tests for {name}'.format(name=method_name) + Fore.RESET)
    print('')

    # run benchmark
    bench = BenchMarker(method, method_name)
    bench.run_full_benchmark(cv_images)

    print('')

    # generate output images for visualization
    result_images = generate_output_images(method, method_name, cv_images)

    print('')

    iou = intersection_over_union.calculate(result_images, ground_truths)

    # return bench for pretty printing
    return bench, iou


# main()
if __name__ == '__main__':
    cv_images = opencv_tools.find_opencv_images('dataset/rawimages')
    ground_truths = opencv_tools.make_binary_images(opencv_tools.find_opencv_images('dataset/groundtruth_png'))

    # run multiple tests (right now only return benches)
    test_results = \
        [
            run_tests_for_method(watershed.apply_watershed, 'watershed', cv_images, ground_truths),
            run_tests_for_method(watershed.apply_watershed, 'watershed', cv_images, ground_truths)
        ]
    pretty_print_results(test_results)
