# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2 as cv
import watershed
import numpy as np
import opencv_tools
from benchmarker import BenchMarker
import tqdm
import sys
from colorama import Fore
from prettytable import PrettyTable


# pretty prints the results in one table for all benches
def pretty_print_results(results):
    print(Fore.GREEN + 'results:' + Fore.RESET)
    print('')
    table = PrettyTable(['method', 'average', 'mean', 'mean_dev'])
    for bench in results:
        name = bench.name
        avg, mean, mean_dev = bench.get_benchmark_results()
        table.add_row([name, avg, mean, mean_dev])
    print(table)


# generates output images for this method
def generate_output_images(method, method_name, cv_images):
    print('generating output images for {}'.format(method_name))
    for i in tqdm.tqdm(range(len(cv_images)), file=sys.stdout):
        output = method(cv_images[i])
        opencv_tools.export_image(output, method_name, str(i))


# runs all tests for one method
def run_tests_for_method(method, method_name, cv_images):
    print(Fore.GREEN + 'running tests for {name}'.format(name=method_name) + Fore.RESET)
    print('')

    # run benchmark
    bench = BenchMarker(method, method_name)
    bench.run_full_benchmark(cv_images)

    print('')

    # generate output images for visualization
    generate_output_images(method, method_name, cv_images)

    print('')

    # accuracy must be implemented

    # return bench for pretty printing
    return bench


# main()
if __name__ == '__main__':
    cv_images = opencv_tools.find_opencv_images('dataset//rawimages')
    imshape = np.array(cv_images[0]).shape

    # # run multiple tests (right now only return benches)
    # test_results = \
    #     [
    #         run_tests_for_method(watershed.apply_watershed, 'watershed', cv_images),
    #         run_tests_for_method(watershed.apply_watershed, 'watershed', cv_images)
    #     ]
    #
    # pretty_print_results(test_results)

    # Example on how to apply watershed.
    # x = watershed.apply_watershed(cv_images[0])
    # opencv_tools.display_image(x)