# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2 as cv
import watershed
import opencv_tools
import benchmarker

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cv_images = opencv_tools.find_opencv_images('inputImages')

    watershed_export_func = lambda image, iterator: opencv_tools.export_image(image, 'watershed', str(iterator))
    watershed_bench = benchmarker.BenchMarker(watershed.apply_watershed, 'watershed', watershed_export_func)

    watershed_bench.run_full_benchmark(cv_images)

    # needs a nice pretty print function for printing all the results
    watershed_avg, watershed_mean, watershed_mean_dev = watershed_bench.get_benchmark_results()

    print('avg: {avg}, mean: {mean}, mean_dev: {mean_dev}'.format(avg=watershed_avg, mean=watershed_mean,
                                                                  mean_dev=watershed_mean_dev))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
