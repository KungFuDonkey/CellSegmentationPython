# This is a sample Python script.
import opencv_tools
import intersection_over_union
from benchmarker import BenchMarker
import tqdm
import sys
from colorama import Fore
from prettytable import PrettyTable
from auto_segmentation import NetworksTrainerLoader
import unet
import watershed


# pretty prints the results in one table for all benches
def pretty_print_results(results):
    print(Fore.GREEN + 'results:' + Fore.RESET)
    print('')
    table = PrettyTable(['method', 'time_avg', 'time_mean', 'time_mean_dev', 'iou_avg', 'iou_mean', 'iou_mean_dev'])
    for (bench_name, bench_avg, bench_mean, bench_mean_dev), (iou_avg, iou_mean, iou_mean_dev) in results:
        table.add_row([
            bench_name,
            round(bench_avg, 3),
            round(bench_mean, 3),
            round(bench_mean_dev, 3),
            round(iou_avg, 3),
            round(iou_mean, 3),
            round(iou_mean_dev, 3)])
    print(table)

    print('')

    print(Fore.GREEN + 'latex table code:' + Fore.RESET)
    for (bench_name, bench_avg, bench_mean, bench_mean_dev), (iou_avg, iou_mean, iou_mean_dev) in results:
        print(
            bench_name + ' & ' +
            str(round(bench_avg, 3)) + ' & ' +
            str(round(bench_mean, 3)) + ' & ' +
            str(round(bench_mean_dev, 3)) + ' & ' +
            str(round(iou_avg, 3)) + ' & ' +
            str(round(iou_mean, 3)) + ' & ' +
            str(round(iou_mean_dev, 3)) + ' \\\\')


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
    bench_result = bench.get_benchmark_results()

    print('')

    # generate output images for visualization
    result_images = generate_output_images(method, method_name, cv_images)

    print('')

    # create iou results
    iou = intersection_over_union.calculate(result_images, ground_truths)

    print('')

    # return bench for pretty printing
    return bench_result, iou


RANDOM_VALIDATION_INDEXES = [6, 18, 22, 39, 46, 53, 66, 77]
RANDOM_TEST_INDEXES = [3, 12, 25, 37, 43, 54, 60, 76]


# main()
if __name__ == '__main__':
    # TODO:
    # Data goed splitten en augmentations toepassen
    # verschillende unet aanpakken: met/zonder fine tuning, loss functions vergelijken, andere encoders. Alles vergelijken met base model, evt combinatie van beste dingen ook nog vergelijken.
    # De gekozen Unet implementaties uitschrijven in het verslag

    # generate data
    cv_images = opencv_tools.find_opencv_images('dataset//rawimages')
    ground_truths = opencv_tools.find_opencv_images('dataset//groundtruth_png')

    # Augment the images, turning 1 image into 8 images.
    # augmented_raw, augmented_ground = opencv_tools.augment_images(cv_images, ground_truths)
    # for img in augmented_ground:
    #     opencv_tools.display_image(img)

    ground_truths_bin = opencv_tools.make_binary_images(ground_truths)
    test_images = []
    test_ground_truth = []
    for index in RANDOM_TEST_INDEXES:
        test_images.append(cv_images[index])
        test_ground_truth.append(ground_truths_bin[index])

    validation_images = []
    validation_ground_truth = []
    for index in RANDOM_VALIDATION_INDEXES:
        validation_images.append(cv_images[index])
        validation_ground_truth.append(ground_truths_bin[index])

    train_images = []
    train_ground_truth = []
    for i in range(len(cv_images)):
        if i in RANDOM_TEST_INDEXES or i in RANDOM_VALIDATION_INDEXES:
            continue
        train_images.append(cv_images[i])
        train_ground_truth.append(ground_truths[i])

    augmented_images, augmented_ground_truths = opencv_tools.augment_images(train_images, train_ground_truth)

    # create and train unet models
    NTL = unet.create_models()
    unet.train_models(NTL, augmented_images, opencv_tools.make_binary_images(augmented_ground_truths), validation_images, validation_ground_truth)

    # run multiple tests (right now only return benches)
    test_results = \
        [
            run_tests_for_method(lambda image : unet.predict(NTL, 0, image), NTL.models_params[0]["model_name"], test_images, test_ground_truth),
            run_tests_for_method(lambda image : unet.predict(NTL, 1, image), NTL.models_params[1]["model_name"], test_images, test_ground_truth),
            run_tests_for_method(lambda image : unet.predict(NTL, 2, image), NTL.models_params[2]["model_name"], test_images, test_ground_truth),
            run_tests_for_method(lambda image : unet.predict(NTL, 3, image), NTL.models_params[3]["model_name"], test_images, test_ground_truth),
            run_tests_for_method(lambda image : unet.predict(NTL, 4, image), NTL.models_params[4]["model_name"], test_images, test_ground_truth),
            run_tests_for_method(lambda image : unet.predict(NTL, 5, image), NTL.models_params[5]["model_name"], test_images, test_ground_truth),
            run_tests_for_method(lambda image : unet.predict(NTL, 6, image), NTL.models_params[6]["model_name"], test_images, test_ground_truth),
            run_tests_for_method(watershed.apply_watershed, 'Watershed', test_images, test_ground_truth),
            run_tests_for_method(watershed.apply_watershed, 'Watershed full', cv_images, ground_truths)
        ]
    pretty_print_results(test_results)
