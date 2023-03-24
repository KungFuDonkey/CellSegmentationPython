# This is a sample Python script.
import copy
import os.path
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
import segmentation_models as sm
import tensorflow as tf
from auto_segmentation import NetworksTrainerLoader
from json_helper import JsonHelper
from keras.optimizers import Adam

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


def make_models_params():
    baseline_params = {"model_name" : "Baseline_model",
                    "backbone" : "resnet34",
                    "input_shape" : (224, 224,3),
                    "output_shape" : (224, 224,1),
                    "train_epochs" : 30,
                    "batch_size" : 10,
                    "optimizer" : Adam, #has base learning rate of 0.001
                    "fine_tune_learning_rate" : 1e-5,
                    "loss" : sm.losses.bce_jaccard_loss,
                    "metrics" : [sm.metrics.iou_score],
                    "fine_tune_epochs" : 10,
                    "early_stopping" : tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                    patience=8,
                                    verbose=1,
                                    restore_best_weights=True)}
    focal_dice_params = copy.deepcopy(baseline_params)
    focal_dice_params["model_name"] = "Focal_dice_model"
    focal_dice_params["loss"] = sm.losses.binary_focal_dice_loss
    models_params = [baseline_params,focal_dice_params]
    return models_params

# main()
if __name__ == '__main__':
    # TODO:
    # Data goed splitten en augmentations toepassen
    # verschillende unet aanpakken: met/zonder fine tuning, loss functions vergelijken, andere encoders. Alles vergelijken met base model, evt combinatie van beste dingen ook nog vergelijken.
    # De gekozen Unet implementaties uitschrijven in het verslag

    JH = JsonHelper()
    models_params = make_models_params()
    JH.pickle_object("./saved_networks/latest_models_params", models_params)

    cv_images = opencv_tools.find_opencv_images('dataset//rawimages')
    ground_truths = opencv_tools.make_binary_images(opencv_tools.find_opencv_images('dataset//groundtruth_png'))
    # test_images = []
    # test_ground_truth = []
    # for index in RANDOM_TEST_INDEXES:
    #     test_images.append(cv_images[index])
    #     test_ground_truth.append(ground_truths[index])

    X_train = cv_images[:60]
    Y_train = ground_truths[:60]
    X_valid = cv_images[60:]
    Y_valid = ground_truths[60:]

    NTL = NetworksTrainerLoader(models_params)
    #NTL.train_models(range(len(models_params)), X_train, Y_train, X_valid, Y_valid)
    #NTL.train_models([0], X_train, Y_train, X_valid, Y_valid)
    #NTL.plot_histories()

    NTL.load_models()
    Y_pred = NTL.apply_network(0, X_valid[0])
    opencv_tools.display_image(Y_valid[0])
    opencv_tools.display_image(Y_pred)
    print("done")

    # # run multiple tests (right now only return benches)
    # test_results = \
    #     [
    #         run_tests_for_method(watershed.apply_watershed, 'watershed', cv_images, ground_truths),
    #         run_tests_for_method(watershed.apply_watershed, 'watershed', cv_images, ground_truths)
    #     ]
    # pretty_print_results(test_results)
