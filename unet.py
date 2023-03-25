from json_helper import JsonHelper
from keras.optimizers import Adam
import segmentation_models as sm
import tensorflow as tf
import copy
from auto_segmentation import NetworksTrainerLoader
import os


def make_models_params():
    baseline_params = {"model_name": "Baseline_model",
                       "backbone": 'efficientnetb0',
                       "input_shape": (224, 224, 3),
                       "output_shape": (224, 224, 1),
                       "train_epochs": 30,
                       "batch_size": 10,
                       "optimizer": Adam,  # has base learning rate of 0.001
                       "fine_tune_learning_rate": 1e-5,
                       "loss": sm.losses.bce_jaccard_loss,
                       "metrics": [sm.metrics.iou_score],
                       "fine_tune_epochs": 10,
                       "early_stopping": tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                          patience=10,
                                                                          verbose=1,
                                                                          restore_best_weights=True)}
    efficientb3_params = copy.deepcopy(baseline_params)
    efficientb3_params["model_name"] = "Efficientb3_model"
    efficientb3_params["backbone"] = 'efficientnetb3'

    efficientb7_params = copy.deepcopy(baseline_params)
    efficientb7_params["model_name"] = "Efficientb7_model"
    efficientb7_params["backbone"] = 'efficientnetb7'

    baseline512x640_params = copy.deepcopy(baseline_params)
    baseline512x640_params["model_name"] = "Baseline512x640_model"
    baseline512x640_params["input_shape"] = (512, 640, 3)
    baseline512x640_params["output_shape"] = (512, 640, 1)

    baseline1024x1280_params = copy.deepcopy(baseline_params)
    baseline1024x1280_params["model_name"] = "Baseline1024x1280_model"
    baseline1024x1280_params["input_shape"] = (1024, 1280, 3)
    baseline1024x1280_params["output_shape"] = (1024, 1280, 1)

    focal_dice_params = copy.deepcopy(baseline_params)
    focal_dice_params["model_name"] = "Focal_dice_model"
    focal_dice_params["loss"] = sm.losses.binary_focal_dice_loss

    mobilenet_params = copy.deepcopy(baseline_params)
    mobilenet_params["model_name"] = "MobilenetV2_model"
    mobilenet_params["backbone"] = "mobilenetv2"

    models_params = [baseline_params, baseline512x640_params, baseline1024x1280_params, efficientb3_params, efficientb7_params, focal_dice_params,
                     mobilenet_params]

    return models_params


def create_models():
    models_params = make_models_params()
    return NetworksTrainerLoader(models_params)


def train_models(NTL, train_images, train_ground_truth, validation_images, validation_ground_truth):
    if os.path.isdir('./saved_networks'):
        for i in range(len(NTL.models_params)):
            if os.path.isdir(os.path.join('./saved_networks', NTL.models_params[i]["model_name"])):
                continue
            NTL.train_models([i], train_images, train_ground_truth, validation_images, validation_ground_truth)
        NTL.load_models()
        return

    os.mkdir('./saved_networks')
    NTL.train_models(range(len(NTL.models_params)), train_images, train_ground_truth, validation_images,
                     validation_ground_truth)
    NTL.load_models()


def predict(NTL, model_index, image):
    return NTL.apply_network(model_index, image)
