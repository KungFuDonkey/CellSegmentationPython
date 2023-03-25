import segmentation_models as sm
import tensorflow as tf
import opencv_tools
import numpy as np
from matplotlib import pyplot as plt
from json_helper import JsonHelper
from skimage.transform import resize


class NetworksTrainerLoader:
    def __init__(self, models_params):
        self.JH = JsonHelper()
        self.histories = []
        self.fine_tune_histories = []
        self.best_epochs = []
        self.trained_models = []
        self.models_params = models_params
        for i in range(len(models_params)):
            self.models_params[i]["model_path"] = "./saved_networks/" + models_params[i]["model_name"] + "/model"
            self.models_params[i]["history_path"] = "./saved_networks/" + models_params[i]["model_name"] + "/history"
            self.models_params[i]["checkpoint"] = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.models_params[i]["model_path"], verbose=1, save_weights_only=True,
                save_best_only=True)
            self.models_params[i]["epochs_path"] = "./saved_networks/" + models_params[i]["model_name"] + "/epochs"
            self.models_params[i]["tensor_board"] = tf.keras.callbacks.TensorBoard(
                log_dir="./saved_networks/" + models_params[i]["model_name"] + "/tensor_board_logs",
                write_images=True, write_steps_per_second=True)

            """
            After training, you can visualize the summary data in TensorBoard by running the following command in your terminal or command prompt:
            tensorboard --logdir my_logs
            This will start a local web server that you can access by opening your web browser and navigating to http://localhost:6006. From here, you can view the training progress, including metrics such as the loss and accuracy, and visualizations such as histograms of weights and biases.
            """

    def __compile_models(self, model_numbers):
        compiled_models = []
        for model_num in model_numbers:
            model = sm.Unet(backbone_name=self.models_params[model_num]["backbone"],
                            input_shape=self.models_params[model_num]["input_shape"], classes=1, activation='sigmoid',
                            encoder_weights="imagenet", encoder_freeze=True)
            model.compile(
                optimizer=self.models_params[model_num]["optimizer"](),
                loss=self.models_params[model_num]["loss"],
                metrics=self.models_params[model_num]["metrics"]
            )
            compiled_models.append((model, model_num))
        return compiled_models

    def __resize_images(self, images, shape):
        return np.float32([resize(im, shape, order=0, preserve_range=True, anti_aliasing=False) for im in images])

    def __preprocess_data(self, model_num, data, is_label):
        if not is_label:
            data = self.__resize_images(data, self.models_params[model_num]["input_shape"])
            backbone_preprocessing = sm.get_preprocessing(self.models_params[model_num]["backbone"])
            data = backbone_preprocessing(data)
        else:
            data = self.__resize_images(data, self.models_params[model_num]["output_shape"])
        return data

    def __preprocess_all(self, model_num, X_train, Y_train, X_valid, Y_valid):
        X_train_ = self.__preprocess_data(model_num, X_train, is_label=False)
        Y_train_ = self.__preprocess_data(model_num, Y_train, is_label=True)
        X_valid_ = self.__preprocess_data(model_num, X_valid, is_label=False)
        Y_valid_ = self.__preprocess_data(model_num, Y_valid, is_label=True)
        return X_train_, Y_train_, X_valid_, Y_valid_

    def __train_model(self, model, model_num, X_train, Y_train, X_valid, Y_valid, fine_tune=False, initial_epoch=0):
        if not fine_tune:
            print("\n Start training of model :", model_num, "with name: ", self.models_params[model_num]["model_name"])
            epochs = self.models_params[model_num]["train_epochs"]
        else:
            print("\n Start fine tuning of model :", model_num, "with name: ",
                  self.models_params[model_num]["model_name"])
            epochs = initial_epoch + self.models_params[model_num]["fine_tune_epochs"]
        X_train_, Y_train_, X_valid_, Y_valid_ = self.__preprocess_all(model_num, X_train, Y_train, X_valid, Y_valid)
        history = model.fit(X_train_, Y_train_, epochs=epochs,
                            batch_size=self.models_params[model_num]["batch_size"],
                            validation_data=(X_valid_, Y_valid_), verbose=1,
                            callbacks=[self.models_params[model_num]["checkpoint"],
                                       self.models_params[model_num]["early_stopping"],
                                       self.models_params[model_num]["tensor_board"]],
                            initial_epoch=initial_epoch)
        return history

    def __compile_for_fine_tuning(self, model, model_num):
        model.trainable = True
        optimizer = self.models_params[model_num]["optimizer"]()
        optimizer.lr = self.models_params[model_num]["fine_tune_learning_rate"]
        model.compile(
            optimizer=optimizer,
            loss=self.models_params[model_num]["loss"],
            metrics=self.models_params[model_num]["metrics"]
        )

    def train_models(self, model_numbers, X_train, Y_train, X_valid, Y_valid):
        compiled_models = self.__compile_models(model_numbers)
        fine_tune_history = {}
        for (model, model_num) in compiled_models:
            history = self.__train_model(model, model_num, X_train, Y_train, X_valid, Y_valid)
            best_epoch = self.models_params[model_num]["early_stopping"].best_epoch
            if self.models_params[model_num]["fine_tune_epochs"] > 0:
                self.__compile_for_fine_tuning(model, model_num)
                ft_history = self.__train_model(model, model_num, X_train, Y_train, X_valid, Y_valid, fine_tune=True,
                                                initial_epoch=best_epoch + 1)
                fine_tune_history = ft_history.history
            self.JH.pickle_object(self.models_params[model_num]["history_path"], (history.history, fine_tune_history))
            self.JH.pickle_object(self.models_params[model_num]["epochs_path"], best_epoch)

    def load_models(self):
        # Loads all models
        compiled_models = self.__compile_models(range(len(self.models_params)))
        for (model, model_num) in compiled_models:
            model.load_weights(self.models_params[model_num]["model_path"]).expect_partial()
        self.trained_models = compiled_models

    def apply_network(self, model_num, image):
        output_shape = image.shape[0:2]
        if not self.trained_models:
            print("Starting with loading the models")
            self.load_models()
        image = self.__preprocess_data(model_num, [image], is_label=False)
        prediction = self.trained_models[model_num][0](image)
        prediction = (np.squeeze(prediction) > 0.5).astype(int)
        return resize(prediction, output_shape, order=0, preserve_range=True, anti_aliasing=False)

    def __load_histories(self):
        for model_num in range(len(self.models_params)):
            histories = self.JH.load_pickle_object(self.models_params[model_num]["history_path"])
            self.histories.append(histories[0])
            self.fine_tune_histories.append(histories[1])
            self.best_epochs.append(self.JH.load_pickle_object(self.models_params[model_num]["epochs_path"]))

        # Merging the two dictionaries:
        # history_up_to_best = {key: history.history[key][:best_epoch + 1] for key in history.history.keys()}
        # for key in history_up_to_best.keys():
        #     combined_history[key] = history_up_to_best.get(key, []) + fine_tune_history.get(key, [])

    def plot_histories(self):
        self.__load_histories()
        for model_num in range(len(self.histories)):
            plt.figure()
            plt.plot(self.histories[model_num]["iou_score"], "b", label="training iou_score")
            plt.plot(self.histories[model_num]["val_iou_score"], "r", label="validation iou_score")
            plt.axvline(self.best_epochs[model_num], 0, 1, color="g", label="early stopping")
            plt.legend()
            plt.xlabel("epochs")
            plt.ylabel("iou_score")
            plt.title(self.models_params[model_num]["model_name"])
            plt.show()

            plt.figure()
            plt.plot(self.histories[model_num]["loss"], "b", label="training loss")
            plt.plot(self.histories[model_num]["val_loss"], "r", label="validation loss")
            plt.axvline(self.best_epochs[model_num], 0, 1, color="g", label="early stopping")
            plt.legend()
            plt.xlabel("epochs")
            plt.ylabel("accuracy")
            plt.title(self.models_params[model_num]["model_name"])
            plt.show()

# https://segmentation-models.readthedocs.io/en/latest/tutorial.html
# https://github.com/qubvel/segmentation_models/blob/master/README.rst
# https://github.com/qubvel/segmentation_models/blob/master/examples/binary%20segmentation%20(camvid).ipynb
# https://segmentation-models.readthedocs.io/en/latest/api.html
