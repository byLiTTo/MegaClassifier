import os

import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.src.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, auc, confusion_matrix
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Precision, Recall, AUC

IMAGE_SIZE = (224, 224)
IMAGE_SHAPE = IMAGE_SIZE + (3,)
SEED = 42


def image_data_generator_v1():
    train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)
    datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)

    return train_datagen, datagen


def image_data_generator_v2():
    train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)
    datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)

    return train_datagen, datagen


def image_data_generator(version: str):
    if version == "v1":
        return image_data_generator_v1()
    elif version == "v2":
        return image_data_generator_v2()
    else:
        return None


def flow_from_dataframe(datasets, generators, batch_size):
    train_images = generators[0].flow_from_dataframe(
        dataframe=datasets[0],
        x_col="file_name",
        y_col="binary_label",
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
        seed=SEED
    )

    validation_images = generators[1].flow_from_dataframe(
        dataframe=datasets[1],
        x_col="file_name",
        y_col="binary_label",
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
        seed=SEED
    )

    test_images = generators[1].flow_from_dataframe(
        dataframe=datasets[2],
        x_col="file_name",
        y_col="binary_label",
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
        seed=SEED
    )

    return train_images, validation_images, test_images


def load_pretrained_v1():
    mobilenet_v2 = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=IMAGE_SHAPE,
    )
    mobilenet_v2.trainable = False

    return mobilenet_v2


def load_pretrained_v2():
    mobilenet_v2 = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=IMAGE_SHAPE,
    )
    mobilenet_v2.trainable = False

    return mobilenet_v2


def load_pretrained(version: str):
    if version == "v1":
        return load_pretrained_v1()
    elif version == "v2":
        return load_pretrained_v2()
    else:
        return None


def compile_model_v1(pretrained_model, name):
    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ], name=name)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy", Precision(name="precision"), Recall(name="recall"), AUC(name="auc")],
    )

    return model


def compile_model_v2(pretrained_model, loss_function, name, weights=None):
    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ], name=name)

    if weights is not None:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=loss_function,
            metrics=["accuracy", Precision(name="precision"), Recall(name="recall"), AUC(name="auc")],
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=loss_function,
            metrics=["accuracy", Precision(name="precision"), Recall(name="recall"), AUC(name="auc")],
        )

    return model


def compile_model(version: str, pretrained_model, name, loss_function=None, weights=None):
    if version == "v1":
        return compile_model_v1(
            pretrained_model=pretrained_model,
            name=name,
        )
    elif version == "v2":
        return compile_model_v2(
            pretrained_model=pretrained_model,
            loss_function=loss_function,
            weights=weights,
            name=name
        )
    else:
        return None


def callbacks_v1(logs_path):
    os.makedirs(logs_path, exist_ok=True)
    return [TensorBoard(log_dir=logs_path)]


def callbacks_v2(logs_path):
    os.makedirs(logs_path, exist_ok=True)
    return [TensorBoard(log_dir=logs_path)]


def callbacks(version, logs_path):
    if version == "v1":
        return callbacks_v1(logs_path)
    elif version == "v2":
        return callbacks_v1(logs_path)
    else:
        return None


def fit(model, images, epochs, call_backs, save_path):
    print("\n\n")
    history = model.fit(
        images[0],
        epochs=epochs,
        validation_data=images[1],
        callbacks=call_backs
    )
    os.makedirs(save_path, exist_ok=True)

    print("\n\n")
    print("Saving model...")
    model.save(save_path)

    return history


def save_training(data, save_path):
    print("\n\n")
    print("Saving training data...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data.to_csv(save_path, sep=";", index=False)


def FocalLoss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        loss = -K.mean(alpha * K.pow(1. - pt, gamma) * K.log(pt))
        return loss

    return loss


def evaluate_model(model_path, test_images, custom_loss=False):
    print("\n\n")
    if custom_loss:
        model = tf.keras.models.load_model(model_path, custom_objects={"loss": FocalLoss()})
    else:
        model = tf.keras.models.load_model(model_path)
    return model.evaluate(test_images)


def save_evaluation(data, save_path):
    print("\n\n")
    print("Saving evaluation data...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data.to_csv(save_path, sep=";", index=False)


def predict_model(model_path, test_images, custom_loss=False):
    print("\n\n")
    if custom_loss:
        model = tf.keras.models.load_model(model_path, custom_objects={"loss": FocalLoss()})
    else:
        model = tf.keras.models.load_model(model_path)
    return model.predict(test_images)


def roc_curve_model(model_path, test_images, custom_loss=False):
    print("\n\n")
    y_true = test_images.classes

    if custom_loss:
        model = tf.keras.models.load_model(model_path, custom_objects={"loss": FocalLoss()})
    else:
        model = tf.keras.models.load_model(model_path)

    y_predictions_probs = model.predict(test_images).flatten()

    fpr, tpr, thresholds = roc_curve(y_true, y_predictions_probs)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, thresholds, roc_auc


def confusion_matrix_model(model_path, test_images, optimal_threshold, custom_loss=False):
    print("\n\n")
    y_true = test_images.classes

    if custom_loss:
        model = tf.keras.models.load_model(model_path, custom_objects={"loss": FocalLoss()})
    else:
        model = tf.keras.models.load_model(model_path)
    y_predictions_probs = model.predict(test_images)
    y_predictions = (y_predictions_probs >= optimal_threshold).astype(int).flatten()

    return confusion_matrix(y_true, y_predictions)
