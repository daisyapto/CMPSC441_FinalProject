# Data collection using kagglehub library
# Images of xrays of brain tumors vs no brain tumors

import keras
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

def dataCollection():
    training_data = keras.utils.image_dataset_from_directory(
        'Add path here to main dataset folder',
        shuffle=False,
        labels='inferred',
        subset='training'
        )
    validation_data = keras.utils.image_dataset_from_directory(
        'Add path here to main dataset folder',
        shuffle=False,
        labels='inferred',
        subset='validation'
    )

    for image, label in training_data.take(5):
        plt.title(label)
        plt.imshow(image)
    for image, label in validation_data.take(5):
        plt.title(label)
        plt.imshow(image)
    #print(data.take(2))
    #print(data['Health'].take(1))
    #classes = data.class_names
    #print(classes[0].take(1))

# https://keras.io/api/data_loading/image/
# https://stackoverflow.com/questions/73672773/access-images-after-tf-keras-utils-image-dataset-from-directory
# https://www.kaggle.com/discussions/product-announcements/552681
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_sample_images.html#sklearn.datasets.load_sample_images
# https://www.kaggle.com/code/nireekshithkumar/notebook7b897cb4ce
# https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset/data
# Google search AI Overview

dataCollection()