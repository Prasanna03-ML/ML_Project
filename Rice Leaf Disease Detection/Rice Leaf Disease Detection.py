import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers,models

Width=3081
Height=897
batch_size=35
dataset=tf.keras.preprocessing.image_dataset_from_directory(
    "Bacterial leaf blight-20200814T055237Z-001",
    shuffle=True,
    Image_size=(Width,Height),
    Batch_size=batch_size
)
class_name=dataset.class_names
class_name
len(class_name)