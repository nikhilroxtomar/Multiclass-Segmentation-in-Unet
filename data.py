
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2

H = 256
W = 256

def process_data(data_path, file_path):
    df = pd.read_csv(file_path, sep=" ", header=None)
    names = df[0].values

    images = [os.path.join(data_path, f"images/{name}.jpg") for name in names]
    masks = [os.path.join(data_path, f"annotations/trimaps/{name}.png") for name in names]

    return images, masks

def load_data(path):
    train_valid_path = os.path.join(path, "annotations/trainval.txt")
    test_path = os.path.join(path, "annotations/test.txt")

    train_x, train_y = process_data(path, train_valid_path)
    test_x, test_y = process_data(path, test_path)

    train_x, valid_x = train_test_split(train_x, test_size=0.2, random_state=42)
    train_y, valid_y = train_test_split(train_y, test_size=0.2, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(x):
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    return x

def read_mask(x):
    x = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))
    x = x - 1
    x = x.astype(np.int32)
    return x

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(2)
    return dataset

def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()

        image = read_image(x)
        mask = read_mask(y)

        return image, mask

    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.int32])
    mask = tf.one_hot(mask, 3, dtype=tf.int32)
    image.set_shape([H, W, 3])
    mask.set_shape([H, W, 3])

    return image, mask


if __name__ == "__main__":
    path = "oxford-iiit-pet/"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)
    print(f"Dataset: Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")

    dataset = tf_dataset(train_x, train_y, batch=8)
    for x, y in dataset:
        print(x.shape, y.shape) ## (8, 256, 256, 3), (8, 256, 256, 3)
