import os
import random
import argparse
import numpy as np
import tensorflow as tf
import segmentation_models as sm


def split_train_test(tfrecords_pattern, rate=0.8, buffer_size=10000):
    filenames = tf.io.gfile.glob(tfrecords_pattern)
    random.shuffle(filenames)
    split_idx = int(len(filenames) * rate)
    

    return filenames[:split_idx], filenames[split_idx:]

def decode_image(image, bit=12):
    image = tf.io.decode_png(image, 1, dtype=tf.dtypes.uint16)
    image = tf.expand_dims(image, axis=-1)
    image = tf.cast(image, tf.dtypes.float32) / (2 ** bit)
    
    return image

def read_tfrecord(example):
    features_description = {
        "filename": tf.io.FixedLenFeature([], tf.string),
        "number": tf.io.FixedLenFeature([], tf.int64),
        "sample": tf.io.FixedLenFeature([], tf.int64),
        "image_raw": tf.io.FixedLenFeature([], tf.string),
        "mask_raw": tf.io.FixedLenFeature([], tf.string)
    }
    
    example = tf.io.parse_single_example(example, features_description, name="nii")
    image = decode_image(example["image_raw"])
    mask = decode_image(example["mask_raw"])

    return image, mask

def load_dataset(filenames):
    dataset = tf.data.TFRecordDataset(
        filenames,
        compression_type="GZIP"
    )
    dataset = dataset.shuffle(2048, reshuffle_each_iteration=False)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset

def get_dataset(filenames, batch=4, repeat=False):
    dataset = load_dataset(filenames)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch)
    if repeat:
        dataset = dataset.repeat()
    
    return dataset


def main(args):
    # Load Dataset
    x_list, y_list = split_train_test(
        os.path.join(args.dataset, "*.tfrecord"),
        rate=args.traing_rate
    )
    print(f"Train: {len(x_list)}")
    print(f"Test: {len(y_list)}")
    
    x_dataset = get_dataset(x_list, batch=args.batch, repeat=True)
    y_dataset = get_dataset(y_list, batch=args.batch)
    
    # Build Model
    preprocess_input = sm.get_preprocessing(args.backbone)
    x_dataset = preprocess_input(x_dataset)
    y_dataset = preprocess_input(y_dataset)
    model = sm.Unet(args.backbone, encoder_weights=None, input_shape=(None, None, 1))

    model.compile(
        'Adam',
        loss=sm.losses.dice_loss,
        metrics=[sm.metrics.iou_score, sm.metrics.f1_score],
    )
    
    # Training
    model.fit(
        x_dataset,
        epochs=args.epoch,
        steps_per_epoch=args.steps,
        validation_data=y_dataset,
        verbose=args.verbose
    )
    
    print("fitted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone",
        default="resnet101",
        help="Model Backbone"
    )
    parser.add_argument(
        "--batch",
        default=8,
        help="Batch Size",
        type=int
    )
    parser.add_argument(
        "--epoch",
        default=10,
        help="Training Epoch",
        type=int
    )
    parser.add_argument(
        "--steps",
        default=1000,
        help="Steps per Epoch",
        type=int
    )
    parser.add_argument(
        "--dataset",
        default="/ich/ICH-Segmentation/datasets/ICH_420/TFRecords/train",
        help="/path/to/dataset"
    )
    parser.add_argument(
        "--traing_rate",
        default=0.8,
        help="Use to split dataset to 'train' and 'valid'",
        type=float
    )
    parser.add_argument(
        "--verbose",
        default=1,
        help="verbose",
        type=int
    )
    main(parser.parse_args())
