import os
import argparse
import tensorflow as tf
from tensorflow import keras
import segmentation_models as sm

def unet(backbone, weight, encoder_weights=None, input_shape=(None, None, 1)):
    model = sm.Unet(
        backbone,
        encoder_weights=encoder_weights,
        input_shape=input_shape
    )
    
    # dice_loss = sm.losses.DiceLoss()
    # focal_loss = sm.losses.BinaryFocalLoss()
    # totol_loss = dice_loss + (1 * focal_loss)
    # metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    
    # model.compile(
    #     keras.optimizers.Adam(),
    #     loss=totol_loss,
    #     metrics=metrics,
    # )
    model.load_weights(weight)
    
    return model


def decode_image(image):
    image = tf.io.decode_png(image, 1, dtype=tf.dtypes.uint16)
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
    mask = tf.cast(decode_image(example["mask_raw"]), dtype=tf.float32)
    
    return (image, mask)

def load_dataset(filenames):
    dataset = tf.data.TFRecordDataset(
        filenames,
        compression_type="GZIP"
    )
    dataset = dataset.map(read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    n_num = 0
    for n_num, _ in enumerate(dataset):
        pass

    return dataset, n_num

def get_dataset(filenames):
    dataset, n_num = load_dataset(filenames)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset, n_num
    

def main(args):
    print(args.dataset)
    test_dataset, _ = get_dataset(
        tf.io.gfile.glob(os.path.join(args.dataset, "*.tfrecord"))
    )
    model = unet(args.backbone, args.weight)
    
    for idx, (image, mask) in enumerate(test_dataset):
        image = image.numpy()
        print(image.shape, image.dtype)
        pr_mask = model.predict(image)
        # print(pr_mask.shape)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone",
        default="resnet101",
        help="Model Backbone"
    )
    parser.add_argument(
        "--weight",
        default=os.path.join(os.getcwd(), "log/ICH_420/ICH-01234567.h5"),
        help="/path/to/weight"
    )
    parser.add_argument(
        "--dataset",
        default=os.path.join(os.getcwd(), "datasets/ICH_420/TFRecords/test"),
        help="/path/to/dataset"
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.getcwd(), "output"),
        help="/path/to/dataset"
    )
    main(parser.parse_args([
        "--backbone", "resnet101",
        "--weight", os.path.join(os.getcwd(), "/workspaces/Intracranial-Hemorrhage/ICH-Segmentation/logs/ICH420-20221107165659/ICH-ICH420-31.h5"),
        "--dataset", os.path.join(os.getcwd(), "/workspaces/Intracranial-Hemorrhage/ICH-Segmentation/datasets/ICH_420/TFRecords/val"),
        "--output", os.path.join(os.getcwd(), "output")
    ]))