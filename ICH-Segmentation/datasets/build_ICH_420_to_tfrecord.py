import albumentations as A
import argparse
import cv2
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm

import build_data
import mount_dataset
from NII_Data import NIIData

DATASET_DIR = "ICH_420"
OUTPUT_DIR = tf.io.gfile.join(DATASET_DIR, "tfrecords")


def get_paired_dataset(image_patten, annotation_patten):
    images = tf.io.gfile.glob(image_patten)
    annotations = tf.io.gfile.glob(annotation_patten)

    images.sort()
    annotations.sort()

    return list(zip(images, annotations))


def extract_head_image_and_center_point(images, bit=12):
    result = []
    points = []
    for origin_img in images:
        _, binary_img = cv2.threshold(
            np.asarray(origin_img),
            1,
            2 ** bit - 1,
            cv2.THRESH_BINARY
        )
        contours, _ = cv2.findContours(
            binary_img.astype(dtype=jnp.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )
        mask = np.zeros(origin_img.shape, dtype=np.uint16)
        if len(contours) > 0:
            max_contours_id = np.argmax(
                [cv2.contourArea(contours[k]) for k in range(len(contours))]
            )

            cv2.drawContours(mask, contours, max_contours_id, 1, cv2.FILLED)

            _m = cv2.moments(contours[max_contours_id])
            c_x = int(_m["m10"] / _m["m00"])
            c_y = int(_m["m01"] / _m["m00"])
        else:
            c_x = c_y = 0

        extract_image = origin_img * mask
        result.append(extract_image)
        points.append((c_x, c_y))
    return jnp.asarray(result), points


def crop_and_padding(images, masks, c_points, scale=512):
    assert len(images) == len(masks) == len(c_points), "Shape not match!"

    result = []
    for image, mask, point in zip(images, masks, c_points):
        min_pos = jnp.asarray(point) - (scale / 2)
        max_pos = min_pos + scale
        min_pos = jnp.clip(min_pos, 0, scale).astype(dtype=jnp.uint16)
        max_pos = jnp.clip(max_pos, 0, scale).astype(dtype=jnp.uint16)
        transform = A.Compose([
            A.Crop(x_min=min_pos[0], y_min=min_pos[1], x_max=max_pos[0], y_max=max_pos[1]),
            A.PadIfNeeded(min_height=scale, min_width=scale, border_mode=cv2.BORDER_CONSTANT, value=0)
        ])

        transformed = transform(image=np.asarray(image), mask=np.asarray(mask))
        result.append([transformed["image"], transformed["mask"]])

    result = jnp.asarray(result)
    return result[:, 0, :, :], result[:, 1, :, :]


def normalize(paired_slices):
    images = paired_slices[0].T
    masks = paired_slices[1].T
    images, c_points = extract_head_image_and_center_point(images)
    images, masks = crop_and_padding(images, masks, c_points)
    return jnp.asarray((images, masks))


def _convert_dataset(paired_file, file_id):
    filename = os.path.basename(paired_file[0])
    nii_data = NIIData(paired_file, (0, 90))
    paired_slices = normalize(nii_data.extract_paired_slices())
    with tf.io.TFRecordWriter(
            tf.io.gfile.join(
                OUTPUT_DIR,
                f"ich-{str(file_id).zfill(4)}.tfrecord"
            )
    ) as writer:
        for idx in range(paired_slices.shape[1]):
            image = paired_slices[0, idx, :, :]
            mask = paired_slices[1, idx, :, :]
            example = build_data.image_seg_to_tfexample(
                image_filename=filename,
                slice_number=idx,
                sample=mask.any(),
                image_raw=tf.io.serialize_tensor(image),
                mask_raw=tf.io.serialize_tensor(mask)
            )
            writer.write(example.SerializeToString())


def main(args):
    paired_dataset = get_paired_dataset(args.images, args.annotations)
    for idx, paired_file in tqdm(enumerate(paired_dataset), desc="Paired"):
        _convert_dataset(paired_file, idx)


if __name__ == '__main__':
    mount_dataset.mount_ich_420_dataset(DATASET_DIR)
    parser = argparse.ArgumentParser()
    parser.add_argument("--images",
                        default="ICH_420/Images/*.nii.gz",
                        help="Dataset images file_template"
                        )
    parser.add_argument("--annotations",
                        default="ICH_420/Labels/*_label.nii.gz",
                        help="Dataset Annotations file_template"
                        )
    parser.add_argument("--output",
                        default="ICH_420/tfrecords",
                        help="tfrecord output folder"
                        )
    main(parser.parse_args())
    mount_dataset.umount(DATASET_DIR)
