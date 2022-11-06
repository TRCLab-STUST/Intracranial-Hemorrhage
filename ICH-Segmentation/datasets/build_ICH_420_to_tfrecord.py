import albumentations as A
import argparse
import cv2
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import math
from tqdm import tqdm, trange

import build_data
from NII_Data import NIIData


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
        
    return jnp.asarray(result, dtype=jnp.uint16), points


def crop_and_padding(images, masks, c_points, scale=512):
    assert len(images) == len(masks) == len(c_points), "Shape not match!"

    result = []
    for image, mask, point in zip(images, masks, c_points):
        min_pos = jnp.asarray(point) - (scale / 2)
        max_pos = min_pos + scale
        min_pos = jnp.clip(min_pos, 0, scale).astype(dtype=jnp.uint16)
        max_pos = jnp.clip(max_pos, 0, scale).astype(dtype=jnp.uint16)
        transform = A.Compose([
            A.Crop(x_min=min_pos[0], y_min=min_pos[1],
                   x_max=max_pos[0], y_max=max_pos[1]),
            A.PadIfNeeded(min_height=scale, min_width=scale,
                          border_mode=cv2.BORDER_CONSTANT, value=0)
        ])
        
        transformed = transform(image=np.asarray(image), mask=np.asarray(mask))
        result.append([transformed["image"], transformed["mask"]])
    
    result = jnp.asarray(result, dtype=jnp.uint16)
    return result[:, 0, :, :], result[:, 1, :, :]


def normalize(paired_slices):
    images, masks = paired_slices[0].T, paired_slices[1].T 
    images, c_points = extract_head_image_and_center_point(images)    
    images, masks = crop_and_padding(images, masks, c_points)    
    return jnp.asarray((images, masks))



_NUM_SHARDS = 4
def _convert_dataset(paired_filepaths, split_name, output_dir):
    num_paired_filepath = len(paired_filepaths)
    num_per_shard = int(math.ceil(num_paired_filepath / _NUM_SHARDS))
    
    for shard_id in trange(_NUM_SHARDS, position=1, desc=f"Shards-{split_name}"):
        output_filepath = os.path.join(
            output_dir,
            f"{split_name}-{shard_id}-of-{_NUM_SHARDS}.tfrecord"
        )
        with tf.io.TFRecordWriter(output_filepath,
                                  options=tf.io.TFRecordOptions(
                                      compression_type="GZIP",
                                      compression_level=9
                                  )) as writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_paired_filepath)
            for idx in trange(start_idx, end_idx, position=2, desc="Nii"):
                paired_file = paired_filepaths[idx]
                filename_encode = os.path.basename(paired_file[0]).encode()
                nii_data = NIIData(paired_file, (0, 90))
                paired_slices = normalize(nii_data.extract_paired_slices())
                
                for n in trange(paired_slices.shape[1], position=3, desc="Slices"):
                    image = jnp.expand_dims(paired_slices[0, n, :, :], axis=-1)
                    mask = jnp.expand_dims(paired_slices[1, n, :, :], axis=-1)
                                       
                    example = build_data.image_seg_to_tfexample(
                        image_filename=filename_encode,
                        slice_number=n,
                        sample=mask.any(),
                        image_raw=tf.io.encode_png(image, compression=9),
                        mask_raw=tf.io.encode_png(mask, compression=9)
                    )

                    writer.write(example.SerializeToString())


def main(args):
    dataset_splits = tf.io.gfile.glob(os.path.join(args.list_folder, "*.txt"))
    for dataset_split in tqdm(dataset_splits, position=0, desc="Processing"):
        split_name = os.path.basename(dataset_split).split(".")[0]
        output_dir = os.path.join(args.output, split_name)
        os.makedirs(output_dir, exist_ok=True)

        paired_filepaths = [x.strip("\n").split(",") for x in open(
            dataset_split, "r", encoding="utf-8")]
        paired_filepaths = [(os.path.join(args.images, paired_filename[0]), os.path.join(
            args.annotations, paired_filename[1])) for paired_filename in paired_filepaths]

        _convert_dataset(paired_filepaths, split_name, output_dir)


if __name__ == '__main__':
    CURRENT_DIR = os.path.abspath(os.path.join(__file__, os.pardir))
    DATASET_DIR = os.path.join(CURRENT_DIR, "ICH_420")
    parser = argparse.ArgumentParser()
    parser.add_argument("--list_folder",
                        default=tf.io.gfile.join(
                            CURRENT_DIR, "ICH_420", "ImageSets", "Segmentation"),
                        help="包含訓練和驗證列表的資料夾"
                        )
    parser.add_argument("--images",
                        default=tf.io.gfile.join(
                            CURRENT_DIR, "ICH_420", "Images"),
                        help="包含影像的資料夾"
                        )
    parser.add_argument("--annotations",
                        default=tf.io.gfile.join(
                            CURRENT_DIR, "ICH_420", "Labels"),
                        help="包含語義分割標記的資料夾"
                        )
    parser.add_argument("--output",
                        default=tf.io.gfile.join(
                            CURRENT_DIR, "ICH_420", "TFRecords"),
                        help="輸出轉換為 TFRecord 的路徑"
                        )
    main(parser.parse_args())
