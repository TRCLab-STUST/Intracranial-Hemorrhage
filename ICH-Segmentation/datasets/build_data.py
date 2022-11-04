import tensorflow as tf


def _bytes_feature(values):
    """Returns a bytes_list from a string / byte."""

    def norm2bytes(value):
        return value.encode() if isinstance(value, str) else value

    if isinstance(values, type(tf.constant(0))):
        values = values.numpy()  # BytesList won't unpack a string from an EagerTensor.

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[norm2bytes(values)]))


def _int64_feature(values):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))


def image_seg_to_tfexample(image_filename, slice_number, sample, image_raw, mask_raw):
    return tf.train.Example(features=tf.train.Features(feature={
        "nii/filename": _bytes_feature(image_filename),
        "slice/number": _int64_feature(slice_number),
        "slice/sample": _int64_feature(bool(sample)),
        "image/raw": _bytes_feature(image_raw),
        "mask/raw": _bytes_feature(mask_raw),
    }))
