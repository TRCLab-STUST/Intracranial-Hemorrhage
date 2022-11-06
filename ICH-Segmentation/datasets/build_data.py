import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_seg_to_tfexample(image_filename, slice_number, sample, image_raw, mask_raw):
    return tf.train.Example(features=tf.train.Features(feature={
        "filename": _bytes_feature(image_filename),
        "number": _int64_feature(slice_number),
        "sample": _int64_feature(bool(sample)),
        "image_raw": _bytes_feature(image_raw),
        "mask_raw": _bytes_feature(mask_raw),
    }))
