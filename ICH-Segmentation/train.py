import os
import argparse
import core.segmentation_models as sm
from keras.utils import Sequence

EPOCH = 100
BS = 8


# def split_dataset(tfrecords_path, test_rate=0.2, buffer_size=10000):
#     dataset = tf.data.Dataset(os.path.join(tfrecords_path, "*."))
#     dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=False)
#
#     if test_rate:
#         sep = int(1.0 / test_rate)
#
#         def is_test(x, y):
#             return x % sep == 0
#
#         def is_train(x, y):
#             return not is_test(x, y)
#
#         def recover(x, y):
#             return y
#
#         test_dataset = dataset.enumerate(start=1).filter(is_test).map(recover)
#         train_dataset = dataset.enumerate(start=1).filter(is_train).map(recover)
#
#     else:
#         test_dataset, test_dataset = dataset, None
#
#     return train_dataset, test_dataset


def build_unet_model():
    model = sm.Unet(
        backbone_name="senet154",
        input_shape=(512, 512, 1),
        encoder_weights="None"
    )
    model.compile(
        'Adam',
        loss=sm.losses.dice_loss,
        metrics=[sm.metrics.iou_score, sm.metrics.f1_score],
    )

    return model
    #
    # model.fit_generator(
    #     generator=Sequence,
    #     epoch=EPOCH,
    #     batch_size=BS,
    #     use_multiprocessing=True
    # )


def main(args):
    model = build_unet_model()
    

if __name__ == '__main__':
    CURRENT_DIR = os.path.abspath(os.path.join(__file__, os.pardir))
    print(CURRENT_DIR)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        default="ICH-Segmentation",
                        help="Name of training model"
                        )
    parser.add_argument("--tfrecords_path",
                        default="datasets/ICH_420/TFRecords/train",
                        help="Training Dataset"
                        )
    parser.add_argument("--batch_size",
                        default=8,
                        help="Model update after go through 'batch_size' data.",
                        type=int
                        )
    parser.add_argument("--learning_rate",
                        default=1e-4,
                        help="Initial learning rate.",
                        type=float
                        )
    parser.add_argument("--loss_fn",
                        default="dice",
                        help="[BCE, MES, dice, ...]"
                        )
    parser.add_argument("--optimizer",
                        default="sgd",
                        help="[adam, sgd]"
                        )
    main(parser.parse_args())
