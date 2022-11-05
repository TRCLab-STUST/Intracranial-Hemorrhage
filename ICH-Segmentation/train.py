import argparse
import core.segmentation_models as sm
from keras.utils import Sequence

EPOCH = 100
BS = 8


def main(args):
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
    
    model.fit_generator(
        generator=Sequence,
        epoch=EPOCH,
        batch_size=BS,
        use_multiprocessing=True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        default="ICH-Segmentation",
                        help="Name of training model"
                        )
    parser.add_argument("--tfrecords_path",
                        default="datasets/ICH_420/tfrecords/ICH_420_val.tfrecord",
                        help="Test tfrecord"
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
