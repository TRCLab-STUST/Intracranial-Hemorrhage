import tensorflow as tf
from tensorflow import keras
from keras import layers, models, Input, Model

# model = models.Sequential(name="U-net")

input = Input(shape=(512, 512, 1), name="input")
# Feature 1, 2, 3, 4
features = []
for idx, filter_size in enumerate([64, 128, 256, 512]):
    x = layers.Conv2D(64, 3, strides=2, padding="same")(input if filter_size == 64 else x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    layers.Conc
    feature = x = layers.Conv2D(filter_size, 3, activation="relu", name=f"conv_{idx}-1", padding="same", strides=2)
    # feature = x = layers.Conv2D(filter_size, 3, activation="relu", name=f"conv_{idx}-2", padding="same")(x)
    if not filter == 1024:
        x = layers.MaxPool2D(2, name=f"feature_{idx}_maxpooling")(x)
    
    features.append(feature)

model = Model(inputs=input, outputs=x)
model.summary()

# Upsampling
# x = layers.UpSampling2D(2)(x)
# layers.cop


# model.add(layers.Conv2D(64, 3, activation="relu", input_shape=(512,512,1)))
# model.add(layers.Conv2D(64, 3, activation="relu"))
# model.add(layers.Conv2D(64, 3, activation="relu"))
# model.add(layers.MaxPool2D(2))
# model.add(layers.Conv2D(64, 3, activation="relu"))
# model.add(layers.Conv2D(64, 3, activation="relu"))
# model.add(layers.Conv2D(64, 3, activation="relu"))
# model.add(layers.MaxPool2D(2))
# model.add(layers.Conv2D(64, 3, activation="relu"))
# model.add(layers.Conv2D(64, 3, activation="relu"))
# model.add(layers.Conv2D(64, 3, activation="relu"))






















