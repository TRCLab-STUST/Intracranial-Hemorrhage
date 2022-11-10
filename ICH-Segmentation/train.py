# %% [markdown]
# # 腦溢血自動切割 (Intracerebral Hemorrhage Segmentation)

# %% [markdown]
# ## 1. 前期準備

# %% [markdown]
# ### 1.1 掛載 Google 雲端硬碟

# %%
# from google.colab import drive
# drive.mount("/content/drive")

# %% [markdown]
# ### 1.2 套件庫

# %% [markdown]
# #### 1.2.1 安裝額外套件

# %%
# !pip install -U segmentation-models keras-applications image-classifiers efficientnet

# %% [markdown]
# #### 1.2.2 匯入套件庫

# %%
import os
import sys
import math
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import segmentation_models as sm
from tqdm import tqdm, trange

# %% [markdown]
# #### 1.2.3 套件版本

# %%
print(f"Python: {sys.version}")
print(f"Tensorflow: {tf.__version__}")
print(f"Tensorflow-Keras: {keras.__version__}")
print(f"Segmentation_model: {sm.__version__}")

# %% [markdown]
# ### 1.3 定義全域變數

# %%
PROJECT_DIR = os.path.abspath("/workspaces/Intracranial-Hemorrhage/ICH-Segmentation")
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
DATASET_DIR = os.path.join(PROJECT_DIR, "datasets")
ICH420_DATASET = os.path.join(DATASET_DIR, "ICH_420")

# %% [markdown]
# ## 2. 資料集
# 

# %% [markdown]
# ### 2.1 ICH420 資料集

# %% [markdown]
# #### 2.1.1 抓取所有 TFRecord 檔案

# %%
tfrecords_filelist = tf.io.gfile.glob(
    os.path.join(ICH420_DATASET, "TFRecords", "train", "*.tfrecord")
)

# %% [markdown]
# #### 2.1.2 將檔案資料載入為 TFRecord 格式

# %%
ich420_tfrecords = tf.data.TFRecordDataset(
    [tfrecords_filelist],
    compression_type="GZIP",
    num_parallel_reads=tf.data.AUTOTUNE
)

# %% [markdown]
# #### 2.1.3 TFRecord 資料解析

# %%
def _ich_tfrecord_parser(example):
  features_description = {
      "filename": tf.io.FixedLenFeature([], tf.string),
      "number": tf.io.FixedLenFeature([], tf.int64),
      "sample": tf.io.FixedLenFeature([], tf.int64),
      "image_raw": tf.io.FixedLenFeature([], tf.string),
      "mask_raw": tf.io.FixedLenFeature([], tf.string)
  }

  example = tf.io.parse_single_example(example, features_description)
  image = tf.io.decode_png(example["image_raw"], channels=1)
  mask = tf.cast(
      tf.io.decode_png(example["mask_raw"], channels=1),
      dtype=tf.float32
  )
  sample = tf.cast(example["sample"], tf.bool)

  return (image, mask), sample

# %% [markdown]
# #### 2.1.4 TFRecord 資料集預處理

# %%
def load_dataset(
    tfrecords, 
    train_rate=0.8, 
    only_sample=None, 
    shuffle=True,
    repeat=False, 
    batch_size=1):
  # 解析資料集
  dataset = tfrecords.map(
      _ich_tfrecord_parser,
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=False
  )
  
  # 提出指定的樣本類型 (None: 全部; True: 正樣本; False: 負樣本)
  if only_sample is not None:
    dataset = dataset.filter(lambda x, y: y == only_sample)
  
  dataset = dataset.map(
      lambda x, _: x,
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=False
  )

  # 計算尺寸
  total_num = 1
  for total_num, _ in enumerate(tqdm(
      dataset, total=total_num, desc="Count"), start=1): 
    pass
  
  print(f"Dataset Total Num: {total_num}")

  # 打散資料集
  dataset = dataset.shuffle(
      total_num, 
      reshuffle_each_iteration=False
  )
  
  # 分為訓練及與測試集
  train_num = int(total_num * train_rate)
  valid_num = total_num - train_num
  train_dataset = dataset.take(train_num)
  valid_dataset = dataset.skip(train_num)
  
  if shuffle:
    train_dataset = train_dataset.shuffle(
        train_num
    )
  
  if repeat:
    train_dataset = train_dataset.repeat()

  train_dataset = train_dataset.batch(
      batch_size,
      drop_remainder=True,
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=False
  )
  valid_dataset = valid_dataset.batch(
      batch_size,
      drop_remainder=True,
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=False
  )

  train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
  valid_dataset = valid_dataset.prefetch(tf.data.AUTOTUNE)
  
  return (train_dataset, valid_dataset), (train_num, valid_num)

# %% [markdown]
# ## 3. 網路模型

# %% [markdown]
# ### 3.1 UNet 模型

# %%
def UNet(
    backbone="efficientnetb3",
    encoder_weights=None,
    input_shape=(None, None, 1),
    optimizer=keras.optimizers.Adam(1e-4),
    loss=(sm.losses.DiceLoss() + (1 * sm.losses.BinaryFocalLoss())),
    metrics=[sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)],
    jit_compile=False
):
  model = sm.Unet(
    backbone, 
    encoder_weights=encoder_weights, 
    input_shape=input_shape
  )
  model.compile(
      optimizer,
      loss=loss,
      metrics=metrics,
      jit_compile=jit_compile
  )

  return model

# %% [markdown]
# ## 4. 訓練模型

# %% [markdown]
# ### 4.1 超參數

# %%
BACKBONE = "efficientnetb3"
TRAING_NAME = "ICH420"
BATCH_SIZE = 4
EPOCH = 2
LEARNING_RATE = 1e-4
TRAINING_RATE = 0.8

# %%
CALLBACKS = [
    keras.callbacks.TensorBoard(log_dir=LOGS_DIR),
    keras.callbacks.ModelCheckpoint(
        f"{TRAING_NAME}-" + "{epoch}.h5",
        save_weights_only=True,
        save_best_only=False
    ),
    keras.callbacks.ReduceLROnPlateau()
]

# %% [markdown]
# ### 載入資料集

# %%
(x_dataset, y_dataset), (x_num, y_num) = load_dataset(
    ich420_tfrecords,
    train_rate=TRAINING_RATE, 
    only_sample=True, 
    shuffle=True,
    repeat=True, 
    batch_size=BATCH_SIZE
)

# %% [markdown]
# ### 載入模型

# %%
model = UNet(
    backbone=BACKBONE,
    optimizer=keras.optimizers.Adam(LEARNING_RATE),
    jit_compile=False
)

# %% [markdown]
# ### 訓練

# %%
history = model.fit(
    x_dataset,
    epochs=EPOCH,
    steps_per_epoch=int(math.ceil(1. * x_num) / BATCH_SIZE),
    callbacks=[keras.callbacks.ReduceLROnPlateau()],
    validation_data=y_dataset,
    verbose=1
)

# %% [markdown]
# ### 訓練歷史圖

# %%
plt.figure(figsize=(12,4))
plt.subplot(1,2,(1))
plt.plot(history.history['iou_score'],linestyle='-.')
plt.plot(history.history['val_iou_score'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='lower right')

plt.subplot(1,2,(2))
plt.plot(history.history['loss'],linestyle='-.')
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper right')
plt.show()


