# Importing neccessary packages
import numpy as np
import pandas as pd

from random import randint

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style('white')

from sklearn.model_selection import train_test_split

from skimage.transform import resize

from keras.preprocessing.image import load_img
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout

# =======================================================================================================
# Params and helper functions
img_size_ori = 101
img_size_target = 128

def upsample(img):
  if img_size_ori == img_size_target:
    return img
  return resize(img, (img_size_target, img_size_target), mode = 'constant', preserve_range = True)

def downsample(img):
  if img_size_ori == img_size_target:
    return img
  return resize(img, (img_size_ori, img_size_ori), mode = 'constant', preserve_range = True)

# =======================================================================================================
# Loading train/test ids and depths
train_df = pd.read_csv("train.csv", index_col = "id", usecols = [0])
depths_df = pd.read_csv("depths.csv", index_col = "id")

train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.isin(train_df.index)]

# Loading the images and masks into DataFrame and divide by 255
train_df["images"] = [np.array(load_img("/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in train_df.index]
train_df["masks"] = [np.array(load_img("/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in train_df.index]


# Calculating the salt coverage and corresponding classes
train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
def cov_to_class(val):
      for i in range(0, 11):
    if val*10<=i:
      return i

train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
# =======================================================================================================
# Some Graphic Visualisations
fig, axs = plt.subplots(1, 2, figsize =(15, 5))
sns.distplot(train_df.coverage, kde = False, ax = axs[0])
sns.distplot(train_df.coverage_class, kde = False, ax = axs[1])
plt.suptitle('Salt Coverage')
axs[0].set_xlabel('Coverage')
axs[1].set_xlabel('Coverage Class')

# Sactter Plot of Coverage vs Coverage Class
plt.scatter(train_df.coverage, train_df.coverage_class)
plt.xlabel('Coverage')
plt.ylabel('Coverage Class')

# Plotting the depth distributions
sns.distplot(train_df.z, label = 'Train')
sns.distplot(test_df.z, label = 'Test')
plt.legend()
plt.title('Depth distribution')

# Showing example images
max_images = 60
grid_width = 15
grid_height = int(max_images / grid_width)
fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))
for i, idx in enumerate(train_df.index[:max_images]):
    img = train_df.loc[idx].images
    mask = train_df.loc[idx].masks
    ax = axs[int(i / grid_width), i % grid_width]
    ax.imshow(img, cmap="Greys")
    ax.imshow(mask, alpha=0.3, cmap="Greens")
    ax.text(1, img_size_ori-1, train_df.loc[idx].z, color="black")
    ax.text(img_size_ori - 1, 1, round(train_df.loc[idx].coverage, 2), color="black", ha="right", va="top")
    ax.text(1, 1, train_df.loc[idx].coverage_class, color="black", ha="left", va="top")
    ax.set_yticklabels([])
    ax.set_xticklabels([])
plt.suptitle("Green: salt. Top-left: coverage class, top-right: salt coverage, bottom-left: depth")

# =======================================================================================================
# Create Train/Test Split Stratified by Salt Coverage
ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
    train_df.index.values,
    np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    train_df.coverage.values,
    train_df.z.values,
    test_size=0.2, stratify=train_df.coverage_class, random_state=1337)

print(x_train.shape)
print(x_valid.shape)

# Upscaling Images
tmp_img = np.zeros((img_size_target, img_size_target), dtype = train_df.images.loc[ids_train[10]].dtype)
tmp_img[:img_size_ori, :img_size_ori] = train_df.images.loc[ids_train[10]]
fig, axs = plt.subplots(1, 2, figsize =(15, 5))
axs[0].imshow(tmp_img, cmap ="Greys")
axs[0].set_title("Original Image")
axs[1].imshow(x_train[10].squeeze(), cmap = "Greys")
axs[1].set_title("Scaled Image")
# =========================================================================================================
# Building Model
def build_model(input_layer, start_neurons):
  # 128->64
  conv1 = Conv2D(start_neurons * 1, (3, 3), activation = 'relu', padding = 'same')(input_layer)
  conv1 = Conv2D(start_neurons * 1, (3, 3), activation = 'relu',padding = 'same')(conv1)
  pool1 = MaxPooling2D((2, 2))(conv1)
  pool1 = Dropout(0.25)(pool1)
  
  # 64->32
  conv2 = Conv2D(start_neurons * 2, (3, 3), activation = 'relu', padding = 'same')(pool1)
  conv2 = Conv2D(start_neurons * 2, (3, 3), activation = 'relu', padding = 'same')(conv2)
  pool2 = MaxPooling2D((2, 2))(conv2)
  pool2 = Dropout(0.5)(pool2)
  
  # 32->16
  conv3 = Conv2D(start_neurons * 4, (3, 3), activation = 'relu', padding = 'same')(pool2)
  conv3 = Conv2D(start_neurons * 4, (3, 3), activation = 'relu', padding = 'same')(conv3)
  pool3 = MaxPooling2D((2, 2))(conv3)
  pool3 = Dropout(0.5)(pool3)
  
  # 16->8
  conv4 = Conv2D(start_neurons * 8, (3, 3), activation = 'relu', padding = 'same')(pool3)
  conv4 = Conv2D(start_neurons * 8, (3, 3), activation = 'relu', padding = 'same')(conv4)
  pool4 = MaxPooling2D((2, 2))(conv4)
  pool4 = Dropout(0.5)(pool4)
  
  # Middle
  convm = Conv2D(start_neurons * 16, (3, 3), activation = 'relu', padding = 'same')(pool4)
  convm = Conv2D(start_neurons * 16, (3, 3), activation = 'relu', padding = 'same')(convm)
  
  # 8->16
  deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides = (2, 2), padding='same')(convm)
  uconv4 = concatenate([deconv4, conv4])
  uconv4 = Dropout(0.5)(uconv4)
  uconv4 = Conv2D(start_neurons * 8, (3, 3), activation = 'relu', padding = 'same')(uconv4)
  uconv4 = Conv2D(start_neurons * 8, (3, 3), activation = 'relu', padding = 'same')(uconv4)
  
  # 16->32
  deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides = (2, 2), padding = 'same')(uconv4)
  uconv3 = concatenate([deconv3, conv3])
  uconv3 = Dropout(0.5)(uconv3)
  uconv3 = Conv2D(start_neurons * 4,(3, 3), activation = 'relu', padding = 'same')(uconv3)
  uconv3 = Conv2D(start_neurons * 4,(3, 3), activation ='relu', padding ='same')(uconv3)
  
  # 32->64
  deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides = (2, 2), padding = 'same')(uconv3)
  uconv2 = concatenate([deconv2, conv2])
  uconv2 = Dropout(0.5)(uconv2)
  uconv2 = Conv2D(start_neurons * 2, (3, 3), activation = 'relu', padding = 'same')(uconv2)
  uconv2 = Conv2D(start_neurons * 2, (3, 3), activation = 'relu', padding = 'same')(uconv2)
  
  # 64->128
  deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides = (2, 2), padding ='same')(uconv2)
  uconv1 = concatenate([deconv1, conv1])
  uconv1 = Dropout(0.5)(uconv1)
  uconv1 = Conv2D(start_neurons * 1, (3, 3), activation = 'relu', padding = 'same')(uconv1)
  uconv1 = Conv2D(start_neurons * 1, (3, 3), activation = 'relu', padding = 'same')(uconv1)
  
  output_layer = Conv2D(1, (1, 1), padding = 'same', activation = 'sigmoid')(uconv1)
  
  return output_layer

input_layer = Input((img_size_target, img_size_target, 1))
output_layer = build_model(input_layer, 16)

model = Model(input_layer, output_layer)
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())

# =========================================================================================================
# Data Augmentation
x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis = 0)
y_train = np.append(y_train, [np.fliplr(y) for y in y_train], axis = 0)

# Visualising Augmented Images
fig, axs = plt.subplots(2, 10, figsize = (15, 3))
for i in range(10):
  axs[0][i].imshow(x_train[i].squeeze(), cmap = 'Greys')
  axs[0][i].imshow(y_train[i].squeeze(), cmap = 'Greens', alpha = 0.3)
  axs[1][i].imshow(x_train[int(len(x_train)/2 + i)].squeeze(), cmap = 'Greys')
  axs[1][i].imshow(y_train[int(len(y_train)/2 + i)].squeeze(), cmap = 'Greens', alpha = 0.3)
  
fig.suptitle("Top row: original images, bottom row: augmented images")
# =========================================================================================================
# Training the model
early_stopping = EarlyStopping(patience=10, verbose=1)
model_checkpoint = ModelCheckpoint("./keras.model", save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)

epochs = 200
batch_size = 32

history = model.fit(x_train, y_train,
                    validation_data=[x_valid, y_valid], 
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping, model_checkpoint, reduce_lr])

model = load_model('./keras.model')

# =========================================================================================================
# Predictions for validation set
preds_valid = model.predict(x_valid).reshape(-1, img_size_target, img_size_target)
preds_valid = np.array([downsample(x) for x in preds_valid])
y_valid_ori = np.array([train_df.loc[idx].masks for idx in ids_valid])

# Visualising Predictions
max_images = 60
grid_width = 15
grid_height = int(max_images / grid_width)
fig, axs = plt.subplots(grid_height, grid_width, figsize = (grid_width, grid_height))
for i, idx in enumerate(ids_valid[:max_images]):
  img = train_df.loc[idx].images
  mask = train_df.loc[idx].masks
  pred = preds_valid[i]
  ax = axs[int(i / grid_width), i % grid_width]
  ax.imshow(img, cmap = 'Greys')
  ax.imshow(mask, alpha = 0.3, cmap ='Greens')
  ax.imshow(pred, alpha = 0.3, cmap = 'OrRd')
  ax.text(1, img_size_ori - 1, train_df.loc[idx].z, color = 'black')
  ax.text(img_size_ori - 1, 1, round(train_df.loc[idx].coverage, 2), color = 'black', ha = 'right', va = 'top')
  ax.text(1, 1, train_df.loc[idx].coverage_class, color = 'black', ha = 'left', va = 'top')
  ax.set_yticklabels([])
  ax.set_xticklabels([])
  
plt.suptitle("Green: salt, Red: prediction, Top-left: Coverage Class, Top-right : salt coverage, Bottom-Left: Depth")

# =========================================================================================================
# Scoring and threshold optimization using IoU
def iou_metric(y_true_in, y_pred_in, print_table = False):
  labels = y_true_in
  y_pred = y_pred_in
  
  true_objects = 2
  pred_objects = 2
  
  intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins = (true_objects, pred_objects))[0]
  
  # Compute areas(needed for finding the union between all objects)
  area_true = np.histogram(labels, bins = true_objects)[0]
  area_pred = np.histogram(y_pred, bins = pred_objects)[0]
  area_true = np.expand_dims(area_true, -1)
  area_pred = np.expand_dims(area_pred, 0)
  
  # Compute Union
  union = area_true + area_pred - intersection
  
  # Exclude background from analysis
  intersection = intersection[1:, 1:]
  union = union[1:, 1:]
  union[union == 0] = 1e-9
  
  # Compute Intersection over Union
  iou = intersection / union
  
  # Precision helper function
  def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis = 1) == 1  # Correct Objects
    false_positives = np.sum(matches, axis = 0) == 0 # Missed Objects
    false_negatives = np.sum(matches, axis = 1) == 0 # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn
  
  # Loop over IoU thresholds
  prec = []
  if print_table:
    print("Thresh\tTP\tFP\tFN\tPrec.")
  for t in np.arange(0.5, 1.0, 0.05):
    tp, fp, fn = precision_at(t, iou)
    if (tp + fp + fn) > 0:
      p = tp /  (tp + fp + fn)
    else:
      p = 0
    if print_table:
      print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
    prec.append(p)
  
  if print_table:
    print("AP\t-\t-\t-\t{1.3f}".format(np.mean(prec)))
  
  return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
  batch_size = y_true_in.shape[0]
  metric = []
  for batch in range(batch_size):
    value = iou_metric(y_true_in[batch], y_pred_in[batch])
    metric.append(value)
  return np.mean(metric)    

thresholds = np.linspace(0, 1, 50)
ious = np.array([iou_metric_batch(y_valid_ori, np.int32(preds_valid > threshold)) for threshold in thresholds])

threshold_best_index = np.argmax(ious[9:-10]) + 9
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

# Plotting the best threshold
plt.plot(thresholds, ious)
plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
plt.xlabel("Threshold")
plt.ylabel("IoU")
plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
plt.legend()

