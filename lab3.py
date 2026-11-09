""" import numpy as np
import cv2
import glob
import os

from keras import Sequential, layers
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import save_model

import matplotlib.pyplot as plt


# ============================================================
# DATA - TRAINING
# ============================================================

data_list = []
label_list = []

print("[INFO] Loading training images...")

for i, address in enumerate(glob.glob("train/*/*")):

    image = cv2.imread(address)

    if image is None:
        print("[WARNING] Could not read:", address)
        continue

    # Resize
    image = cv2.resize(image, (64, 64))

    # Normalize
    image = image / 255

    data_list.append(image)

    # Get label from folder name
    label = os.path.basename(os.path.dirname(address))
    label_list.append(label)

    if i % 200 == 0:
        print(f'[INFO] {i} images processed!')


X_train = np.array(data_list)
y_train = np.array(label_list)


# ============================================================
# DATA - TESTING
# ============================================================

test_data_list = []
test_label_list = []

print("\n[INFO] Loading test images...")

for address in glob.glob("test/*/*"):

    image = cv2.imread(address)

    if image is None:
        print("[WARNING] Could not read:", address)
        continue

    # Resize
    image = cv2.resize(image, (64, 64))

    # Normalize
    image = image / 255

    test_data_list.append(image)

    # Get label from folder name
    label = os.path.basename(os.path.dirname(address))
    test_label_list.append(label)


X_test = np.array(test_data_list)
y_test = np.array(test_label_list)


print("\n[INFO] Training data shape:", X_train.shape)
print("[INFO] Test data shape:", X_test.shape)


# ============================================================
# LABEL ENCODING
# ============================================================

le = LabelEncoder()

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

print("[INFO] Classes:", le.classes_)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# ============================================================
# MODEL - CNN
# ============================================================

model = Sequential([

    layers.Conv2D(6, (3,3), activation='relu'),

    layers.MaxPool2D((2,2)),

    layers.Conv2D(16, (5,5), activation='relu'),

    layers.MaxPool2D((2,2)),

    layers.Flatten(),

    layers.Dense(16, activation='relu'),

    layers.Dense(2, activation='softmax')

])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# ============================================================
# SHOW MODEL
# ============================================================

model.build((None, 64, 64, 3))
model.summary()


# ============================================================
# TRAIN
# ============================================================

print("\n[INFO] Starting training...")

H = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10
)


# ============================================================
# EVALUATE
# ============================================================

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\n================================")
print("TEST RESULTS")
print("================================")

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("================================")


# ============================================================
# PLOT ACCURACY
# ============================================================

epochs = np.arange(
    0,
    len(H.history['accuracy'])
)

plt.plot(
    epochs,
    H.history['accuracy'],
    label='accuracy'
)

plt.plot(
    epochs,
    H.history['val_accuracy'],
    label='val accuracy'
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.title("Cat vs Dog - Accuracy")

plt.show()


# ============================================================
# PLOT LOSS
# ============================================================

plt.plot(
    epochs,
    H.history['loss'],
    label='loss'
)

plt.plot(
    epochs,
    H.history['val_loss'],
    label='val loss'
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.title("Cat vs Dog - Loss")

plt.show()


# ============================================================
# SAVE MODEL
# ============================================================

save_model(
    model,
    'cat_dog_cnn.h5'
)

print("\n[INFO] Model saved as cat_dog_cnn.h5")

 """


import numpy as np
import cv2
import glob
import os

from keras import Sequential, layers
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import save_model

import matplotlib.pyplot as plt


# ============================================================
# DATA - TRAINING
# ============================================================

data_list = []
label_list = []

print("[INFO] Loading training images...")

for i, address in enumerate(glob.glob("train/*/*")):

    image = cv2.imread(address)

    if image is None:
        print("[WARNING] Could not read:", address)
        continue

    # Resize
    image = cv2.resize(image, (64, 64))

    # Normalize
    image = image / 255

    data_list.append(image)

    # Get label from folder name
    label = os.path.basename(os.path.dirname(address))
    label_list.append(label)

    if i % 200 == 0:
        print(f"[INFO] {i} images processed!")


X_train = np.array(data_list)
y_train = np.array(label_list)

index = np.arange(len(X_train))
np.random.shuffle(index)

X_train = X_train[index]
y_train = y_train[index]


# ============================================================
# DATA - TESTING
# ============================================================

test_data_list = []
test_label_list = []

print("\n[INFO] Loading test images...")

for address in glob.glob("test/*/*"):

    image = cv2.imread(address)

    if image is None:
        print("[WARNING] Could not read:", address)
        continue

    # Resize
    image = cv2.resize(image, (64, 64))

    # Normalize
    image = image / 255

    test_data_list.append(image)

    # Get label from folder name
    label = os.path.basename(os.path.dirname(address))
    test_label_list.append(label)


X_test = np.array(test_data_list)
y_test = np.array(test_label_list)


# ============================================================
# SHOW DATA
# ============================================================

print("\n[INFO] Training data shape:", X_train.shape)
print("[INFO] Test data shape:", X_test.shape)


# ============================================================
# LABEL ENCODING
# ============================================================

le = LabelEncoder()

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

print("[INFO] Classes:", le.classes_)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# ============================================================
# MODEL - CNN
# ============================================================

model = Sequential([

    layers.Conv2D(
        32,
        (3, 3),
        activation='relu'
    ),

    layers.MaxPool2D(
        (2, 2)
    ),

    layers.Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),

    layers.MaxPool2D(
        (2, 2)
    ),

    layers.Flatten(),

    layers.Dense(
        16,
        activation='relu'
    ),

    layers.Dense(
        2,
        activation='softmax'
    )
])
# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# ============================================================
# SHOW MODEL
# ============================================================

model.build((None, 64, 64, 3))
model.summary()


# ============================================================
# TRAIN
# ============================================================

print("\n[INFO] Starting training...")

H = model.fit(
    X_train,
    y_train,

    # Use part of the TRAINING data for validation
    validation_split=0.2,

    epochs=10,
    batch_size=32
)


# ============================================================
# EVALUATE
# ============================================================

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\n================================")
print("TEST RESULTS")
print("================================")

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("================================")


# ============================================================
# PLOT ACCURACY
# ============================================================

epochs = np.arange(
    0,
    len(H.history['accuracy'])
)

plt.plot(
    epochs,
    H.history['accuracy'],
    label='accuracy'
)

plt.plot(
    epochs,
    H.history['val_accuracy'],
    label='val accuracy'
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.title("Cat vs Dog - Accuracy")

plt.show()


# ============================================================
# PLOT LOSS
# ============================================================

plt.plot(
    epochs,
    H.history['loss'],
    label='loss'
)

plt.plot(
    epochs,
    H.history['val_loss'],
    label='val loss'
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.title("Cat vs Dog - Loss")

plt.show()


# ============================================================
# SAVE MODEL
# ============================================================

save_model(
    model,
    'cat_dog_cnn.h5'
)

print("\n[INFO] Model saved as cat_dog_cnn.h5")

