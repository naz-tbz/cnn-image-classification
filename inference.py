import numpy as np
import cv2

from keras.models import load_model


# ============================================================
# GET IMAGE PATH FROM USER
# ============================================================

image_path = input("Enter image path: ")


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(image_path)

if image is None:
    print("Could not read image.")
    exit()


# ============================================================
# SAME PREPROCESSING AS TRAINING
# ============================================================

image = cv2.resize(
    image,
    (64, 64)
)

image = image / 255


# ============================================================
# LOAD MODEL
# ============================================================

model = load_model("cat_dog_cnn.h5")


# ============================================================
# PREDICT
# ============================================================

prediction = model.predict(
    np.array([image])
)

label = np.argmax(prediction[0])


# ============================================================
# RESULT
# ============================================================

if label == 0:
    print("Prediction: Cat")
else:
    print("Prediction: Dog")

print(
    "Confidence:",
    prediction[0][label] * 100,
    "%"
)

