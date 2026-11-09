# CNN Image Classification — Cats vs Dogs

This project uses a convolutional neural network (CNN) to classify cats and dogs.
The original Lab 3 Python code, training/test images, sample images, and saved model are retained.

## Files
- `lab3.py` — trains/evaluates the CNN, shows graphs, and saves the model.
- `inference.py` — loads the saved model and predicts a user-selected image.
- `train/Cat`, `train/Dog` — training images.
- `test/Cat`, `test/Dog` — test images.
- `internet_cat*.png`, `internet_dog*.png` — sample images.
- `cat_dog_cnn.h5` — original saved trained model.

## Install
From this project directory:

```bash
python -m pip install -r requirements.txt
```

## Predict with the saved model
```bash
python inference.py
```
When prompted, enter an image path such as `internet_cat1.png`.

## Train again (optional)
```bash
python lab3.py
```
Run commands from the project directory so relative paths resolve.
Training displays graphs and overwrites `cat_dog_cnn.h5`.

## Dataset Information

The original training dataset contains over 2,000 cat and dog images. The `train` folder was not uploaded to GitHub to keep the repository small and easy to download.

The `test` folder and sample images are included for demonstration purposes.

The original training code is provided in `lab3.py`. To train the model again, the training dataset must be added to the project.

A previously trained CNN model (`cat_dog_cnn.h5`) is included for image classification.

## Important
The original Python code has not been changed. Syntax was checked, but training and
inference were not run in this environment. Before making the repository public,
confirm you have permission to redistribute the included training/test/sample images.
