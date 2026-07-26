

import os

# Base Directory Mapping
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

# Ensure required directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


IMG_HEIGHT = 256
IMG_WIDTH = 256
CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
BATCH_SIZE = 32


BINARY_CLASSES = ["Normal", "Glaucoma"]
MULTICLASS_CLASSES = ["Normal", "Glaucoma", "Diabetic Retinopathy", "Cataract"]


MODEL_WEIGHTS_PATH = os.path.join(MODEL_DIR, "eye_disease_model.h5")
CONFUSION_MATRIX_PATH = os.path.join(REPORT_DIR, "confusion_matrix.png")