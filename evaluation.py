

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import config

def build_evaluation_model():
    """
    Rebuilds the Functional Keras model architecture for testing 
    to maintain graph compatibility with the weights file.
    """
    inputs = tf.keras.Input(shape=config.INPUT_SHAPE)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv2d_0')(inputs)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_1')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv2d_2')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="glaucoma_cnn_eval")
    
    try:
        model.load_weights(config.MODEL_WEIGHTS_PATH)
        print("[INFO] Model weights loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model weights from {config.MODEL_WEIGHTS_PATH}: {e}")
        
    return model

def evaluate_performance(X_test, y_true):
    """
    Generates quantitative diagnostic metrics for the EYESENTINEL model.
    """
    model = build_evaluation_model()
    
    print("[INFO] Running inference on test dataset...")
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    # 1. Classification Report
    print("\n" + "="*40)
    print("      CLINICAL CLASSIFICATION REPORT      ")
    print("="*40)
    print(classification_report(y_true, y_pred, target_names=config.BINARY_CLASSES))
    
    # 2. ROC-AUC Score Calculation
    auc_score = roc_auc_score(y_true, y_pred_probs)
    print(f"ROC-AUC Score: {auc_score:.4f}\n")
    
    # 3. Confusion Matrix Plotting
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.BINARY_CLASSES, 
                yticklabels=config.BINARY_CLASSES)
    
    plt.title('EYESENTINEL - Diagnostic Confusion Matrix')
    plt.ylabel('Ground Truth (Actual)')
    plt.xlabel('Model Prediction')
    plt.tight_layout()
    
    plt.savefig(config.CONFUSION_MATRIX_PATH)
    print(f"[INFO] Confusion matrix visual saved to: {config.CONFUSION_MATRIX_PATH}")

if __name__ == "__main__":
   
    print("[INFO] Initializing evaluation pipeline with synthetic data...")
    test_samples = 100
    dummy_X = np.random.random((test_samples, config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS))
    dummy_Y = np.random.randint(0, 2, test_samples)
    
    evaluate_performance(dummy_X, dummy_Y)