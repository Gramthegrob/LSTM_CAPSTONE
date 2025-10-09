import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('/content/drive/MyDrive/Dataset/Improved_All_Combined_hr_rsp_binary.csv')
df_clean = df.dropna(subset=["HR"]).reset_index(drop=True)

# Feature & label
features = df_clean[["HR"]].values
labels = df_clean["Label"].values

# Normalize
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Create sequences
def create_sequences(features, labels, window_size=128):
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:i+window_size])
        y.append(labels[i+window_size])
    return np.array(X), np.array(y)

X, y = create_sequences(features_scaled, labels, window_size=128)
y = to_categorical(y, num_classes=2)

# Split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y.argmax(axis=1), random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=1/3, stratify=y_temp.argmax(axis=1), random_state=42)

# Model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(128, 1)),
    BatchNormalization(),
    LSTM(128, return_sequences=True),
    BatchNormalization(),
    LSTM(128),
    Dense(100, activation='tanh'),
    Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# Predict
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Report
print(classification_report(y_true_classes, y_pred_classes))

# Plotting
def plot_training_history(history):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels=['Not Stressed', 'Stressed']):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# Show plots
plot_training_history(history)
plot_confusion_matrix(y_true_classes, y_pred_classes)
