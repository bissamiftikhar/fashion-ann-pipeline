import numpy as np
import tensorflow as tf
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

x_test = np.load("data/processed/x_test.npy")
y_test = np.load("data/processed/y_test.npy")

model = tf.keras.models.load_model("models/model.h5")
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {accuracy:.4f}")

with open("metrics.json", "w") as f:
    json.dump({"loss": float(loss), "accuracy": float(accuracy)}, f, indent=2)

y_pred = np.argmax(model.predict(x_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d")
plt.savefig("models/confusion_matrix.png")
print("Metrics written to metrics.json")
