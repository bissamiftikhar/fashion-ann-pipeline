import numpy as np
import yaml
import tensorflow as tf
import pandas as pd
import os

os.makedirs("models", exist_ok=True)
with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]

x_train = np.load("data/processed/x_train.npy")
y_train = np.load("data/processed/y_train.npy")
x_val   = np.load("data/processed/x_val.npy")
y_val   = np.load("data/processed/y_val.npy")

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(params["dense_units"], activation="relu"),
    tf.keras.layers.Dropout(params["dropout_rate"]),
    tf.keras.layers.Dense(10, activation="softmax"),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history = model.fit(
    x_train, y_train,
    epochs=params["epochs"],
    batch_size=params["batch_size"],
    validation_data=(x_val, y_val),
)
model.save("models/model.h5")
pd.DataFrame(history.history).to_csv("models/history.csv", index=False)
print("Model saved to models/model.h5")
