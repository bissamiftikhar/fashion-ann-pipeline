import numpy as np
import yaml
import os

os.makedirs("data/processed", exist_ok=True)
with open("params.yaml") as f:
    params = yaml.safe_load(f)

x_train = np.load("data/raw/x_train.npy")
y_train = np.load("data/raw/y_train.npy")
x_test  = np.load("data/raw/x_test.npy")
y_test  = np.load("data/raw/y_test.npy")

x_train = x_train / 255.0
x_test  = x_test  / 255.0

split = int(len(x_train) * (1 - params["preprocess"]["test_size"]))
x_val, y_val = x_train[split:], y_train[split:]
x_train, y_train = x_train[:split], y_train[:split]

np.save("data/processed/x_train.npy", x_train)
np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/x_val.npy",   x_val)
np.save("data/processed/y_val.npy",   y_val)
np.save("data/processed/x_test.npy",  x_test)
np.save("data/processed/y_test.npy",  y_test)
print("Processed data saved to data/processed/")
