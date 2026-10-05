import tensorflow as tf
import numpy as np
import yaml

print("tensorflow", tf.__version__)
print("numpy", np.__version__)
with open("params.yaml") as f:
    print("params loaded:", list(yaml.safe_load(f).keys()))
