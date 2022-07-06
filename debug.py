import torch
import numpy as np

a= [[[-2, -2, -2], [-2, -2, -2]]]
a = np.array(a)
preds = np.argmax(a, axis=-1)
print(preds)