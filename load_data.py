import numpy as np
from helpers import load_csv_data

x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/dataset/dataset/")
print("x_train shape:", x_train.shape)

np.save("data/x_train.npy", x_train)
np.save("data/x_test.npy", x_test)
np.save("data/y_train.npy", y_train)
np.save("data/test_ids.npy", test_ids)
print("Saved!")