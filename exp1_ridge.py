import numpy as np

x_train = np.load("data/x_train.npy")
x_test = np.load("data/x_test.npy")
y_train = np.load("data/y_train.npy")
test_ids = np.load("data/test_ids.npy")

header = open("data/dataset/dataset/x_train.csv").readline()
names = np.array(header.strip().split(",")[1:])

admin = ["_STATE", "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE", "SEQNO", "_PSU"]
keep = ~np.isin(names, admin)

nan_frac = np.isnan(x_train).mean(axis=0)
keep = keep & (nan_frac <= 0.8)

x_train = x_train[:, keep]
x_test = x_test[:, keep]
names = names[keep]
print("after dropping columns:", x_train.shape)

medians = np.nanmedian(x_train, axis=0)
x_train = np.where(np.isnan(x_train), medians, x_train)
x_test = np.where(np.isnan(x_test), medians, x_test)

mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
std[std == 0] = 1
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

x_train = np.hstack([np.ones((x_train.shape[0], 1)), x_train])
x_test = np.hstack([np.ones((x_test.shape[0], 1)), x_test])
print("final shapes:", x_train.shape, x_test.shape)

y = np.where(y_train == 1, 1, 0)

np.random.seed(1)
indices = np.random.permutation(len(y))
n_val = int(0.2 * len(y))
val_idx = indices[:n_val]
tr_idx = indices[n_val:]

x_tr, y_tr = x_train[tr_idx], y[tr_idx]
x_val, y_val = x_train[val_idx], y[val_idx]

def f1_score(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return 2 * tp / (2 * tp + fp + fn)

from implementations import ridge_regression

w, loss = ridge_regression(y_tr, x_tr, 1e-4)
scores = x_val @ w

best_f1 = 0
best_t = 0
for t in np.arange(0.05, 0.6, 0.01):
    y_pred = np.where(scores > t, 1, 0)
    f1 = f1_score(y_val, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_t = t
print("best threshold:", round(best_t, 2), "validation F1:", round(best_f1, 4))

with open("results.txt", "a") as f:
    f.write("exp1 ridge | val F1 " + str(round(best_f1, 4)) + " | threshold " + str(round(best_t, 2)) + "\n")

from helpers import create_csv_submission

w, loss = ridge_regression(y, x_train, 1e-4)
test_scores = x_test @ w
test_pred = np.where(test_scores > best_t, 1, -1)
create_csv_submission(test_ids, test_pred, "submission_1_ridge.csv")
print("submission written")