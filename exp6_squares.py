import numpy as np

x_train = np.load("data/x_train.npy")
x_test = np.load("data/x_test.npy")
y_train = np.load("data/y_train.npy")
test_ids = np.load("data/test_ids.npy")

header = open("data/dataset/dataset/x_train.csv").readline()
names = np.array(header.strip().split(",")[1:])

def replace_codes(x, j, nan_codes, zero_code):
    is_code = np.isin(x[:, j], nan_codes)
    x[is_code, j] = np.nan
    if zero_code is not None:
        x[x[:, j] == zero_code, j] = 0

for j in range(x_train.shape[1]):
    col_max = np.nanmax(x_train[:, j])
    if col_max == 9:
        nan_codes = [7, 9]
        zero_code = None
    elif col_max == 99:
        nan_codes = [77, 99]
        zero_code = 88
    elif col_max == 999:
        nan_codes = [777, 999]
        zero_code = 888
    elif col_max == 9999:
        nan_codes = [7777, 9999]
        zero_code = None
    else:
        continue
    replace_codes(x_train, j, nan_codes, zero_code)
    replace_codes(x_test, j, nan_codes, zero_code)

admin = ["_STATE", "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE", "SEQNO", "_PSU"]
keep = ~np.isin(names, admin)

nan_frac = np.isnan(x_train).mean(axis=0)
keep = keep & (nan_frac <= 0.8)

x_train = x_train[:, keep]
x_test = x_test[:, keep]
names = names[keep]
print("after dropping columns:", x_train.shape)

n_unique = np.zeros(x_train.shape[1])
for j in range(x_train.shape[1]):
    column = x_train[:, j]
    present_values = column[~np.isnan(column)]
    n_unique[j] = len(np.unique(present_values))

is_cat = (n_unique >= 3) & (n_unique <= 10)
print("categorical columns:", np.sum(is_cat))

onehot_train = []
onehot_test = []
for j in np.where(is_cat)[0]:
    column = x_train[:, j]
    values = np.unique(column[~np.isnan(column)])
    for v in values:
        onehot_train.append(x_train[:, j] == v)
        onehot_test.append(x_test[:, j] == v)
    onehot_train.append(np.isnan(x_train[:, j]))
    onehot_test.append(np.isnan(x_test[:, j]))

onehot_train = np.array(onehot_train, dtype=np.float32).T
onehot_test = np.array(onehot_test, dtype=np.float32).T
print("one-hot columns:", onehot_train.shape[1])

x_train = x_train[:, ~is_cat]
x_test = x_test[:, ~is_cat]

medians = np.nanmedian(x_train, axis=0)
x_train = np.where(np.isnan(x_train), medians, x_train)
x_test = np.where(np.isnan(x_test), medians, x_test)

mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
std[std == 0] = 1
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

sq_train = x_train ** 2
sq_test = x_test ** 2

sq_mean = sq_train.mean(axis=0)
sq_std = sq_train.std(axis=0)
sq_std[sq_std == 0] = 1
sq_train = (sq_train - sq_mean) / sq_std
sq_test = (sq_test - sq_mean) / sq_std
print("squared columns:", sq_train.shape[1])

x_train = np.hstack([np.ones((x_train.shape[0], 1)), x_train, sq_train, onehot_train])
x_test = np.hstack([np.ones((x_test.shape[0], 1)), x_test, sq_test, onehot_test])
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

from implementations import reg_logistic_regression

def best_threshold(probs, y_true):
    best_f1 = 0
    best_t = 0
    for t in np.arange(0.05, 0.6, 0.01):
        y_pred = np.where(probs > t, 1, 0)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

lambdas = [1e-3]
max_iters = 1000

best_f1 = 0
for lam in lambdas:
    w, loss = reg_logistic_regression(y_tr, x_tr, lam, np.zeros(x_tr.shape[1]), max_iters, 0.5)
    t, f1 = best_threshold(sigmoid(x_val @ w), y_val)
    print("lambda", lam, "| loss", round(loss, 4), "| threshold", round(t, 2), "| F1", round(f1, 4))
    with open("results.txt", "a") as f:
        f.write("exp6 squares lambda " + str(lam) + " iters " + str(max_iters) + " | val F1 " + str(round(f1, 4)) + "\n")
    if f1 > best_f1:
        best_f1 = f1
        best_t = t
        best_lam = lam

print("best lambda:", best_lam, "threshold:", round(best_t, 2), "F1:", round(best_f1, 4))

from helpers import create_csv_submission

w, loss = reg_logistic_regression(y, x_train, best_lam, np.zeros(x_train.shape[1]), max_iters, 0.5)
test_pred = np.where(sigmoid(x_test @ w) > best_t, 1, -1)
create_csv_submission(test_ids, test_pred, "submission_6_squares.csv")
print("submission written")