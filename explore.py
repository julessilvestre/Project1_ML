import numpy as np

x_train = np.load("data/x_train.npy")
x_test = np.load("data/x_test.npy")
y_train = np.load("data/y_train.npy")
print("x_train:", x_train.shape)
print("x_test:", x_test.shape)

header = open("data/dataset/dataset/x_train.csv").readline()
names = np.array(header.strip().split(",")[1:])
print("number of names:", len(names))
print(names[:10])

values, counts = np.unique(y_train, return_counts=True)
print("label values:", values)
print("counts:", counts)
print("fractions:", counts / len(y_train))

nan_frac = np.isnan(x_train).mean(axis=0)
print("columns with no NaN:", np.sum(nan_frac == 0))
print("columns with >50% NaN:", np.sum(nan_frac > 0.5))
print("columns with >80% NaN:", np.sum(nan_frac > 0.8))

n_unique = []
for j in range(x_train.shape[1]):
    col = x_train[:, j]
    col = col[~np.isnan(col)]
    n_unique.append(len(np.unique(col)))
n_unique = np.array(n_unique)
print("binary columns (2 values):", np.sum(n_unique == 2))
print("small categorical (3-10 values):", np.sum((n_unique >= 3) & (n_unique <= 10)))
print("many values (>10):", np.sum(n_unique > 10))

to_check = ["GENHLTH", "SEX", "SMOKE100", "_AGEG5YR", "_BMI5"]

for name in to_check:
    j = np.where(names == name)[0][0]
    column = x_train[:, j]
    present_values = column[~np.isnan(column)]
    values, counts = np.unique(present_values, return_counts=True)
    print(name, "- missing fraction:", round(nan_frac[j], 3))
    print("  values:", values[:12])
    print("  counts:", counts[:12])