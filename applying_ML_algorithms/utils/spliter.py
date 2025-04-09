import pandas as pd
import numpy as np

def train_test_spliter(data, target_col, train_size = None, test_size = 0.2):
    
    if train_size is None: train_size = 1 - test_size

    counts = data[target_col].value_counts()
    inds = counts.index
    train = pd.DataFrame(columns = data.columns)
    test = pd.DataFrame(columns = data.columns)

    for idx in inds:
        num_datapoints = counts[idx]
        train_subset = data[data[target_col] == idx].iloc[:int(num_datapoints * train_size), :]
        test_subset = data[data[target_col] == idx].iloc[int(num_datapoints * train_size):, :]
        train = pd.concat([train, train_subset])
        test = pd.concat([test, test_subset])

    train = train.sample(frac = 1).reset_index(drop = True) ## for shuffling the data
    test = test.sample(frac = 1).reset_index(drop = True)

    x_train = train.drop(columns = [target_col], axis = 1)
    y_train = train[target_col]
    x_test = test.drop(columns = [target_col], axis = 1)
    y_test = test[target_col]

    return (x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy())