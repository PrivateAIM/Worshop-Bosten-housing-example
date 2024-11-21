import numpy as np
import pandas as pd
from tensorflow.keras.datasets import boston_housing
import os

def save_split_boston_housing_csv(n, output_dir):
    # Load Boston Housing dataset
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()

    # Calculate the size of each split for train and test data
    train_split_size = len(x_train) // n
    test_split_size = len(x_test) // n

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Split and save the training data
    for i in range(n):
        train_start_idx = i * train_split_size
        train_end_idx = (i + 1) * train_split_size if i != n - 1 else len(x_train)

        x_train_split = x_train[train_start_idx:train_end_idx]
        y_train_split = y_train[train_start_idx:train_end_idx]

        train_df = pd.DataFrame(x_train_split)
        train_df['target'] = y_train_split

        train_df.to_csv(os.path.join(output_dir, f'train_split_{i}.csv'), index=False)

    # Split and save the test data
    for i in range(n):
        test_start_idx = i * test_split_size
        test_end_idx = (i + 1) * test_split_size if i != n - 1 else len(x_test)

        x_test_split = x_test[test_start_idx:test_end_idx]
        y_test_split = y_test[test_start_idx:test_end_idx]

        test_df = pd.DataFrame(x_test_split)
        test_df['target'] = y_test_split

        test_df.to_csv(os.path.join(output_dir, f'test_split_{i}.csv'), index=False)

    print(f'Successfully saved Boston Housing train and test data into {n} parts in {output_dir}')

# Example usage
save_split_boston_housing_csv(3, 'boston_housing_splits_csv')