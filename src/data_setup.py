from typing import Tuple
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import torch


def preprocess_data(filename: str) -> torch.Tensor:
    """Function to preprocess data

    Args:
        filename (str): Filename of data to be preprocessed.

    Returns:
        torch.Tensor: preprocessed data
    """
    data = pd.read_csv(filename)

    data_t = torch.Tensor(data.values)
    data_t = torch.nn.functional.normalize(data_t)

    return data_t


def get_train_valid_test_datasets(
    train_filename: str, test_filename: str
) -> Tuple[TensorDataset, ...]:
    train_df = pd.read_csv(train_filename)
    test_df = pd.read_csv(test_filename)

    X_train, X_valid, y_train, y_valid = train_test_split(
        train_df.iloc[:, :-1].values, train_df.iloc[:, -1].values, random_state=0
    )

    X_train_np = X_train.astype(np.float32)
    y_train_np = y_train.astype(np.float32).reshape(-1, 1)
    X_valid_np = X_valid.astype(np.float32)
    y_valid_np = y_valid.astype(np.float32).reshape(-1, 1)
    X_test_np = test_df.iloc[:, :-1].values.astype(np.float32)
    y_test_np = test_df.iloc[:, -1].values.astype(np.float32).reshape(-1, 1)

    scaler = StandardScaler()
    X_train_scaled_np = scaler.fit_transform(X_train_np)
    X_valid_scaled_np = scaler.fit_transform(X_valid_np)
    X_test_scaled_np = scaler.transform(X_test_np)

    X_train_tensor = torch.tensor(X_train_scaled_np, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
    X_valid_tensor = torch.tensor(X_valid_scaled_np, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid_np, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled_np, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    return train_dataset, valid_dataset, test_dataset


def get_classification_datasets(
    train_filename: str, test_filename: str
) -> Tuple[TensorDataset, ...]:
    """Load CSV files and return TensorDatasets with int64 labels for classification.

    Args:
        train_filename: Path to the training CSV (last column is the target class).
        test_filename: Path to the test CSV.

    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset).
    """
    train_df = pd.read_csv(train_filename)
    test_df = pd.read_csv(test_filename)

    X_train, X_valid, y_train, y_valid = train_test_split(
        train_df.iloc[:, :-1].values, train_df.iloc[:, -1].values, random_state=0
    )

    X_train_np = X_train.astype(np.float32)
    y_train_np = y_train.astype(np.int64)
    X_valid_np = X_valid.astype(np.float32)
    y_valid_np = y_valid.astype(np.int64)

    X_test_np = test_df.iloc[:, :-1].values.astype(np.float32)
    y_test_np = test_df.iloc[:, -1].values.astype(np.int64)

    scaler = StandardScaler()
    X_train_scaled_np = scaler.fit_transform(X_train_np)
    X_valid_scaled_np = scaler.fit_transform(X_valid_np)
    X_test_scaled_np = scaler.transform(X_test_np)

    X_train_tensor = torch.tensor(X_train_scaled_np, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.int64)
    X_valid_tensor = torch.tensor(X_valid_scaled_np, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid_np, dtype=torch.int64)
    X_test_tensor = torch.tensor(X_test_scaled_np, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_np, dtype=torch.int64)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    return train_dataset, valid_dataset, test_dataset
