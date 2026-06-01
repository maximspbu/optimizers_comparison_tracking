"""Dataset registry with unified loading and task-specific loaders.

Unified interface (backward-compatible):
    get_dataset(name, train_fraction, seed) -> (train_ds, val_ds, test_ds, DatasetMeta)

Task-specific interfaces (used by runner):
    load_tabular_regression(name, seed, data_dir, max_rows)
        -> (train_ds, val_ds, test_ds, n_features, X_scaler, y_scaler)

    load_openml_tabular(name, seed, data_dir)
        -> (train_ds, val_ds, test_ds, input_shape, num_classes)

    load_image_dataset(name, seed, data_dir)
        -> (train_ds, val_ds, test_ds, num_classes)
"""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from torch.utils.data import Dataset, Subset, TensorDataset

from .downloader import download_kaggle_dataset, download_zip_member, resolve_local_path

log = logging.getLogger(__name__)

TaskType = str  # "regression" | "tabular_classification" | "image_classification"


@dataclass(frozen=True)
class DatasetMeta:
    name: str
    task_type: TaskType
    input_shape: tuple[int, ...]
    num_classes: int  # 0 for regression


DatasetTuple = Tuple[Dataset, Dataset, Dataset, DatasetMeta]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = "./data"


def _split_and_scale_tabular(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    y_dtype: type,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """60/20/20 split + StandardScaler fit on train only."""
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.25, random_state=seed)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    def _tensor(arr: np.ndarray) -> torch.Tensor:
        if y_dtype is np.float32:
            return torch.tensor(arr.astype(np.float32).reshape(-1, 1))
        return torch.tensor(arr.astype(np.int64))

    return (
        TensorDataset(torch.tensor(X_train), _tensor(y_train)),
        TensorDataset(torch.tensor(X_val), _tensor(y_val)),
        TensorDataset(torch.tensor(X_test), _tensor(y_test)),
    )


def _subsample_train(
    dataset: Dataset, train_fraction: float, seed: int
) -> Dataset:
    if train_fraction >= 1.0:
        return dataset
    if train_fraction <= 0.0:
        raise ValueError(f"train_fraction must be in (0, 1], got {train_fraction}")
    n = len(dataset)  # type: ignore[arg-type]
    k = max(1, int(round(n * train_fraction)))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)[:k].tolist()
    return Subset(dataset, idx)


def _preprocess_X(X_df: pd.DataFrame) -> np.ndarray:
    """Encode categoricals with OrdinalEncoder and impute missing values."""
    cat_cols = X_df.select_dtypes(include=["category", "object"]).columns.tolist()
    num_cols = X_df.select_dtypes(include="number").columns.tolist()
    parts: list[np.ndarray] = []
    if num_cols:
        X_num = X_df[num_cols].values.astype(np.float32)
        parts.append(SimpleImputer(strategy="mean").fit_transform(X_num))
    if cat_cols:
        X_cat = X_df[cat_cols]
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        parts.append(enc.fit_transform(X_cat).astype(np.float32))
    return np.hstack(parts) if parts else np.empty((len(X_df), 0), dtype=np.float32)


# ---------------------------------------------------------------------------
# Regression – sklearn-based (legacy registry)
# ---------------------------------------------------------------------------


def _load_california(seed: int) -> DatasetTuple:
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing()
    train, val, test = _split_and_scale_tabular(data.data, data.target, seed, np.float32)
    return train, val, test, DatasetMeta("california_housing", "regression", (8,), 0)


def _load_diabetes(seed: int) -> DatasetTuple:
    from sklearn.datasets import load_diabetes

    data = load_diabetes()
    train, val, test = _split_and_scale_tabular(data.data, data.target, seed, np.float32)
    return train, val, test, DatasetMeta("diabetes", "regression", (10,), 0)


def _load_wine_quality(seed: int) -> DatasetTuple:
    from sklearn.datasets import fetch_openml

    data = fetch_openml(name="wine-quality-red", version=1, as_frame=False)
    X = np.asarray(data.data, dtype=np.float32)
    y = np.asarray(data.target, dtype=np.float32)
    train, val, test = _split_and_scale_tabular(X, y, seed, np.float32)
    return train, val, test, DatasetMeta("wine_quality", "regression", (X.shape[1],), 0)


def _load_concrete(seed: int) -> DatasetTuple:
    from sklearn.datasets import fetch_openml

    data = fetch_openml(data_id=4353, as_frame=False)
    X = np.asarray(data.data, dtype=np.float32)
    y = np.asarray(data.target, dtype=np.float32)
    train, val, test = _split_and_scale_tabular(X, y, seed, np.float32)
    return train, val, test, DatasetMeta("concrete", "regression", (X.shape[1],), 0)


# ---------------------------------------------------------------------------
# Regression – UCI downloads (notebook-style)
# ---------------------------------------------------------------------------

_REGRESSION_CONFIGS: dict[str, dict] = {
    "superconductivity": {
        "kaggle_paths": [
            "/kaggle/input/superconduct/train.csv",
            "/kaggle/input/superconductivity/train.csv",
        ],
        "download_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00464/superconduct.zip",
        "zip_member": "train.csv",
        "local_file_tpl": "{data_dir}/superconductivity_train.csv",
        "header": 0,
        "target_last": True,
    },
    "yearmsd": {
        "kaggle_paths": [
            "/kaggle/input/year-prediction-msd/YearPredictionMSD.txt",
            "/kaggle/input/yearpredictionmsd/YearPredictionMSD.txt",
        ],
        "download_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip",
        "zip_member": "YearPredictionMSD.txt",
        "local_file_tpl": "{data_dir}/YearPredictionMSD.txt",
        "header": None,
        "target_last": False,
    },
}


def _resolve_regression_path(name: str, data_dir: str) -> str:
    cfg = _REGRESSION_CONFIGS[name]
    candidates = cfg["kaggle_paths"] + [cfg["local_file_tpl"].format(data_dir=data_dir)]
    found = resolve_local_path(candidates)
    if found:
        return found
    dest = cfg["local_file_tpl"].format(data_dir=data_dir)
    return download_zip_member(cfg["download_url"], dest, cfg["zip_member"])


class _TabularDataset(Dataset):
    """Simple Dataset wrapping float X/y arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_tabular_regression(
    name: str,
    seed: int = 42,
    data_dir: str = _DEFAULT_DATA_DIR,
    max_rows: int | None = None,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[Dataset, Dataset, Dataset, int, StandardScaler, StandardScaler]:
    """Load a tabular regression dataset.

    Returns:
        (train_ds, val_ds, test_ds, n_features, X_scaler, y_scaler)
    """
    if name not in _REGRESSION_CONFIGS:
        raise ValueError(f"unknown regression dataset {name!r}; available: {list(_REGRESSION_CONFIGS)}")
    cfg = _REGRESSION_CONFIGS[name]
    path = _resolve_regression_path(name, data_dir)
    log.info("Loading %s from %s", name, path)
    df = pd.read_csv(path, header=cfg["header"])
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    if cfg["target_last"]:
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.float32)
    else:
        X = df.iloc[:, 1:].values.astype(np.float32)
        y = df.iloc[:, 0].values.astype(np.float32)

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X = X_scaler.fit_transform(X).astype(np.float32)
    y = y_scaler.fit_transform(y.reshape(-1, 1)).ravel().astype(np.float32)

    n = len(X)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    tr, va, te = idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]

    return (
        _TabularDataset(X[tr], y[tr]),
        _TabularDataset(X[va], y[va]),
        _TabularDataset(X[te], y[te]),
        X.shape[1],
        X_scaler,
        y_scaler,
    )


# ---------------------------------------------------------------------------
# Tabular classification – OpenML / local CSV (notebook-style)
# ---------------------------------------------------------------------------

_CLS_LOCAL_CSV_CANDIDATES: dict[str, list[str]] = {
    "higgs-small": [
        "/kaggle/input/higgs-small/higgs-small.csv",
        "/kaggle/input/higgs_small/higgs_small.csv",
        "/kaggle/working/higgs-small.csv",
        "./higgs-small.csv",
    ],
    "jannis": [
        "/kaggle/input/jannis/jannis.csv",
        "/kaggle/input/jannis-dataset/jannis.csv",
        "/kaggle/working/jannis.csv",
        "./jannis.csv",
    ],
    "helena": [
        "/kaggle/input/helena/helena.csv",
        "/kaggle/input/helena-dataset/helena.csv",
        "/kaggle/working/helena.csv",
        "./helena.csv",
    ],
    "adult": [
        "/kaggle/input/datasets/organizations/uciml/adult-census-income/adult.csv",
        "/kaggle/input/adult-census-income/adult.csv",
        "/kaggle/input/adult/adult.csv",
        "./adult.csv",
    ],
    "creditcard": [
        "/kaggle/input/datasets/organizations/mlg-ulb/creditcardfraud/creditcard.csv",
        "/kaggle/input/creditcardfraud/creditcard.csv",
        "/kaggle/input/creditcard-fraud-detection/creditcard.csv",
        "./creditcard.csv",
    ],
}

_CLS_OPENML_IDS: dict[str, int] = {
    "higgs-small": 44129,
    "jannis": 41168,
    "helena": 41169,
    "adult": 1590,
}

_ADULT_COLUMNS: list[str] = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education.num",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
    "native.country",
    "income",
]

_ADULT_UCI_URLS: dict[str, str] = {
    "adult.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "adult.test": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
}


def _download_file(url: str, dest_path: str, timeout: int = 120) -> str:
    if os.path.exists(dest_path):
        log.info("Cache hit: %s", dest_path)
        return dest_path
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s ...", url)
    with urllib.request.urlopen(url, timeout=timeout) as src, open(dest_path, "wb") as dst:
        dst.write(src.read())
    log.info("Saved to %s", dest_path)
    return dest_path


def _fetch_cls_via_sklearn(data_id: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import fetch_openml

    ds = fetch_openml(data_id=data_id, as_frame=True, parser="auto", n_retries=5, delay=5.0)
    X = _preprocess_X(ds.data)
    y = pd.Categorical(ds.target).codes.astype(np.int64)
    return X, y


def _fetch_cls_via_openml_pkg(data_id: int) -> tuple[np.ndarray, np.ndarray]:
    import openml

    ds = openml.datasets.get_dataset(
        data_id,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False,
    )
    X_df, y_s, _, _ = ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")
    X = _preprocess_X(X_df)
    y = pd.Categorical(y_s).codes.astype(np.int64)
    return X, y


def _load_cls_from_local_csv(name: str, data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    candidates = _CLS_LOCAL_CSV_CANDIDATES.get(name, []) + [f"{data_dir}/{name}.csv"]
    path = resolve_local_path(candidates)
    if path is None:
        raise FileNotFoundError(
            f"No local CSV for '{name}'. Checked: {candidates}"
        )
    log.info("Loaded %s from local CSV: %s", name, path)
    df = pd.read_csv(path)
    X = _preprocess_X(df.iloc[:, :-1])
    y = pd.Categorical(df.iloc[:, -1]).codes.astype(np.int64)
    return X, y


def _load_adult_from_uci(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    adult_dir = Path(data_dir) / "adult_uci"
    data_path = _download_file(_ADULT_UCI_URLS["adult.data"], str(adult_dir / "adult.data"))
    test_path = _download_file(_ADULT_UCI_URLS["adult.test"], str(adult_dir / "adult.test"))

    train = pd.read_csv(
        data_path,
        names=_ADULT_COLUMNS,
        skipinitialspace=True,
        na_values="?",
    )
    test = pd.read_csv(
        test_path,
        names=_ADULT_COLUMNS,
        skipinitialspace=True,
        na_values="?",
        comment="|",
    )
    test["income"] = test["income"].astype(str).str.rstrip(".")
    df = pd.concat([train, test], ignore_index=True)
    X = _preprocess_X(df.drop(columns=["income"]))
    y = pd.Categorical(df["income"]).codes.astype(np.int64)
    return X, y


def _load_creditcard_from_kaggle(data_dir: str, kaggle_json: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    dest_dir = Path(data_dir) / "creditcardfraud"
    csv_path = dest_dir / "creditcard.csv"
    if not csv_path.exists():
        download_kaggle_dataset(
            "mlg-ulb/creditcardfraud",
            str(dest_dir),
            kaggle_json=kaggle_json,
        )
    if not csv_path.exists():
        raise FileNotFoundError(f"creditcard.csv not found after Kaggle download: {csv_path}")
    log.info("Loaded creditcard from Kaggle CSV: %s", csv_path)
    df = pd.read_csv(csv_path)
    if "Class" not in df.columns:
        raise ValueError(f"creditcard.csv must contain 'Class' target column; got {list(df.columns)}")
    X = df.drop(columns=["Class"]).values.astype(np.float32)
    y = df["Class"].astype(np.int64).values
    return X, y


def load_openml_tabular(
    name: str,
    seed: int = 42,
    data_dir: str = _DEFAULT_DATA_DIR,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    kaggle_json: str | None = None,
) -> tuple[TensorDataset, TensorDataset, TensorDataset, int, int]:
    """Load a tabular classification dataset via verified local/direct sources.

    Returns:
        (train_ds, val_ds, test_ds, input_shape, num_classes)
    """
    try:
        X, y = _load_cls_from_local_csv(name, data_dir)
    except FileNotFoundError as e_local:
        log.info("No local CSV (%s); trying verified direct source.", e_local)
        if name == "adult":
            X, y = _load_adult_from_uci(data_dir)
        elif name == "creditcard":
            X, y = _load_creditcard_from_kaggle(data_dir, kaggle_json=kaggle_json)
        else:
            data_id = _CLS_OPENML_IDS.get(name)
            if data_id is None:
                raise ValueError(f"No verified loader or OpenML data_id for '{name}'.") from e_local
            try:
                X, y = _fetch_cls_via_sklearn(data_id)
            except Exception as e_sk:
                log.warning("sklearn failed (%s); trying openml pkg.", e_sk)
                X, y = _fetch_cls_via_openml_pkg(data_id)

    test_frac = 1.0 - train_frac - val_frac
    if test_frac <= 0:
        raise ValueError(f"train_frac + val_frac must be < 1, got {train_frac + val_frac}")
    stratify = y if len(np.unique(y)) > 1 else None
    X_tv, X_test, y_tv, y_test = train_test_split(
        X,
        y,
        test_size=test_frac,
        random_state=seed,
        stratify=stratify,
    )
    val_rel = val_frac / (train_frac + val_frac)
    stratify_tv = y_tv if len(np.unique(y_tv)) > 1 else None
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_tv,
        y_tv,
        test_size=val_rel,
        random_state=seed,
        stratify=stratify_tv,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    def _ds(Xp: np.ndarray, yp: np.ndarray) -> TensorDataset:
        return TensorDataset(torch.tensor(Xp), torch.tensor(yp))

    return (
        _ds(X_train, y_train),
        _ds(X_val, y_val),
        _ds(X_test, y_test),
        X.shape[1],
        int(y.max()) + 1,
    )


# ---------------------------------------------------------------------------
# Legacy tabular classification (sklearn-based for registry)
# ---------------------------------------------------------------------------


def _load_covtype(seed: int) -> DatasetTuple:
    from sklearn.datasets import fetch_covtype

    data = fetch_covtype()
    y = data.target.astype(np.int64) - 1
    train, val, test = _split_and_scale_tabular(data.data, y, seed, np.int64)
    return train, val, test, DatasetMeta("covertype", "tabular_classification", (data.data.shape[1],), 7)


def _load_adult_legacy(seed: int) -> DatasetTuple:
    from sklearn.datasets import fetch_openml

    data = fetch_openml(name="adult", version=2, as_frame=True)
    X = data.data.copy()
    for col in X.select_dtypes(include="category").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.fillna(0).to_numpy(dtype=np.float32)
    y = LabelEncoder().fit_transform(data.target.astype(str)).astype(np.int64)
    train, val, test = _split_and_scale_tabular(X, y, seed, np.int64)
    return train, val, test, DatasetMeta("adult", "tabular_classification", (X.shape[1],), 2)


def _load_digits(seed: int) -> DatasetTuple:
    from sklearn.datasets import load_digits

    data = load_digits()
    train, val, test = _split_and_scale_tabular(data.data, data.target.astype(np.int64), seed, np.int64)
    return train, val, test, DatasetMeta("digits", "tabular_classification", (64,), 10)


def _load_letter(seed: int) -> DatasetTuple:
    from sklearn.datasets import fetch_openml

    data = fetch_openml(name="letter", version=1, as_frame=False)
    X = np.asarray(data.data, dtype=np.float32)
    y = LabelEncoder().fit_transform(np.asarray(data.target)).astype(np.int64)
    train, val, test = _split_and_scale_tabular(X, y, seed, np.int64)
    return train, val, test, DatasetMeta("letter", "tabular_classification", (X.shape[1],), 26)


# ---------------------------------------------------------------------------
# Image classification – torchvision + local ImageFolder
# ---------------------------------------------------------------------------

_IMAGE_CONFIGS: dict[str, dict] = {
    "places365": {
        "type": "torchvision",
        "num_classes": 365,
        "img_size": 224,
    },
    "intel": {
        "type": "imagefolder",
        "num_classes": 6,
        "img_size": 224,
        "kaggle_dataset": "puneet6060/intel-image-classification",
        "train_paths": [
            "/kaggle/input/intel-image-classification/seg_train/seg_train",
            "/kaggle/input/intel-image-classification/seg_train",
        ],
        "test_paths": [
            "/kaggle/input/intel-image-classification/seg_test/seg_test",
            "/kaggle/input/intel-image-classification/seg_test",
        ],
    },
    "cifar100": {
        "type": "torchvision",
        "num_classes": 100,
        "img_size": 224,
    },
}


class _SubsetWithTransform(Dataset):
    """Wraps a Subset to apply a per-split transform."""

    def __init__(self, subset: Dataset, transform=None) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def _get_image_transforms(img_size: int = 224, augment: bool = False, dataset_name: str | None = None):
    from torchvision import transforms

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if dataset_name == "cifar100":
        if augment:
            return transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandomCrop(img_size, padding=16),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            normalize,
        ])

    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])


def load_image_dataset(
    name: str,
    seed: int = 42,
    data_dir: str = _DEFAULT_DATA_DIR,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    kaggle_json: str | None = None,
) -> tuple[Dataset, Dataset, Dataset, int]:
    """Load an image classification dataset.

    Returns:
        (train_ds, val_ds, test_ds, num_classes)
    """
    if name not in _IMAGE_CONFIGS:
        raise ValueError(f"unknown image dataset {name!r}; available: {list(_IMAGE_CONFIGS)}")

    cfg = _IMAGE_CONFIGS[name]
    img_size = cfg["img_size"]
    num_classes = cfg["num_classes"]

    if name == "places365":
        from torchvision.datasets import Places365

        Path(data_dir).mkdir(parents=True, exist_ok=True)
        full = Places365(root=data_dir, split="val", small=True, download=True, transform=None)
        n = len(full)
        n_train = int(train_frac * n)
        n_val = int(val_frac * n)
        n_test = n - n_train - n_val
        gen = torch.Generator().manual_seed(seed)
        t_sub, v_sub, te_sub = torch.utils.data.random_split(
            full, [n_train, n_val, n_test], generator=gen
        )
        return (
            _SubsetWithTransform(t_sub, _get_image_transforms(img_size, augment=True, dataset_name=name)),
            _SubsetWithTransform(v_sub, _get_image_transforms(img_size, augment=False, dataset_name=name)),
            _SubsetWithTransform(te_sub, _get_image_transforms(img_size, augment=False, dataset_name=name)),
            num_classes,
        )

    if name == "cifar100":
        from torchvision.datasets import CIFAR100

        Path(data_dir).mkdir(parents=True, exist_ok=True)
        full_train = CIFAR100(root=data_dir, train=True, download=True, transform=None)
        test_ds = CIFAR100(
            root=data_dir,
            train=False,
            download=True,
            transform=_get_image_transforms(img_size, augment=False, dataset_name=name),
        )
        n = len(full_train)
        n_t = int((1 - val_frac) * n)
        n_v = n - n_t
        gen = torch.Generator().manual_seed(seed)
        t_sub, v_sub = torch.utils.data.random_split(full_train, [n_t, n_v], generator=gen)
        return (
            _SubsetWithTransform(t_sub, _get_image_transforms(img_size, augment=True, dataset_name=name)),
            _SubsetWithTransform(v_sub, _get_image_transforms(img_size, augment=False, dataset_name=name)),
            test_ds,
            num_classes,
        )

    if name == "intel":
        from torchvision.datasets import ImageFolder

        # Build candidate paths including data_dir
        train_paths = cfg["train_paths"] + [
            f"{data_dir}/seg_train/seg_train",
            f"{data_dir}/seg_train",
            f"{data_dir}/intel-image-classification/seg_train/seg_train",
            f"{data_dir}/intel-image-classification/seg_train",
        ]
        test_paths = cfg["test_paths"] + [
            f"{data_dir}/seg_test/seg_test",
            f"{data_dir}/seg_test",
            f"{data_dir}/intel-image-classification/seg_test/seg_test",
            f"{data_dir}/intel-image-classification/seg_test",
        ]

        train_dir = resolve_local_path(train_paths)
        if train_dir is None:
            # Attempt Kaggle download
            from .downloader import download_kaggle_dataset

            log.info("Intel dataset not found; attempting Kaggle download...")
            download_kaggle_dataset(
                cfg["kaggle_dataset"],
                f"{data_dir}/intel-image-classification",
                kaggle_json=kaggle_json,
            )
            train_dir = resolve_local_path(train_paths)
            if train_dir is None:
                raise FileNotFoundError(
                    f"Intel train dir not found after download. Checked: {train_paths}\n"
                    "Upload the dataset to the data volume or provide a kaggle.json."
                )

        full = ImageFolder(train_dir, transform=None)
        n = len(full)
        n_t = int((1 - val_frac) * n)
        n_v = n - n_t
        gen = torch.Generator().manual_seed(seed)
        t_sub, v_sub = torch.utils.data.random_split(full, [n_t, n_v], generator=gen)
        train_ds = _SubsetWithTransform(t_sub, _get_image_transforms(img_size, augment=True, dataset_name=name))
        val_ds = _SubsetWithTransform(v_sub, _get_image_transforms(img_size, augment=False, dataset_name=name))

        test_dir = resolve_local_path(test_paths)
        if test_dir:
            test_ds = ImageFolder(test_dir, transform=_get_image_transforms(img_size, augment=False, dataset_name=name))
        else:
            log.warning("Intel test dir not found — reusing val split.")
            test_ds = val_ds

        return train_ds, val_ds, test_ds, num_classes

    raise ValueError(f"unhandled image dataset: {name!r}")


# ---------------------------------------------------------------------------
# Legacy vision (torchvision CIFAR / FashionMNIST)
# ---------------------------------------------------------------------------


def _load_vision(name: str, seed: int, root: str = _DEFAULT_DATA_DIR) -> DatasetTuple:
    import torchvision
    from torchvision import transforms

    Path(root).mkdir(parents=True, exist_ok=True)
    if name == "cifar10":
        cls, nc = torchvision.datasets.CIFAR10, 10
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        input_shape = (3, 32, 32)
    elif name == "cifar100":
        cls, nc = torchvision.datasets.CIFAR100, 100
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        input_shape = (3, 32, 32)
    elif name == "fashion_mnist":
        cls, nc = torchvision.datasets.FashionMNIST, 10
        normalize = transforms.Normalize((0.2860,), (0.3530,))
        input_shape = (1, 28, 28)
    else:
        raise ValueError(f"unknown vision dataset {name!r}")

    tfm = transforms.Compose([transforms.ToTensor(), normalize])
    full_train = cls(root=root, train=True, download=True, transform=tfm)
    test_ds = cls(root=root, train=False, download=True, transform=tfm)
    n = len(full_train)
    n_val = int(round(n * 0.1))
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(full_train, [n - n_val, n_val], generator=gen)
    return train_ds, val_ds, test_ds, DatasetMeta(name, "image_classification", input_shape, nc)


# ---------------------------------------------------------------------------
# Unified registry (backward-compatible)
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Callable[[int], DatasetTuple]] = {
    "california_housing": _load_california,
    "diabetes": _load_diabetes,
    "wine_quality": _load_wine_quality,
    "concrete": _load_concrete,
    "covertype": _load_covtype,
    "adult": _load_adult_legacy,
    "digits": _load_digits,
    "letter": _load_letter,
    "cifar10": lambda s: _load_vision("cifar10", s),
    "cifar100": lambda s: _load_vision("cifar100", s),
    "fashion_mnist": lambda s: _load_vision("fashion_mnist", s),
}

DATASETS_BY_TASK: dict[TaskType, tuple[str, ...]] = {
    "regression": ("superconductivity", "yearmsd", "california_housing", "diabetes"),
    "tabular_classification": ("adult", "creditcard", "higgs-small", "jannis", "helena"),
    "image_classification": ("cifar100", "intel", "places365", "cifar10"),
}

# Notebook-style dataset names per task (defaults for runner)
REGRESSION_DATASETS: tuple[str, ...] = ("superconductivity", "yearmsd")
TABULAR_CLS_DATASETS: tuple[str, ...] = ("adult", "creditcard")
IMAGE_CLS_DATASETS: tuple[str, ...] = ("cifar100", "intel")

TRAIN_FRACTIONS: tuple[float, ...] = (0.1, 0.3, 0.5, 1.0)


def get_dataset(
    name: str,
    train_fraction: float = 1.0,
    seed: int = 0,
    data_dir: str = _DEFAULT_DATA_DIR,
) -> DatasetTuple:
    """Load a dataset by name (unified registry, backward-compatible)."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown dataset {name!r}; known: {list(_REGISTRY)}")
    train_ds, val_ds, test_ds, meta = _REGISTRY[name](seed)
    train_ds = _subsample_train(train_ds, train_fraction, seed)
    return train_ds, val_ds, test_ds, meta


def list_datasets(task: TaskType | None = None) -> tuple[str, ...]:
    if task is None:
        return tuple(_REGISTRY)
    return DATASETS_BY_TASK.get(task, ())
