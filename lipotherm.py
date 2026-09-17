"""
LIPOTHERM: GDGT based sea-surface temperature calibration

This script trains the LIPOTHERM calibration on the training data and then
uses it to reconstruct SST for a down-core time series. 

Usage:
    python lipotherm.py data/example_downcore.csv

Reference:
    Kumar, V., Tiwari, M., Roy, B. (2026). LIPOTHERM: A calibration framework
    for GDGT paleothermometry based on machine learning. Geochemistry,
    Geophysics, Geosystems.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import StratifiedShuffleSplit


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

script_file = Path(__file__)
script_path = script_file.resolve()
SCRIPT_FOLDER = script_path.parent
TRAINING_FILE = SCRIPT_FOLDER / "data" / "training_data.csv"

# settings for splitting the training data into training and test sets
RANDOM_SEED = 42
TEST_FRACTION = 0.20
N_BINS = 10

# Extra-Trees model hyperparameters.

N_ESTIMATORS = 1500
MAX_DEPTH = 20
MIN_SAMPLES_SPLIT = 3
MIN_SAMPLES_LEAF = 1
MAX_FEATURES = 0.8

# bootstrap
N_BOOTSTRAP = 150
BOOTSTRAP_SEED_START = 2000

# The six isoGDGT components and the column names accepted for each one.
COMPONENTS = {
    "GDGT-0": ["fGDGT_0", "GDGT-0", "GDGT.0", "GDGT0"],
    "GDGT-1": ["fGDGT_1", "GDGT-1", "GDGT.1", "GDGT1"],
    "GDGT-2": ["fGDGT_2", "GDGT-2", "GDGT.2", "GDGT2"],
    "GDGT-3": ["fGDGT_3", "GDGT-3", "GDGT.3", "GDGT3"],
    "Cren": ["fGDGT_cren", "Cren", "Crenarchaeol", "cren"],
    "Cren'": ["fGDGT_cren_prime", "Cren'", "Cren.", "Cren_prime", "cren_prime"],
}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def get_arguments():
    parser = argparse.ArgumentParser(
        description="Reconstruct SST from isoGDGT fractional abundances using LIPOTHERM."
    )
    parser.add_argument("input", help="csv file with the six isoGDGT fractional abundances")
    parser.add_argument("--out", default=None, help="name of the output csv file")
    parser.add_argument("--n-boot", type=int, default=N_BOOTSTRAP,
                        help="number of bootstrap runs (default: 150)")
    args = parser.parse_args()
    return args


def find_column(df, possible_names):
    """Return the first name from possible_names that is a column in df."""
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def read_gdgt_fractions(df):
    """Get the six GDGT fractions from a table, fix zeros and scale rows to 1."""
    column_names = []
    missing = []

    for component in COMPONENTS:
        possible_names = COMPONENTS[component]
        column = find_column(df, possible_names)
        if column is None:
            missing.append(component)
        else:
            column_names.append(column)

    if len(missing) > 0:
        print("Could not find these GDGT components in the input file:")
        for component in missing:
            accepted = COMPONENTS[component]
            accepted_text = ", ".join(accepted)
            print(f"  {component}  (accepted column names: {accepted_text})")
        columns_text = ", ".join(df.columns)
        print(f"Columns found in the file: {columns_text}")
        sys.exit(1)

    selected = df[column_names]
    X = selected.to_numpy(dtype=float)

    missing_mask = np.isnan(X)
    if missing_mask.any():
        print("The input file has missing values.")
        print("All six GDGT components are needed for every sample.")
        sys.exit(1)


    # 
    n_columns = X.shape[1]
    for col in range(n_columns):
        is_zero = X[:, col] <= 0
        if is_zero.any():
            non_zero_values = X[~is_zero, col]
            smallest = non_zero_values.min()
            X[is_zero, col] = smallest / 2.0

    #
    row_sums = X.sum(axis=1, keepdims=True)
    X = X / row_sums

    return X


def pairwise_log_ratios(X):
    """15 pairwise log-ratios ln(Gi / Gj) from the six fractions."""
    n_components = X.shape[1]
    ratios = []

    for i in range(n_components):
        for j in range(i + 1, n_components):
            numerator = X[:, i]
            denominator = X[:, j]
            ratio = numerator / denominator
            log_ratio = np.log(ratio)
            ratios.append(log_ratio)

    features = np.column_stack(ratios)
    return features


def split_training_data(X, y):
    """Split into 80% training and 20% test"""
    sst_bins = pd.qcut(y, N_BINS, labels=False)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_FRACTION,
        random_state=RANDOM_SEED,
    )
    splits = splitter.split(X, sst_bins)
    train_index, test_index = next(splits)
    return train_index, test_index


def train_model(X_train, y_train, seed):
    model = ExtraTreesRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def calculate_rmse(observed, predicted):
    errors = predicted - observed
    squared_errors = errors ** 2
    mean_squared_error = np.mean(squared_errors)
    rmse = np.sqrt(mean_squared_error)
    return rmse


def bootstrap_spread(X_train, y_train, X_new, n_boot):
    """
    Resample the training data n_boot times, retrain the model and
    predict X_new.
    """
    n_train = len(y_train)
    n_new = len(X_new)
    all_predictions = np.zeros((n_boot, n_new))

    for b in range(n_boot):
        # 
        rng = np.random.RandomState(BOOTSTRAP_SEED_START + b)
        sample_index = rng.randint(0, n_train, n_train)
        X_resampled = X_train[sample_index]
        y_resampled = y_train[sample_index]

        model = train_model(X_resampled, y_resampled, seed=b)
        all_predictions[b] = model.predict(X_new)

        if (b + 1) % 25 == 0:
            print(f"  finished {b + 1} of {n_boot}")

    spread = all_predictions.std(axis=0)
    return spread


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    args = get_arguments()

    # ---- train the model --------------------------------------------

    print("Reading training data ...")
    training_data = pd.read_csv(TRAINING_FILE)
    training_fractions = read_gdgt_fractions(training_data)
    X_data = pairwise_log_ratios(training_fractions)

    sst_column = training_data["SST"]
    y_data = sst_column.to_numpy(dtype=float)

    
    train_index, test_index = split_training_data(X_data, y_data)

    X_train = X_data[train_index]
    y_train = y_data[train_index]
    X_test = X_data[test_index]
    y_test = y_data[test_index]

    print(f"Samples in training data: {len(y_data)}")
    print(f"Used to fit the model:    {len(y_train)}")
    print(f"Kept aside for testing:   {len(y_test)}")

    print("Training the model ...")
    model = train_model(X_train, y_train, seed=RANDOM_SEED)

    # ---- check the model on the test set ----------------------------

    test_predictions = model.predict(X_test)
    test_rmse = calculate_rmse(y_test, test_predictions)
    print(f"Test RMSE: {test_rmse:.2f} deg C")

    # ---- predict SST for the time series ----------------------------

    print(f"Reading {args.input} ...")
    series = pd.read_csv(args.input)
    series_fractions = read_gdgt_fractions(series)
    X_series = pairwise_log_ratios(series_fractions)

    sst = model.predict(X_series)
    print(f"Samples in time series: {len(sst)}")

    

    print(f"Running {args.n_boot} bootstrap runs (this takes a few minutes) ...")
    sigma_bootstrap = bootstrap_spread(X_train, y_train, X_series, args.n_boot)

    rmse_squared = test_rmse ** 2
    bootstrap_squared = sigma_bootstrap ** 2
    sigma_total = np.sqrt(rmse_squared + bootstrap_squared)

    upper = sst + sigma_total
    lower = sst - sigma_total

    # ---- outputs -------------------------------------------

    output = series.copy()
    output["LIPOTHERM_SST"] = np.round(sst, 3)
    output["SST_plus_1sigma"] = np.round(upper, 3)
    output["SST_minus_1sigma"] = np.round(lower, 3)

    if args.out is None:
        input_path = Path(args.input)
        input_name = input_path.stem
        output_file = SCRIPT_FOLDER / f"{input_name}_lipotherm.csv"
    else:
        output_file = Path(args.out)

    output.to_csv(output_file, index=False)

    print("Done.")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
