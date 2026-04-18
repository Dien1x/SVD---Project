import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

def get_digit_data(number, file_path, set_type, tunicate = False):
    '''
    Load the data for a specific digit from the Excel file.
    Parameters:
    - number: The digit to load (0-9)
    - file_path: Path to the Excel file
    - set_type: 'train' or 'test' to specify which sheet to load
    - tunicate: If True, transpose the result to (num_images, 256)
    Returns:
    - number: The digit that was loaded
    - selected: The loaded data for the specified digit
    '''

    # Load data
    df = pd.read_excel(file_path, sheet_name=f"a{set_type}", header=None)
    labels = pd.read_excel(file_path, sheet_name=f"d{set_type}", header=None).values.flatten()

    # Convert to numpy
    X = df.values  # shape: (256, N)

    # Find matching columns, create a boolean array where True indicates the column corresponds to the desired number
    # a numpy thing: we can directly compare the labels array to the number, resulting in a boolean array
    mask = labels == number

    if not mask.any():
        raise ValueError(f"Number {number} not found in dataset")

    # Select only those columns
    selected = X[:, mask]  # shape: (256, num_selected)

    # Transpose to (num_images, 256) → better format
    if tunicate:
        selected = selected.T

    return number, selected

def select_u_columns_by_kappa(
    digit,
    k,
    file_path="data.xlsx",
    set_type="zip",
    show_sigma_plot=False,
):
    """
    Load one digit's samples, run SVD, and select the first k columns of U.

    Parameters
    ----------
    digit : int
        Digit class to load (e.g., 0-9).
    k : int
        Number of leading U columns to keep.
    file_path : str
        Path to the Excel data file.
    set_type : str
        Dataset split key used by get_digit_data (e.g., "zip").
    show_sigma_plot : bool
        If True, plot the singular values and mark selected k.

    Returns
    -------
    k : int
        Number of selected U columns.
    U_selected : np.ndarray
        Matrix with the selected leading U columns, shape (features, k).
    """

    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")

    _, array = get_digit_data(digit, file_path, set_type, tunicate=False)
    U, S, Vt = np.linalg.svd(array, full_matrices=False)

    max_k = U.shape[1]
    if k > max_k:
        raise ValueError(f"k={k} is too large. Maximum allowed for this digit is {max_k}.")

    U_selected = U[:, :k]

    if show_sigma_plot:
        plt.figure(figsize=(6, 4))
        plt.plot(np.arange(1, len(S) + 1), S, marker="o", linewidth=1)
        plt.axvline(k, color="g", linestyle="--", label=f"k={k}")
        plt.title(f"Digit {digit} singular values")
        plt.xlabel("Index")
        plt.ylabel("Sigma")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return k, U_selected

def select_u_columns_by_sigma(
    digit,
    file_path="data.xlsx",
    set_type="zip",
    energy_threshold=0.9,
    show_sigma_plot=False,
):
    """
    Load one digit's samples, run SVD, and select the leading columns of U.

    Parameters
    ----------
    digit : int
        Digit class to load (e.g., 0-9).
    file_path : str
        Path to the Excel data file.
    set_type : str
        Dataset split key used by get_digit_data (e.g., "zip").
    energy_threshold : float
        Fraction of singular-value energy to keep in [0, 1].
    show_sigma_plot : bool
        If True, plot the singular values and cumulative energy.

    Returns
    -------
    k : int
        Number of selected U columns.
    U_selected : np.ndarray
        Matrix with the selected leading U columns, shape (features, k).
    """

    if not (0 < energy_threshold <= 1):
        raise ValueError("energy_threshold must be in (0, 1].")

    _, array = get_digit_data(digit, file_path, set_type, tunicate=False)
    U, S, Vt = np.linalg.svd(array, full_matrices=False)

    # energy of each singular value, cause the energy is proportional to the square of the singular values (information content)
    sigma_energy = S ** 2 
    
    # an array of the cumulative energy ratio, where each element is the sum of the first i singular value energies divided by the total energy
    # normalized cumulative energy, so we can compare it to the energy_threshold
    cumulative_energy = np.cumsum(sigma_energy) / np.sum(sigma_energy)

    # find the smallest k such that the cumulative energy is greater than or equal to the threshold
    k = int(np.searchsorted(cumulative_energy, energy_threshold) + 1)
    U_selected = U[:, :k]

    if show_sigma_plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # first plot: singular values distribution
        axes[0].plot(np.arange(1, len(S) + 1), S, marker="o", linewidth=1)
        axes[0].set_title(f"Digit {digit} singular values")
        axes[0].set_xlabel("Index")
        axes[0].set_ylabel("Sigma")
        axes[0].grid(alpha=0.3)

        # second plot: cumulative energy with threshold and selected k
        axes[1].plot(np.arange(1, len(S) + 1), cumulative_energy, marker="o", linewidth=1)
        axes[1].axhline(energy_threshold, color="r", linestyle="--", label=f"threshold={energy_threshold}")
        axes[1].axvline(k, color="g", linestyle="--", label=f"k={k}")
        axes[1].set_title("Cumulative energy")
        axes[1].set_xlabel("k")
        axes[1].set_ylabel("Energy ratio")
        axes[1].set_ylim(0, 1.02)
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    return k, U_selected

def build_digit_u_dictionary_from_0_to_9(
    file_path="data.xlsx",
    set_type="zip",
    optimized = True,
    energy_threshold=0.9,
    k=5,
):
    """
    Build a dictionary of U-bases for digits 0-9.

    Returns
    -------
    dict
        {
            digit: {
                "k": int,
                "U_cols": np.ndarray  # shape (256, k)
            }
        }
    """
    digit_u_dict = {}

    for digit in range(10):
        if optimized:
            selected_k, U_cols = select_u_columns_by_sigma(
                digit=digit,
                file_path=file_path,
                set_type=set_type,
                energy_threshold=energy_threshold,
                show_sigma_plot=False,
            )
        else:
            selected_k, U_cols = select_u_columns_by_kappa(
                digit=digit,
                k=k,
                file_path=file_path,
                set_type=set_type,
                show_sigma_plot=False,
            )

        digit_u_dict[digit] = {"k": selected_k, "U_cols": U_cols}

    return digit_u_dict


def predict_digit_by_relative_residual(unknown_image, digit_u_dict):
    """
    Classify an unknown image by minimum relative residual to each digit subspace.

    Parameters
    ----------
    unknown_image : np.ndarray
        Input image as shape (256,) or (16,16).
    digit_u_dict : dict
        Output from build_digit_u_dictionary().

    Returns
    -------
    int
        Predicted digit in [0, 9].
    """
    if not digit_u_dict:
        raise ValueError("digit_u_dict is empty. Build/load a dictionary with entries for digits 0-9.")

    x = np.asarray(unknown_image, dtype=float).reshape(-1)

    if x.size != 256:
        raise ValueError(f"unknown_image must have 256 values, got {x.size}.")

    x_norm = np.linalg.norm(x)
    if x_norm == 0:
        raise ValueError("unknown_image norm is zero; cannot compute relative residual.")

    best_digit = None
    best_score = np.inf

    for digit, info in digit_u_dict.items():
        U = info["U_cols"]

        # Least-squares projection: min_c ||x - Uc||_2
        coeffs, _, _, _ = np.linalg.lstsq(U, x, rcond=None)
        x_proj = U @ coeffs

        residual = x - x_proj
        relative_residual = np.linalg.norm(residual) / x_norm

        if relative_residual < best_score:
            best_score = relative_residual
            best_digit = digit

    return int(best_digit)

def evaluate_on_test_set(
    digit_u_dict,
    file_path="data.xlsx",
    set_type="test",
):
    """
    Evaluate classifier on the full test set and return confusion matrix + accuracy.

    Rows = true digit, Columns = predicted digit.
    """
    confusion = np.zeros((10, 10), dtype=int)

    for true_digit in range(10):
        _, samples = get_digit_data(true_digit, file_path, set_type, tunicate=True)

        for sample in samples:
            pred_digit = predict_digit_by_relative_residual(sample, digit_u_dict)
            confusion[true_digit, pred_digit] += 1

    total = confusion.sum()
    correct = np.trace(confusion)
    accuracy = (correct / total) * 100 if total > 0 else 0.0

    return confusion, accuracy


def print_results(confusion, accuracy):
    row_labels = [f"true_{d}" for d in range(10)]
    col_labels = [f"pred_{d}" for d in range(10)]

    df_conf = pd.DataFrame(confusion, index=row_labels, columns=col_labels)

    # Add per-digit totals and per-digit correct percentage
    row_totals = confusion.sum(axis=1)
    per_digit_acc = np.divide(
        np.diag(confusion),
        row_totals,
        out=np.zeros_like(row_totals, dtype=float),
        where=row_totals != 0,
    ) * 100

    df_conf["total"] = row_totals
    df_conf["correct_%"] = np.round(per_digit_acc, 2)

    print("Confusion table (rows=true digit, columns=predicted digit):")
    print(df_conf.to_string())
    print(f"\nOverall accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    for i in range(2, 31, 2):
        u_dict = build_digit_u_dictionary_from_0_to_9(optimized=False, k=i)
        # 'wb' means write-binary
        with open(f"trained_models/trained_u_dict_k_{i}.pkl", "wb") as file:
            pickle.dump(u_dict, file)

        print(f"Dictionary {i} has been pickled!")

    for i in range(75, 98, 1):
        u_dict = build_digit_u_dictionary_from_0_to_9(optimized=True, energy_threshold=i/100)
        # 'wb' means write-binary
        with open(f"trained_models/trained_u_dict_optimized_{i/100:.2f}.pkl", "wb") as file:
            pickle.dump(u_dict, file)

        print(f"Optimized Dictionary for threshold {i/100:.2f} has been pickled!")

# ===== New optimization utilities (added without changing existing functions) =====

def load_digit_dataset_from_excel(file_path="data.xlsx", set_type="zip"):
    """
    Load one split from Excel and return a dictionary: digit -> matrix (256, n_samples).
    """
    df = pd.read_excel(file_path, sheet_name=f"a{set_type}", header=None)
    labels = pd.read_excel(file_path, sheet_name=f"d{set_type}", header=None).values.flatten()

    X = df.values  # (256, N)
    dataset = {}
    for digit in range(10):
        mask = labels == digit
        dataset[digit] = X[:, mask]

    return dataset


def split_train_validation_stratified(
    file_path="data.xlsx",
    train_set_type="zip",
    val_ratio=0.5,
    random_state=42,
):
    """
    Stratified split on the training split.

    Returns
    -------
    train_sub : dict
        digit -> matrix (256, n_train_sub_samples)
    val_set : dict
        digit -> matrix (256, n_val_samples)
    """
    if not (0 < val_ratio < 1):
        raise ValueError("val_ratio must be in (0, 1).")

    rng = np.random.default_rng(random_state)
    full_train = load_digit_dataset_from_excel(file_path=file_path, set_type=train_set_type)

    train_sub = {}
    val_set = {}

    for digit in range(10):
        Xd = full_train[digit]  # (256, n)
        n = Xd.shape[1]
        if n == 0:
            raise ValueError(f"No samples found for digit {digit} in split '{train_set_type}'.")

        perm = rng.permutation(n)
        n_val = int(round(n * val_ratio))

        # keep both sets non-empty when possible
        if n > 1:
            n_val = max(1, min(n - 1, n_val))
        else:
            n_val = 1

        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        train_sub[digit] = Xd[:, train_idx]
        val_set[digit] = Xd[:, val_idx]

    return train_sub, val_set


def precompute_digit_svd_bases(dataset_by_digit):
    """
    Precompute SVD per digit from an in-memory dataset dictionary.

    Parameters
    ----------
    dataset_by_digit : dict
        digit -> matrix (256, n_samples)

    Returns
    -------
    svd_cache : dict
        digit -> {"U": U, "S": S, "max_k": max_k}
    """
    svd_cache = {}

    for digit in range(10):
        if digit not in dataset_by_digit:
            raise ValueError(f"dataset_by_digit is missing digit key {digit}.")

        Xd = np.asarray(dataset_by_digit[digit], dtype=float)
        if Xd.ndim != 2 or Xd.shape[0] != 256:
            raise ValueError(
                f"Digit {digit} matrix must have shape (256, n_samples), got {Xd.shape}."
            )
        if Xd.shape[1] == 0:
            raise ValueError(f"Digit {digit} has no samples.")

        U, S, Vt = np.linalg.svd(Xd, full_matrices=False)
        svd_cache[digit] = {"U": U, "S": S, "max_k": U.shape[1]}

    return svd_cache


def build_k_map_from_energy_threshold(svd_cache, energy_threshold=0.9):
    """
    Build per-digit k map from cumulative singular-value energy threshold.
    """
    if not (0 < energy_threshold <= 1):
        raise ValueError("energy_threshold must be in (0, 1].")

    k_map = {}
    for digit in range(10):
        if digit not in svd_cache:
            raise ValueError(f"svd_cache is missing digit key {digit}.")

        S = svd_cache[digit]["S"]
        sigma_energy = S ** 2
        cumulative_energy = np.cumsum(sigma_energy) / np.sum(sigma_energy)
        k = int(np.searchsorted(cumulative_energy, energy_threshold) + 1)

        max_k = int(svd_cache[digit]["max_k"])
        k_map[digit] = max(1, min(k, max_k))

    return k_map


def build_u_dictionary_from_svd_cache(svd_cache, k_map):
    """
    Build digit_u_dict in the same format your classifier already uses.
    """
    digit_u_dict = {}

    for digit in range(10):
        if digit not in svd_cache:
            raise ValueError(f"svd_cache is missing digit key {digit}.")
        if digit not in k_map:
            raise ValueError(f"k_map is missing digit key {digit}.")

        U = svd_cache[digit]["U"]
        max_k = int(svd_cache[digit]["max_k"])
        k = int(k_map[digit])
        if k < 1:
            raise ValueError(f"k for digit {digit} must be >= 1.")

        k = min(k, max_k)
        digit_u_dict[digit] = {"k": k, "U_cols": U[:, :k]}

    return digit_u_dict


def evaluate_on_dataset(digit_u_dict, dataset_by_digit):
    """
    Generalized evaluator over any in-memory dataset dictionary.

    Parameters
    ----------
    digit_u_dict : dict
        digit -> {"k": int, "U_cols": np.ndarray}
    dataset_by_digit : dict
        digit -> matrix (256, n_samples)

    Returns
    -------
    confusion : np.ndarray
        Shape (10, 10), rows=true digits, cols=predicted digits.
    accuracy : float
        Overall accuracy in percentage.
    """
    confusion = np.zeros((10, 10), dtype=int)

    for true_digit in range(10):
        if true_digit not in dataset_by_digit:
            raise ValueError(f"dataset_by_digit is missing digit key {true_digit}.")

        Xd = np.asarray(dataset_by_digit[true_digit], dtype=float)
        if Xd.ndim != 2 or Xd.shape[0] != 256:
            raise ValueError(
                f"Digit {true_digit} matrix must have shape (256, n_samples), got {Xd.shape}."
            )

        for j in range(Xd.shape[1]):
            sample = Xd[:, j]
            pred_digit = predict_digit_by_relative_residual(sample, digit_u_dict)
            confusion[true_digit, pred_digit] += 1

    total = confusion.sum()
    correct = np.trace(confusion)
    accuracy = (correct / total) * 100 if total > 0 else 0.0

    return confusion, accuracy


def tune_k_map_coordinate_descent(
    train_sub_by_digit,
    val_by_digit,
    initial_k_map,
    k_min=1,
    k_max=45,
    max_passes=3,
    verbose=True,
):
    """
    Tune per-digit k values with coordinate descent on validation accuracy.

    Returns
    -------
    best_k_map : dict
    best_accuracy : float
    history : list[dict]
    train_svd_cache : dict
    """
    if k_min < 1 or k_max < k_min:
        raise ValueError("k range is invalid. Require 1 <= k_min <= k_max.")

    train_svd_cache = precompute_digit_svd_bases(train_sub_by_digit)

    best_k_map = {d: int(initial_k_map[d]) for d in range(10)}
    for d in range(10):
        max_allowed = min(k_max, int(train_svd_cache[d]["max_k"]))
        best_k_map[d] = max(k_min, min(best_k_map[d], max_allowed))

    best_u_dict = build_u_dictionary_from_svd_cache(train_svd_cache, best_k_map)
    _, best_accuracy = evaluate_on_dataset(best_u_dict, val_by_digit)

    history = [
        {
            "pass": 0,
            "digit": None,
            "k_map": best_k_map.copy(),
            "accuracy": float(best_accuracy),
        }
    ]

    for pass_idx in range(1, max_passes + 1):
        improved_in_pass = False

        for digit in range(10):
            max_allowed = min(k_max, int(train_svd_cache[digit]["max_k"]))
            if max_allowed < k_min:
                continue

            local_best_k = best_k_map[digit]
            local_best_accuracy = best_accuracy

            for k_candidate in range(k_min, max_allowed + 1):
                if k_candidate == best_k_map[digit]:
                    continue

                trial_k_map = best_k_map.copy()
                trial_k_map[digit] = k_candidate

                trial_u_dict = build_u_dictionary_from_svd_cache(train_svd_cache, trial_k_map)
                _, trial_accuracy = evaluate_on_dataset(trial_u_dict, val_by_digit)

                if trial_accuracy > local_best_accuracy:
                    local_best_accuracy = trial_accuracy
                    local_best_k = k_candidate

            if local_best_k != best_k_map[digit]:
                best_k_map[digit] = local_best_k
                best_accuracy = local_best_accuracy
                improved_in_pass = True
                history.append(
                    {
                        "pass": pass_idx,
                        "digit": digit,
                        "k_map": best_k_map.copy(),
                        "accuracy": float(best_accuracy),
                    }
                )
                if verbose:
                    print(
                        f"Pass {pass_idx}: updated digit {digit} k -> {local_best_k}, "
                        f"val accuracy = {best_accuracy:.2f}%"
                    )

        if verbose:
            print(f"End of pass {pass_idx}, validation accuracy = {best_accuracy:.2f}%")

        if not improved_in_pass:
            if verbose:
                print("No improvement in this pass. Stopping early.")
            break

    return best_k_map, float(best_accuracy), history, train_svd_cache
