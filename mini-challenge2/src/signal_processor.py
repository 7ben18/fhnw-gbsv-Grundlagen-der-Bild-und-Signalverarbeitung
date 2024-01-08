import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from statsmodels.tsa.stattools import ccf


def slice_signal(data, slice_length, random_seed=42):
    """
    Slice a signal or data series randomly.

    Args:
        data (pd.DataFrame): The input data containing a 'signal' column.
        slice_length (int): The length of the sliced signal.
        random_seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        pd.Series: The sliced signal.
    """
    signal_length = len(data)
    np.random.seed(random_seed)
    start_position = np.random.randint(0, signal_length - slice_length)
    sliced_signal = data.iloc[start_position : start_position + slice_length]
    return sliced_signal


def plot_data_and_correlation(data, sliced_data, data_name, single_row=True):
    """
    Plot original and sliced data, cross-correlation, and normalized cross-correlation.

    Args:
        data (pd.DataFrame): The original data.
        sliced_data (pd.DataFrame): The sliced data.
        data_name (str): Name or label for the data being analyzed.
        single_row (bool, optional): Whether to arrange the plots in a single row. Default is True.

    Returns:
        None
    """

    corr = ccf(sliced_data["signal"], data["signal"], adjusted=False)

    corr_df = pd.DataFrame({"lags": corr}).reset_index()
    corr_df.columns = ["lags", "corr_norm"]

    if single_row:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    else:
        fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    axes[0].plot(
        data["time"],
        data["signal"],
        label="Original Data",
        color="blue",
        linestyle="--",
        alpha=0.75,
    )
    axes[0].plot(
        sliced_data["time"], sliced_data["signal"], label="Sliced Data", color="green"
    )
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Signal")
    axes[0].set_title("Original and Sliced Data")
    axes[0].legend()

    axes[1].plot(
        corr_df["lags"],
        corr_df["corr_norm"],
        label="Normalized Crosscorrelation",
        color="blue",
    )
    axes[1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Correlation")
    axes[1].set_title("Correlation")
    axes[1].legend()

    plt.suptitle(f"Orginal vs. {data_name} Kreuzkorrelation", fontsize=16)

    if single_row:
        plt.tight_layout()
    else:
        plt.subplots_adjust(hspace=0.5)

    plt.show()


class SignalProcessor:
    """
    SignalProcessor is a class for processing and manipulating signal data.

    This class provides various methods for modifying signal data, including adding noise, multiplying by noise,
    adding random noise, multiplying by random noise, standardizing, normalizing, shuffling, reversing, and
    log-transforming the signal.

    Parameters:
    - data (pd.Series): A Pandas Series containing the signal data. The Series should have a "signal" column

    Methods:
    - addition_noise(noise=0.25): Add a constant value of noise to the signal.
    - multiply_noise(noise=0.25): Multiply the signal by a constant noise value.
    - add_random_noise(noise_amplitude=0.25, random_seed=42): Add random noise to the signal.
    - multiply_random_noise(noise_amplitude=0.25, random_seed=42): Multiply the signal by random noise.
    - standardize_signal(): Standardize the signal by subtracting the mean and dividing by the standard deviation.
    - normalize_min_max_signal(): Normalize the signal to a range of [0, 1].
    - shuffle_signal(random_seed=42): Shuffle the order of signal values.
    - reverse_signal(): Reverse the order of signal values.
    - log_transform_signal(): Apply a log transformation to the signal.

    Note: The original data is not modified by any of these methods, and a modified copy of the data
    is returned.
    """

    def __init__(self, data):
        self.data = data

    def _copy_data(self):
        return self.data.copy()

    def addition_noise(self, noise=0.25):
        modified_data = self._copy_data()
        modified_data["signal"] = modified_data["signal"] + noise
        return modified_data

    def multiply_noise(self, noise=0.25):
        modified_data = self._copy_data()
        modified_data["signal"] = modified_data["signal"] * noise
        return modified_data

    def add_random_noise(self, noise_amplitude=0.25, random_seed=42):
        np.random.seed(random_seed)
        noise = noise_amplitude * np.random.normal(
            size=len(self.data), scale=noise_amplitude
        )
        modified_data = self._copy_data()
        modified_data["signal"] = modified_data["signal"] + noise
        return modified_data

    def multiply_random_noise(self, noise_amplitude=0.25, random_seed=42):
        np.random.seed(random_seed)
        noise = noise_amplitude * np.random.normal(
            size=len(self.data), scale=noise_amplitude
        )
        modified_data = self._copy_data()
        modified_data["signal"] = modified_data["signal"] * noise
        return modified_data

    def standardize_signal(self):
        modified_data = self._copy_data()
        modified_data["signal"] = (
            modified_data["signal"] - modified_data["signal"].mean()
        ) / modified_data["signal"].std()
        return modified_data

    def normalize_min_max_signal(self):
        modified_data = self._copy_data()
        modified_data["signal"] = (
            modified_data["signal"] - modified_data["signal"].min()
        ) / (modified_data["signal"].max() - modified_data["signal"].min())
        return modified_data

    def shuffle_signal(self, random_seed=42):
        np.random.seed(random_seed)
        modified_data = self._copy_data()
        modified_data["signal"] = np.random.permutation(modified_data["signal"].values)
        return modified_data

    def reverse_signal(self):
        modified_data = self._copy_data()
        modified_data["signal"] = modified_data["signal"].values[::-1]
        return modified_data

    def log_transform_signal(self):
        modified_data = self._copy_data()
        # make sure that the signal is positive with abs
        modified_data["signal"] = np.abs(modified_data["signal"])
        modified_data["signal"] = np.log(modified_data["signal"])
        return modified_data
