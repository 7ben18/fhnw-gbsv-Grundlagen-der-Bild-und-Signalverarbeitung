import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def slice_signal(data, slice_length, random_seed=42):
    signal_length = len(data)
    np.random.seed(random_seed)
    start_position = np.random.randint(0, signal_length - slice_length)
    sliced_signal = data.iloc[start_position : start_position + slice_length]
    return sliced_signal


def plot_data_and_correlation(data, sliced_data, data_name, single_row=True):
    corr = signal.correlate(data["signal"], sliced_data["signal"], mode="same")
    corr_norm = corr / np.max(corr)

    corr_norm_df = pd.DataFrame({"lags": corr_norm}).reset_index()
    corr_norm_df.columns = ["lags", "corr_norm"]

    if single_row:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    else:
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))

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

    axes[1].plot(corr, label="Crosscorrelation", color="blue")
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Correlation")
    axes[1].set_title("Correlation")
    axes[1].legend()

    axes[2].plot(corr_norm, label="Normalized Crosscorrelation", color="blue")
    axes[2].set_xlabel("Lag")
    axes[2].set_ylabel("Normalized Crosscorrelation")
    axes[2].set_title("Normalized Crosscorrelation")
    axes[2].legend()

    plt.suptitle(f"Orginal vs. {data_name} Kreuzkorrelation", fontsize=16)

    if single_row:
        plt.tight_layout()
    else:
        plt.subplots_adjust(hspace=0.5)

    plt.show()


class SignalProcessor:
    def __init__(self, data):
        self.data = data

    def _copy_data(self):
        return self.data.copy()

    def addition_noise(self, noise=0.25):
        modified_data = self._copy_data()
        modified_data['signal'] = modified_data['signal'] + noise
        return modified_data

    def multiply_noise(self, noise=0.25):
        modified_data = self._copy_data()
        modified_data['signal'] = modified_data['signal'] * noise
        return modified_data

    def add_random_noise(self, noise_amplitude=0.25, random_seed=42):
        np.random.seed(random_seed)
        noise = noise_amplitude * np.random.normal(size=len(self.data), scale=noise_amplitude)
        modified_data = self._copy_data()
        modified_data['signal'] = modified_data['signal'] + noise
        return modified_data

    def multiply_random_noise(self, noise_amplitude=0.25, random_seed=42):
        np.random.seed(random_seed)
        noise = noise_amplitude * np.random.normal(size=len(self.data), scale=noise_amplitude)
        modified_data = self._copy_data()
        modified_data['signal'] = modified_data['signal'] * noise
        return modified_data

    def standardize_signal(self):
        modified_data = self._copy_data()
        modified_data['signal'] = (modified_data['signal'] - modified_data['signal'].mean()) / modified_data['signal'].std()
        return modified_data

    def normalize_min_max_signal(self):
        modified_data = self._copy_data()
        modified_data['signal'] = (modified_data['signal'] - modified_data['signal'].min()) / (modified_data['signal'].max() - modified_data['signal'].min())
        return modified_data

    def shuffle_signal(self, random_seed=42):
        np.random.seed(random_seed)
        modified_data = self._copy_data()
        modified_data['signal'] = np.random.permutation(modified_data['signal'].values)
        return modified_data
    
    def reverse_signal(self):
        modified_data = self._copy_data()
        modified_data['signal'] = modified_data['signal'].values[::-1]
        return modified_data

    def log_transform_signal(self):
        modified_data = self._copy_data()
        # make sure that the signal is positive with abs
        modified_data['signal'] = np.abs(modified_data['signal'])
        modified_data['signal'] = np.log(modified_data['signal'])
        return modified_data