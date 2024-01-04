import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from skimage import morphology


def plot_2_images(image1, image2, titles, cmaps=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    images = [image1, image2]

    if cmaps is None:
        cmaps = ["gray", "gray"]

    for i in range(2):
        axes[i].imshow(images[i], cmap=cmaps[i])
        axes[i].set_title(titles[i])

    plt.tight_layout()
    plt.show()


def plot_three_images(
    image1: np.ndarray,
    image2: np.ndarray,
    image3: np.ndarray,
    cmap: str = "gray",
    titles: list[str] = None,
):
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    ax[0].imshow(image1, cmap=cmap)
    ax[0].axis("off")
    ax[1].imshow(image2, cmap=cmap)
    ax[1].axis("off")
    ax[2].imshow(image3, cmap=cmap)
    ax[2].axis("off")

    if titles is not None:
        for idx, title in enumerate(titles):
            ax[idx].set_title(title)

    plt.tight_layout()
    plt.show()


def display_images(original_image, segmentation_image, grey_image, mask, title):
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    axes[0].imshow(original_image)
    axes[0].set_title(
        f"Original Image \nBild Dimension: {original_image.height} x {original_image.width}"
    )
    axes[1].imshow(segmentation_image)
    axes[1].set_title(title)
    axes[2].imshow(grey_image, cmap="gray")
    axes[2].set_title(
        f"Grey Scale Segmentation Image \nBild Dimension: {grey_image.height} x {grey_image.width}"
    )
    axes[3].imshow(mask)
    axes[3].set_title(
        f"Segmentation Mask \nBild Dimension: {mask.shape[0]} x {mask.shape[1]}"
    )
    plt.show()


def apply_color_threshold(
    image, threshold_channel_r, threshold_channel_g, threshold_channel_b
):
    filtered_image = image.copy()

    filtered_image[filtered_image[:, :, 0] < threshold_channel_r] = 0
    filtered_image[filtered_image[:, :, 1] < threshold_channel_g] = 0
    filtered_image[filtered_image[:, :, 2] < threshold_channel_b] = 0

    return filtered_image


def threshold_filter(
    image_path, export_path, threshold_r, threshold_g, threshold_b, save_image=True
):
    image = Image.open(image_path)
    image_array = np.array(image)

    filter_image = image_array.copy()
    filter_image[filter_image[:, :, 0] < threshold_r] = 0
    filter_image[filter_image[:, :, 1] < threshold_g] = 0
    filter_image[filter_image[:, :, 2] < threshold_b] = 0

    image_title = f"r-{threshold_r}_g-{threshold_g}_b-{threshold_b}"
    export_path = f"{export_path}pokemon-image-thresh-{image_title}.jpg"

    if save_image:
        filter_image = Image.fromarray(filter_image)
        filter_image.save(export_path)


def threshold_filter_all_colors(
    export_path, import_path, range_r=(0, 1), range_g=(0, 1), range_b=(0, 1)
):
    if not os.listdir(export_path):
        for r in range(range_r[0], range_r[1]):
            for g in range(range_g[0], range_g[1]):
                for b in range(range_b[0], range_b[1]):
                    threshold_filter(import_path, export_path, r, g, b, save_image=True)
    else:
        print("Files already exist in the folder.")


def convert_to_grayscale(image):
    return image.convert("L")


def extract_object(orginal_image, mask):
    mask[mask > 0] = 1
    if mask.ndim == 2:
        mask_3_channels = np.dstack((mask, mask, mask))
    else:
        mask_3_channels = mask
    extracted_object = orginal_image * mask_3_channels
    return extracted_object


def skeleton(image: np.array):
    skeleton = morphology.skeletonize(image)
    return skeleton


def n_pixel_in_mask(binary_mask: np.array, relativ=False, return_value=False) -> int:
    total_pixel = binary_mask.size
    print(f"Total number of pixel: {total_pixel}")
    if relativ:
        rel_pixel = np.round(np.sum(binary_mask) / binary_mask.size, 2)
        print(f"Relation of pixel in the segmentation mask: {rel_pixel}")
    abs_pixel = np.sum(binary_mask)
    print(f"Number of pixel in the segmentation mask: {abs_pixel}")
    if return_value:
        return total_pixel, abs_pixel, rel_pixel
