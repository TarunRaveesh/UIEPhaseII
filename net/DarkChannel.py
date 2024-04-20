import cv2
import numpy as np


def estimate_dark_channel(image, patch_size=15):
    """
    Estimates the dark channel of an image.

    Args:
        image (numpy.ndarray): The input image (RGB or grayscale).
        patch_size (int, optional): Size of the local patch for calculating minima. Defaults to 15.

    Returns:
        numpy.ndarray: The estimated dark channel of the image.
    """

    # Handle grayscale images
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=2)  # Replicate as RGB for consistent processing

    # Calculate minimum across color channels
    dark_channel = np.min(image, axis=2)

    # Apply minimum filter for patch refinement
    kernel = np.ones((patch_size, patch_size))
    dark_channel_refined = cv2.erode(dark_channel, kernel)  # You'll need to import cv2

    return dark_channel_refined


def estimate_transmission_map(image, dark_channel, omega=0.95):
    """
    Estimates the transmission map based on the dark channel prior.

    Args:
        image (numpy.ndarray): The input image (RGB).
        dark_channel (numpy.ndarray): The estimated dark channel of the image.
        omega (float, optional): Parameter to refine the transmission map. Defaults to 0.95.

    Returns:
        numpy.ndarray: The estimated transmission map.
    """

    # Calculate atmospheric light (A) as top 0.1% brightest pixels in the dark channel
    num_pixels = dark_channel.shape[0] * dark_channel.shape[1]
    top_idx = int(num_pixels * 0.001)  # Top 0.1%
    A = np.max(image[np.argpartition(dark_channel.flatten(), -top_idx)[-top_idx:]])

    # Simplified transmission map calculation
    transmission_map = 1 - omega * (dark_channel / A)

    return transmission_map


def estimate_backscatter(image, transmission_map, A):
    """
    Estimates the backscatter component of an underwater image.

    Args:
        image (numpy.ndarray): The input image (RGB).
        transmission_map (numpy.ndarray): The estimated transmission map.
        A (float or tuple): The atmospheric light.

    Returns:
        numpy.ndarray: The estimated backscatter component.
    """

    backscatter = (1 - transmission_map) * A

    return backscatter 
