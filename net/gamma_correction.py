import cv2
import numpy as np

def gamma_correction(image, gamma=1.0):
    """
    Perform gamma correction on the input image.

    Parameters:
        image (numpy.ndarray): Input image.
        gamma (float): Gamma value (default is 1.0).

    Returns:
        numpy.ndarray: Gamma-corrected image.
    """
    # Ensure gamma is a positive float
    if gamma <= 0:
        raise ValueError("Gamma value must be greater than 0.")

    # Normalize pixel values to the range [0, 1]
    normalized_image = image / 255.0

    # Apply gamma correction
    gamma_corrected = np.power(normalized_image, gamma)

    # Denormalize to the original range [0, 255]
    gamma_corrected = (gamma_corrected * 255).astype(np.uint8)

    return gamma_corrected

