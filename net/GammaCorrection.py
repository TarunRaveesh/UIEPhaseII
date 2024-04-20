import torch
import torchvision.transforms.functional as F
from PIL import Image


def gamma_correction(image, gamma):
    """Apply gamma correction to an image tensor."""
    # Convert the image tensor to a PIL image
    image_pil = F.to_pil_image(image)

    # Apply gamma correction
    image_corrected = F.adjust_gamma(image_pil, gamma)

    # Convert the corrected image back to a tensor
    image_corrected_tensor = F.to_tensor(image_corrected)

    return image_corrected_tensor
