import torch


def gamma_correction(image, gamma):
    normalized_image = image / 255.0
    gamma_corrected = torch.pow(normalized_image, gamma)

    # Denormalize to the original range [0, 255]
    # PyTorch equivalent of astype('uint8')
    gamma_corrected = (gamma_corrected * 255).clamp(0, 255).byte()

    return gamma_corrected
