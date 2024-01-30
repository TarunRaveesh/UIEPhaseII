from __future__ import division
import numpy as np
import math
from scipy.ndimage import gaussian_filter

def getSSIM(X, Y):
    """
       Computes the mean structural similarity between two images.
    """
    assert (X.shape == Y.shape), "Image-patche provided have different dimensions"
    nch = 1 if X.ndim==2 else X.shape[-1]
    mssim = []
    for ch in range(nch):
        Xc, Yc = X[...,ch].astype(np.float64), Y[...,ch].astype(np.float64)
        mssim.append(compute_ssim(Xc, Yc))
    return np.mean(mssim)


def compute_ssim(X, Y):
    """
       Compute the structural similarity per single channel (given two images)
    """
    # variables are initialized as suggested in the paper
    K1 = 0.01
    K2 = 0.03
    sigma = 1.5
    win_size = 5

    # means
    ux = gaussian_filter(X, sigma)
    uy = gaussian_filter(Y, sigma)

    # variances and covariances
    uxx = gaussian_filter(X * X, sigma)
    uyy = gaussian_filter(Y * Y, sigma)
    uxy = gaussian_filter(X * Y, sigma)

    # normalize by unbiased estimate of std dev
    N = win_size ** X.ndim
    unbiased_norm = N / (N - 1)  # eq. 4 of the paper
    vx  = (uxx - ux * ux) * unbiased_norm
    vy  = (uyy - uy * uy) * unbiased_norm
    vxy = (uxy - ux * uy) * unbiased_norm

    R = 255
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2
    # compute SSIM (eq. 13 of the paper)
    sim = (2 * ux * uy + C1) * (2 * vxy + C2)
    D = (ux ** 2 + uy ** 2 + C1) * (vx + vy + C2)
    SSIM = sim/D
    mssim = SSIM.mean()

    return mssim



def getPSNR(X, Y):
    #assume RGB image
    target_data = np.array(X, dtype=np.float64)
    ref_data = np.array(Y, dtype=np.float64)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.) )
    if rmse == 0: return 100
    else: return 20*math.log10(255.0/rmse)
    
    
    
import os
import numpy as np
from PIL import Image
from glob import glob
from os.path import join, basename

def calculate_metrics(gtr_dir, gen_dir, im_res=(256, 256)):
    gtr_paths = sorted(glob(join(gtr_dir, "*.*")))
    gen_paths = sorted(glob(join(gen_dir, "*.*")))
    ssims, psnrs = [], []

    for gtr_path, gen_path in zip(gtr_paths, gen_paths):
        gtr_f = basename(gtr_path).split('.')[0]
        gen_f = basename(gen_path).split('.')[0]

        if (gtr_f == gen_f):
            r_im = Image.open(gtr_path).resize(im_res)
            g_im = Image.open(gen_path).resize(im_res)

            # Calculate SSIM on RGB channels
            ssim = getSSIM(np.array(r_im), np.array(g_im))
            ssims.append(ssim)

            # Calculate PSNR on L channel (SOTA norm)
            r_im = r_im.convert("L")
            g_im = g_im.convert("L")
            psnr = getPSNR(np.array(r_im), np.array(g_im))
            psnrs.append(psnr)

    return np.array(ssims), np.array(psnrs)

# Define directories
gtr_directory = 'clean_test'  # Ground truth directory
gen_directory = 'results'   # Generated images directory

# Calculate SSIM and PSNR
ssim_scores, psnr_scores = calculate_metrics(gtr_directory, gen_directory)

# Calculate average SSIM and PSNR
avg_ssim = np.mean(ssim_scores)
avg_psnr = np.mean(psnr_scores)

print(f"Average SSIM: {avg_ssim}")
print(f"Average PSNR: {avg_psnr}")
