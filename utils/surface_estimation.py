
import gc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm

import cv2
import gpytorch
import torch

from joblib import Parallel, delayed

from scipy.optimize import curve_fit
from scipy.ndimage import binary_dilation, binary_erosion
from sklearn.preprocessing import minmax_scale


""" Binary Mask Construction """

def get_masks(sample,
              segmentation_model, 
              plot: bool = False,
              verbose: bool = False):
    """ 
    
    """
    if verbose:
        print("Constructing Binary Masks...")

    image_R = sample[0]
    image_S = sample[1]
    image_T = sample[2]

    # Cell segmentation can be performed on a combined image, averaging all channels in
    # reflected, scattered and transmitted images
    combined_image = minmax_scale(image_R.mean(0)-image_S.mean(0)+image_T.mean(0))
    
    # Segment cells
    mask, _, _, _ = segmentation_model.eval(combined_image, channels=[0, 0], diameter=35)

    # Background mask
    background_mask = np.where(mask == 0, 1, 0)

    # Dilude background mask a bit
    background_mask = binary_erosion(background_mask, iterations=10)

    # Compute the average value of the background and filter out pixels deviating to 
    # remove pixels the segmentation missed. Assumes that background is quite uniform. 
    intensity_deviation_threshold = -0.15
    background_pixels = combined_image[background_mask == 1]
    mean_intensity = np.mean(background_pixels)
    deviation_mask = (combined_image - mean_intensity) < intensity_deviation_threshold
    background_mask[deviation_mask] = 0

    # Compute local mean for the background
    intensity_deviation_threshold = 0
    local_mean = cv2.blur(combined_image * background_mask, (32, 32))
    local_count = cv2.blur(background_mask.astype(float), (32, 32))
    local_mean[local_count > 0] /= local_count[local_count > 0]
    deviation_mask = (combined_image - local_mean) < intensity_deviation_threshold
    background_mask[deviation_mask] = 0
    
    # Cell mask
    cell_mask = np.where(mask != 0, 1, 0)
    cell_mask = binary_dilation(cell_mask, iterations=5)

    # Plot combined image and masks
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        colormap = plt.cm.jet
        colormap.set_under(color='black')
        norm = Normalize(vmin=0.5, vmax=mask.max())
        axes[0].imshow(combined_image, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Combined Image")
        axes[0].axis(False)
        masked_mask = np.ma.masked_where(mask == 0, mask)
        axes[1].imshow(combined_image, cmap="gray", vmin=0, vmax=1)
        axes[1].imshow(masked_mask, cmap=colormap, norm=norm, alpha=0.3)
        axes[1].set_title("Cell Segmentation")
        axes[1].axis(False)
        plt.suptitle("Cell Segmentation", fontsize=24)
        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(combined_image, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Combined Image")
        axes[0].axis(False)
        background_image = combined_image.copy()
        background_image[background_mask == 0] = 0
        axes[1].imshow(background_image, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Isolated Background")
        axes[1].axis(False)
        plt.suptitle("Background Mask", fontsize=24)
        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(combined_image, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Combined Image")
        axes[0].axis(False)
        cell_image = combined_image.copy()
        cell_image[cell_mask == 0] = 0
        axes[1].imshow(cell_image, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Isolated Cells")
        axes[1].axis(False)
        plt.suptitle("Cell Mask", fontsize=24)
        plt.tight_layout()
        plt.show()

    return background_mask, cell_mask


""" Background Surface Estimation """

def fit_polynomial_background(sample: np.ndarray, 
                              mask: np.ndarray, 
                              degree: int = 2,
                              verbose: bool = False):
    """
    Fit a polynomial background to an image considering only background pixels indicated by a binary mask. 
    
    Parameters:
        image (ndarray): Input image of shape (C, H, W) or (H, W). Pixel values should be floats. 
        mask (ndarray): Binary mask of shape (H, W), where `0` indicates background pixels.
        degree (int): Degree of the polynomial. Currently only support degree 1, 2, or 3. 
    
    Returns:
        tuple: Normalized image and fitted background.
    """

    def poly1d(x, y, p):
        """ Evaluate a 1D polynomial """
        return p[0] + p[1]*x + p[2]*y

    def poly2d(x, y, p):
        """ Evaluate a 2D polynomial """
        return p[0] + p[1]*x + p[2]*y + p[3]*x**2 + p[4]*y**2 + p[5]*x*y

    def poly3d(x, y, p):
        """ Evaluate a 3D polynomial """
        return p[0] + p[1]*x + p[2]*y + p[3]*x**2 + p[4]*y**2 + p[5]*x*y + p[6]*x**3 + p[7]*y**3 + p[8]*x**2*y + p[9]*x*y**2

    # TODO: Fix shape assert
    # # Check format of inputs
    # if len(image.shape) == 2:
    #     image = np.expand_dims(image, 0)
    # assert len(image.shape) == 3, \
    #     f"Error: Invalid image shape. Got shape {image.shape}, but require shape on form (C, H, W). "
    # assert mask.shape == image.shape[1:], \
    #     f"Error: Mask shape does not match image shape. Got mask shape {mask.shape}, but require shape on form (C, H, W). "
    if verbose: 
        print(f"Fitting Background Surface...")

    # Set polynomial degree and initial guess
    if degree == 1:
        poly = poly1d
        p0 = [0]*3
    elif degree == 2:
        poly = poly2d
        p0 = [0]*6
    elif degree == 3:
        poly = poly3d
        p0 = [0]*10
    else:
        raise ValueError("Only polynomials of degree 1, 2, and 3 are currently supported.")

    # Help function to fit background per channel
    def fit_polynomial_background_(channel, mask):
        H, W = channel.shape
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        x = x.ravel()
        y = y.ravel()
        z = channel.ravel()
        m = mask.ravel()

        # Filter data based on mask
        x = x[m == 1]
        y = y[m == 1]
        z = z[m == 1]

        # Fit 2D polynomial
        opt_params, _ = curve_fit(lambda data, *p: poly(data[0], data[1], p), (x, y), z, p0=p0)

        # Generate the fitted background
        fitted_channel_background = poly(
            np.meshgrid(np.arange(W), np.arange(H))[0], 
            np.meshgrid(np.arange(W), np.arange(H))[1], 
            opt_params
        )

        # Normalize channel by subracting the estimated background
        normalized_channel = channel - fitted_channel_background

        return normalized_channel, fitted_channel_background
    
    # Allocate normalized sample and fitted background
    background_corrected_sample = np.zeros_like(sample)
    fitted_background = np.zeros_like(sample)

    # Normalize sample channels by fitting polynomial background plane
    results = Parallel(n_jobs=-1)(
        delayed(lambda i, c: (i, c, *fit_polynomial_background_(sample[i, c], mask)))(i, c) 
        for i in range(sample.shape[0]) 
        for c in range(sample.shape[1])
    )

    # Assign results to the original arrays
    for i, c, bg_corrected, fitted_bg in results:
        background_corrected_sample[i, c] = bg_corrected
        fitted_background[i, c] = fitted_bg

    return background_corrected_sample, fitted_background


""" Red Blood Cell Intensity Magnitude Surface Estimation """

def fit_gpr_surface(sample: np.ndarray, 
                    mask: np.ndarray, 
                    sample_fraction: float = 0.001,
                    verbose: bool = False):
    
    class SparseGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(SparseGPModel, self).__init__(train_x, train_y, likelihood)

            # Constant mean function
            self.mean_module = gpytorch.means.ConstantMean()

            # RBF kernel covariance function
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )

            # Set fixed lengthscale
            self.covar_module.base_kernel.lengthscale = torch.tensor(200.0).cuda()

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def sample_local_average_magnitude(channel, mask, sample_fraction):
        channel = abs(channel)
        H, W = channel.shape
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        x = x.ravel()
        y = y.ravel()
        z = channel.ravel()
        m = mask.ravel()

        # Filter by mask
        x_mask = x[m == 1]
        y_mask = y[m == 1]
        z_mask = z[m == 1]

        # Sample cell pixels to compute local average from
        num_cell_samples = int(sample_fraction * 10 * len(x_mask))
        idx = np.random.choice(len(x_mask), size=num_cell_samples, replace=False)
        x_cell_sample = x_mask[idx]
        y_cell_sample = y_mask[idx]
        z_cell_sample = z_mask[idx]
        coord_cell_sample = np.vstack((x_cell_sample, y_cell_sample)).T

        # Sample points to compute local average to
        num_samples = int(sample_fraction * len(x))
        idx = np.random.choice(len(x), size=num_samples, replace=False)
        x_sample = x[idx]
        y_sample = y[idx]
        coord_sample = np.vstack([x_sample, y_sample]).T

        # Set neighborhood size
        # TODO: Phrase in terms of fraction of points? (num_samples)
        num_neighbors = int(50000*sample_fraction*10)

        # Generate new data points in low-density areas
        sampled_points = np.zeros((num_samples, coord_sample.shape[1]))
        sampled_values = np.zeros(num_samples)

        def sample_mean_value(i):
            distances = np.linalg.norm(coord_cell_sample - coord_sample[i], axis=1)
            nearest_idx = np.argsort(distances)[:num_neighbors]
            mean_value = np.mean(z_cell_sample[nearest_idx])
            return (coord_sample[i], mean_value)
        
        results = Parallel(n_jobs=-1)(
            delayed(sample_mean_value)(i) for i in range(num_samples)
        )

        sampled_points = np.array([result[0] for result in results])
        sampled_values = np.array([result[1] for result in results])
        
        return sampled_points, sampled_values

    def fit_gpr_channel(channel, mask):
        H, W = channel.shape
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        x = x.ravel()
        y = y.ravel()

        train_x, train_y = sample_local_average_magnitude(channel, mask, sample_fraction)
        train_x = torch.tensor(train_x).cuda()
        train_y = torch.tensor(train_y).cuda()

        # Define likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
        model = SparseGPModel(train_x, train_y, likelihood).cuda()

        # Train model
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(100):  # Number of training iterations
            optimizer.zero_grad()
            output = model(train_x)
            mll_loss = -mll(output, train_y)

            loss = mll_loss
            loss.backward()
            optimizer.step()

            if (i==50) and (loss.item() > 0):   # Reduce learning rate after 50 iterations
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        if verbose: 
            print(f"MLL Loss: {mll_loss.item():.4f}")

        model.eval()
        likelihood.eval()
        points = torch.tensor(np.vstack((x, y)).T, dtype=torch.float32).cuda()
        pred_y = torch.zeros(len(points), dtype=torch.float32, device=points.device)
        batch_size = 100000
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(0, points.size(0), batch_size):
                batch = points[i:i + batch_size]
                output = model(batch).mean
                pred_y[i:i + batch_size] = output
        pred_y = pred_y.cpu().numpy().reshape(H, W)

        # Cleanup memory
        del train_x, train_y, points, model, likelihood
        torch.cuda.empty_cache()
        gc.collect()

        return channel / pred_y, pred_y

    # Normalize image
    if verbose: 
        print(f"Fitting RBC Intensity Magnitude Surface...")
    normalized_sample = np.zeros_like(sample)
    fitted_surface = np.zeros_like(sample)
    for i in range(sample.shape[0]):
        for c in range(sample.shape[1]):
            if verbose: 
                print(f"Processing Channel: {c}")
            normalized_sample[i,c], fitted_surface[i,c] = fit_gpr_channel(sample[i,c], mask)

    return normalized_sample, fitted_surface