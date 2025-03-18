
import gc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import cv2
import gpytorch
import torch

from joblib import Parallel, delayed
from scipy.optimize import curve_fit
from scipy.ndimage import binary_dilation, binary_erosion
from sklearn.preprocessing import minmax_scale
from scipy.interpolate import bisplrep, bisplev, griddata

from .data import check_sample_format


""" Binary Mask Construction """

def get_masks(
    sample: np.ndarray,
    segmentation_model, 
    plot: bool = False,
    verbose: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """ Constructs the binary background and cell masks

    Args:
        sample: A numpy array with the sample data, format (M, C, H, W),
                M - number of modalities, C - number of channels, H - image height, W - image width.
        segmentation_model: A CellPose segmentation model.
        plot: A boolean to indicate whether to plot the masks or not. 
        verbose: A boolean to indicate whether to print detailed information during processing or not.
    
    Returns:
        tuple - Two 2D arrays, one with the background mask and one with the cell mask of shape (H, W).
    """
    if verbose:
        print("Constructing Binary Masks...")
    
    # Get modalities from sample
    image_R = sample[0]
    image_S = sample[1]
    image_T = sample[2]

    # Cell segmentation can be performed on a combined image, averaging all channels in
    # the reflectance, the inverted scattering and the transmittance images
    combined_image = minmax_scale(image_R.mean(0)+(1-image_S.mean(0))+image_T.mean(0))
    
    # Segment cells
    mask, _, _, _ = segmentation_model.eval(combined_image, channels=[0, 0], diameter=35)

    # Get background pixels
    background_mask = np.where(mask == 0, 1, 0)

    # Erode background mask a bit
    background_mask = binary_erosion(background_mask, iterations=10)

    # Compute the average value of the background and filter out pixels deviating to 
    # remove pixels the segmentation missed. Assumes that background is quite uniform. 
    intensity_deviation_threshold = -0.15
    background_pixels = combined_image[background_mask == 1]
    mean_intensity = np.mean(background_pixels)
    deviation_mask = (combined_image - mean_intensity) < intensity_deviation_threshold
    background_mask[deviation_mask] = 0

    # Compute local average value and filter out pixels with lower values to
    # remove background deformations from the background mask. 
    intensity_deviation_threshold = 0
    local_mean = cv2.blur(combined_image * background_mask, (32, 32))
    local_count = cv2.blur(background_mask.astype(float), (32, 32))
    local_mean[local_count > 0] /= local_count[local_count > 0]
    deviation_mask = (combined_image - local_mean) < intensity_deviation_threshold
    background_mask[deviation_mask] = 0
    
    # Get cell pixels
    cell_mask = np.where(mask != 0, 1, 0)

    # Dilate the mask a bit
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

def fit_polynomial_background(
    sample: np.ndarray, 
    mask: np.ndarray, 
    degree: int = 2,
    verbose: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """ Fits a polynomial background surface to a sample.

    Fits a  polynomial background surface to background pixels indicated by a background mask through 
    least squares optimization which in this case is equal to minimizing mean squared error between 
    the surface and background pixels. 
    
    Parameters:
        sample: Input sample of shape (M, C, H, W), M - number of modalities, C - number of channels, 
                H - image height, W - image width. Pixel values should be floats.
        mask: Binary mask of shape (H, W), where 1 indicates background pixels.
        degree: Degree of the polynomial. Currently only support degrees 1, 2, or 3. 
        verbose: A boolean to indicate whether to print detailed information during processing or not.
    
    Returns:
        tuple - Background-corrected sample and fitted background surfaces as numpy arrays of same size as input sample. 
    """
    if verbose: 
        print(f"Fitting Background Surface...")

    def poly1d(x, y, p):
        """ 1D polynomial """
        return p[0] + p[1]*x + p[2]*y

    def poly2d(x, y, p):
        """ 2D polynomial """
        return p[0] + p[1]*x + p[2]*y + p[3]*x**2 + p[4]*y**2 + p[5]*x*y

    def poly3d(x, y, p):
        """ 3D polynomial """
        return p[0] + p[1]*x + p[2]*y + p[3]*x**2 + p[4]*y**2 + p[5]*x*y + p[6]*x**3 + p[7]*y**3 + p[8]*x**2*y + p[9]*x*y**2

    # Check format of inputs
    sample = check_sample_format(sample, allow_multiple=False)
        
    assert mask.shape == sample.shape[2:], \
        f"Error: Mask and sample image dimensions do not match. Got mask shape {mask.shape} and sample image shape {sample.shape[2:]}."

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

    def fit_polynomial_background_(channel, mask):
        """ Help function to fit polynomial to background. """

        # Extract x, y coordinates, z values and mask values
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
        optimal_params, _ = curve_fit(lambda data, *p: poly(data[0], data[1], p), (x, y), z, p0=p0)

        # Compute background over the entire image from optimal polynomial parameters
        fitted_channel_background = poly(
            np.meshgrid(np.arange(W), np.arange(H))[0], 
            np.meshgrid(np.arange(W), np.arange(H))[1], 
            optimal_params
        )

        # Normalize channel by background subtraction
        normalized_channel = channel - fitted_channel_background
        return normalized_channel, fitted_channel_background
    
    # Allocate normalized sample and fitted background
    background_corrected_sample = np.zeros_like(sample)
    fitted_background = np.zeros_like(sample)

    # Fit polynomial planes to each channel in parallel
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

def fit_rbc_surface(
    sample: np.ndarray, 
    mask: np.ndarray, 
    method: str,
    sample_fraction: float = 0.001,
    verbose: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """ Fits a surface to the local average cell intensity magnitude. 

    Fits a surface to sampled values of the local average cell intensity magnitude. Cell pixels are indicated by the mask. 

    Implemented methods include: 
        Linear: Forms a surface by interpolating all points in the domain from the sampled subset. 
        B-Spline: Fits a smooth surface using a cubic bivariate spline representation.
        GPR: Fits a smooth surface through Gaussian process regression with a constant mean function and an RBF kernel 
        covariance function with a fixed lengthscale. 
    
    Parameters:
        sample: Input sample of shape (M, C, H, W), M - number of modalities, C - number of channels, 
                H - image height, W - image width. Pixel values should be floats.
        mask: Binary mask of shape (H, W), where 1 indicates cell pixels.
        method: Method of fitting the surface to sampled values of the local average cell intensity magnitude. Implemented
                methods are ['linear', 'b-spline', 'gpr'].
        sample_fraction: The fraction of local average cell intensity magnitudes to sample and fit the surface to.
        verbose: A boolean to indicate whether to print detailed information during processing or not.
    
    Returns:
        tuple - Normalized sample and fitted intensity magnitude surfaces as numpy arrays of same size as input sample. 
    """
    if verbose: 
        print(f"Fitting RBC Intensity Magnitude Surface...")
        
    # Check format of inputs
    sample = check_sample_format(sample, allow_multiple=False)
        
    assert mask.shape == sample.shape[2:], \
        f"Error: Mask and sample image dimensions do not match. Got mask shape {mask.shape} and sample image shape {sample.shape[2:]}."
    
    assert method.lower() in ["gpr", "b-spline", "linear"], \
        f"Error: Method '{method}' not implemented. Implemented methods are: 'linear', 'b-spline', 'gpr'."
    
    
    def sample_local_average_magnitude(
        channel: np.ndarray, 
        mask: np.ndarray, 
        sample_fraction: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """ Samples local average cell intensity magnitudes from channel.
        
        Computes the local average value by computing the distance between the point and all other points 
        and taking the average of the nearest neighborhood from a sampled subset of points in the image. 
        
        Args: 
            channel: A one channel image (H, W) from which to sample local average intensity magnitudes.
            mask: Binary mask of shape (H, W), where 1 indicates cell pixels.
            sample_fraction: The fraction of values to sample.
        
        Returns:
            tuple - Sampled points (x and y coordinates) and corresponding values of local average cell 
            intensity magnitude.
        """
        
        # Get channel magnitudes instead of intensity
        channel = abs(channel)

        # Extract x, y coordinates, z values and mask values
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

        # Sample cell pixels to compute the local average from 
        # (10 times as many as the sample fraction of the subset we are returning)
        num_cell_samples = int(sample_fraction * 10 * len(x_mask))
        idx = np.random.choice(len(x_mask), size=num_cell_samples, replace=False)
        x_cell_sample = x_mask[idx]
        y_cell_sample = y_mask[idx]
        z_cell_sample = z_mask[idx]
        coord_cell_sample = np.vstack((x_cell_sample, y_cell_sample)).T

        # Sample points to compute local average at
        num_samples = int(sample_fraction * len(x))
        idx = np.random.choice(len(x), size=num_samples, replace=False)
        x_sample = x[idx]
        y_sample = y[idx]

        # To create a convex hull that includes the entire image, we manually add the four image 
        # corners to the sampled coordinates (required by the linear interpolation method)
        num_samples += 4
        x_sample = np.concatenate([x_sample, [0, 0, W-1, W-1]])
        y_sample = np.concatenate([y_sample, [0, H-1, 0, H-1]])

        coord_sample = np.vstack([x_sample, y_sample]).T

        # Set neighborhood size
        N = 50000           # The neighborhood size in the original image (before sampling)
        num_neighbors = int(N*sample_fraction*10)   # The neighborhood size after sampling

        def sample_mean_value(i: int) -> tuple[np.ndarray, float]:
            """ Compute the local average value. """
            distances = np.linalg.norm(coord_cell_sample - coord_sample[i], axis=1)
            nearest_idx = np.argsort(distances)[:num_neighbors]
            mean_value = np.mean(z_cell_sample[nearest_idx])
            return (coord_sample[i], mean_value)
        
        # Sample local average values in parallel
        results = Parallel(n_jobs=-1)(
            delayed(sample_mean_value)(i) for i in range(num_samples)
        )

        # Collect results into two arrays for the sampled coordinates and computed values
        sampled_points = np.array([result[0] for result in results])
        sampled_values = np.array([result[1] for result in results])
        return sampled_points, sampled_values

    def fit_gpr_channel(
        channel: np.ndarray, 
        mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """ Fits a smooth surface through Gaussian process regression with a constant mean function and an RBF kernel 
        covariance function with a fixed lengthscale. 
        
        Args: 
            channel: A one channel image (H, W) from which to sample local average intensity magnitudes.
            mask: Binary mask of shape (H, W), where 1 indicates cell pixels.
        
        Returns:
            tuple - The normalized channel and estimated surface.
        """
        
        class GPModel(gpytorch.models.ExactGP):
            """ The Gaussian process model.

            Defines a Gaussian process model with a constant mean function and an RBF kernel covariance function with a
            fixed lengthscale.
            
            Attributes:
                mean_module: The mean function module.
                covar_module: The covariance function module.
            """
            def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: gpytorch.likelihoods.Likelihood):
                """ Initializes a Gaussian process model instance. """
                super(GPModel, self).__init__(train_x, train_y, likelihood)

                # Constant mean function
                self.mean_module = gpytorch.means.ConstantMean()

                # RBF kernel covariance function
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel()
                )

                # Set fixed lengthscale
                self.covar_module.base_kernel.lengthscale = torch.tensor(200.0).cuda()

            def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
                """ Forward process of the Gaussian process model. """
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        # Set device to gpu if gpu is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Extract x, y coordinates, z values and mask values
        H, W = channel.shape
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        x = x.ravel()
        y = y.ravel()

        # Sample a subset of coordinates and corresponding computed local average cell intensity magnitude values
        train_points, train_values = sample_local_average_magnitude(channel, mask, sample_fraction)
        train_points = torch.tensor(train_points, device=device)
        train_values = torch.tensor(train_values, device=device)

        # Define likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = GPModel(train_points, train_values, likelihood).to(device)

        # Train model
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)            # Optimizer
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)   # Loss

        # Train for 100 iterations
        for i in range(100):
            # Compute loss
            optimizer.zero_grad()
            output = model(train_points)
            mll_loss = -mll(output, train_values)
            loss = mll_loss

            # Backprop and update weights
            loss.backward()
            optimizer.step()

            # Reduce learning rate after 50 iterations
            if (i==50) and (loss.item() > 0):
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Write final loss if verbose
        if verbose: 
            print(f"MLL Loss: {mll_loss.item():.4f}")

        # Get surface estimation at all points by model inference
        model.eval()
        likelihood.eval()
        points = torch.tensor(np.vstack((x, y)).T, dtype=torch.float32, device=device)
        pred_values = torch.zeros(len(points), dtype=torch.float32, device=device)
        batch_size = 100000     # Batch size for model inference
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(0, points.size(0), batch_size):
                batch = points[i:i + batch_size]
                output = model(batch).mean
                pred_values[i:i + batch_size] = output
        pred_values = pred_values.cpu().numpy().reshape(H, W)

        # Cleanup memory
        del train_points, train_values, points, model, likelihood
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

        if verbose:
            print(f"Min: {pred_values.min():.3f}, Max: {pred_values.max():.3f}, Min abs: {abs(pred_values).min():.3f}")

        # Return normalized channel (channel / pred_values) and estimated surface (pred_values)
        return channel / pred_values, pred_values
    
    def fit_bspline_channel(
        channel: np.ndarray, 
        mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """ Fits a smooth surface using a cubic bivariate spline representation. 
        
        Args: 
            channel: A one channel image (H, W) from which to sample local average intensity magnitudes.
            mask: Binary mask of shape (H, W), where 1 indicates cell pixels.
        
        Returns:
            tuple - The normalized channel and estimated surface.
        """
        
        # Sample a subset of coordinates and corresponding computed local average cell intensity magnitude values
        H, W = channel.shape
        train_points, train_values = sample_local_average_magnitude(channel, mask, sample_fraction)
        x_train = train_points[:,0]
        y_train = train_points[:,1]
        
        # Fit cubic bivariate spline to sampled data
        kx = 3
        ky = 3
        tck = bisplrep(
            x=x_train,
            y=y_train,
            z=train_values,
            kx=kx,
            ky=ky,
            s=0.025,
            task=0
        )

        # Evaluate the spline over the full grid
        pred_z = bisplev(np.arange(W), np.arange(H), tck).T

        if verbose:
            print(f"Min: {pred_z.min():.3f}, Max: {pred_z.max():.3f}, Min abs: {abs(pred_z).min():.3f}")

        # Return normalized channel (channel / pred_z) and estimated surface (pred_z)
        return channel / pred_z, pred_z
    
    def linear_interpolate_channel(
        channel: np.ndarray,
        mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """ Forms a surface by interpolating all points in the domain from the sampled subset. 
        
        Args: 
            channel: A one channel image (H, W) from which to sample local average intensity magnitudes.
            mask: Binary mask of shape (H, W), where 1 indicates cell pixels.
        
        Returns:
            tuple - The normalized channel and estimated surface.
        """
        
        # Sample a subset of coordinates and corresponding computed local average cell intensity magnitude values
        H, W = channel.shape
        train_points, train_values = sample_local_average_magnitude(channel, mask, sample_fraction)
        x_train = train_points[:,0]
        y_train = train_points[:,1]
        
        # Linearly interpolate entire domain from sampled points
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        pred_z = griddata((x_train, y_train), train_values, (x, y), method='linear')

        if verbose:
            print(f"Min: {pred_z.min():.3f}, Max: {pred_z.max():.3f}, Min abs: {abs(pred_z).min():.3f}")

        # Return normalized channel (channel / pred_z) and estimated surface (pred_z)
        return channel / pred_z, pred_z
        
    # Allocate normalized sample and fitted background
    normalized_sample = np.zeros_like(sample)
    fitted_surface = np.zeros_like(sample)

    # Linear interpolation method
    if method.lower() == "linear":
        # Fit polynomial planes to each channel in parallel
        results = Parallel(n_jobs=-1)(
            delayed(lambda i, c: (i, c, *linear_interpolate_channel(sample[i, c], mask)))(i, c) 
            for i in range(sample.shape[0]) 
            for c in range(sample.shape[1])
        )
        # Assign results to the original arrays
        for i, c, norm_channel, fitted_surf in results:
            normalized_sample[i, c] = norm_channel
            fitted_surface[i, c] = fitted_surf
        
    # Cubic bivariate spline fitting method
    elif method.lower() == "b-spline":
        # Fit polynomial planes to each channel in parallel
        results = Parallel(n_jobs=-1)(
            delayed(lambda i, c: (i, c, *fit_bspline_channel(sample[i, c], mask)))(i, c) 
            for i in range(sample.shape[0]) 
            for c in range(sample.shape[1])
        )
        # Assign results to the original arrays
        for i, c, norm_channel, fitted_surf in results:
            normalized_sample[i, c] = norm_channel
            fitted_surface[i, c] = fitted_surf
    
    # Gaussian process regression method
    elif method.lower() == "gpr":
        for i in range(sample.shape[0]):
            for c in range(sample.shape[1]):
                if verbose: 
                    print(f"Processing Modality/Channel: {i}/{c}")
                normalized_sample[i,c], fitted_surface[i,c] = fit_gpr_channel(sample[i,c], mask)

    return normalized_sample, fitted_surface
