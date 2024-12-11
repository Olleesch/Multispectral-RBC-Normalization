import numpy as np
# from itertools import combinations_with_replacement
# from sklearn.preprocessing import minmax_scale
from scipy.optimize import curve_fit


def poly1d(x, y, p):
    """ Evaluate a 2D polynomial """
    return p[0] + p[1]*x + p[2]*y

def poly2d(x, y, p):
    """ Evaluate a 2D polynomial """
    return p[0] + p[1]*x + p[2]*y + p[3]*x**2 + p[4]*y**2 + p[5]*x*y

def poly3d(x, y, p):
    """ Evaluate a 3D polynomial """
    return p[0] + p[1]*x + p[2]*y + p[3]*x**2 + p[4]*y**2 + p[5]*x*y + p[6]*x**3 + p[7]*y**3 + p[8]*x**2*y + p[9]*x*y**2

def fit_polynomial_background(image: np.ndarray, 
                              mask: np.ndarray, 
                              degree: int = 2):
    """
    Fit a polynomial background to an image considering only background pixels indicated by a binary mask. 
    
    Parameters:
        image (ndarray): Input image of shape (C, H, W) or (H, W). Pixel values should be floats. 
        mask (ndarray): Binary mask of shape (H, W), where `0` indicates background pixels.
        degree (int): Degree of the polynomial. Currently only support degree 1, 2, or 3. 
    
    Returns:
        tuple: Normalized image and fitted background.
    """

    # Check format of inputs
    if len(image.shape) == 2:
        image = np.expand_dims(image, 0)
    assert len(image.shape) == 3, \
        f"Error: Invalid image shape. Got shape {image.shape}, but require shape on form (C, H, W). "
    assert mask.shape == image.shape[1:], \
        f"Error: Mask shape does not match image shape. Got mask shape {mask.shape}, but require shape on form (C, H, W). "

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
        x = x[m == 0]
        y = y[m == 0]
        z = z[m == 0]

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
    
    # Allocate normalized image and fitted background
    normalized_image = np.zeros_like(image)
    fitted_background = np.zeros_like(image)

    # Normalize image by fitting polynomial background plane
    for c in range(image.shape[0]):
        normalized_image[c], fitted_background[c] = fit_polynomial_background_(image[c], mask)

    return normalized_image, fitted_background

