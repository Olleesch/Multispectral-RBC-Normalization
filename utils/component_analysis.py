import numpy as np
from sklearn.decomposition import PCA
from .data import check_image_format, check_sample_format


class ImagePCA():
    """ Class to perform PCA on image channels.
    
    Basically wraps sklearn.decomposition.PCA with code to ensure
    correct formatting and reshaping of image channels. 

    Attributes:
        n_components: The number of principal components to compute.
        pca: The PCA model object.
    """

    def __init__(self, n_components: int = 3):
        """ Initializes an image PCA model instance.
        
        Args:
            n_components: The number of principal components to compute.
        """
        # Set number of output channels (generally 3 for RGB visualization)
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
    
    def fit(self, images: np.ndarray):
        """ Fit image PCA model to image data.

        Args: 
            images: Input images on which to fit the pca model to. 
                    Expects input images in format (N,C,H,W) or (C,H,W),
                    N - number of images, C - number of channels, 
                    H - image height, W - image width.
        """
        # Check correct format of input images
        images = check_image_format(images)

        # Reshape to match expected format of PCA
        images = images.transpose(0, 2, 3, 1)
        _, _, C = images[0].shape
        data = images.reshape(-1, C)

        # Fit model
        self.pca.fit(data)

    def transform(self, images: np.ndarray) -> np.ndarray:
        """ Transform images to principal components.
        
        Args: 
            images: Input images to transform to principal components. 
                    Expects input images in format (N,C,H,W) or (C,H,W),
                    N - number of images, C - number of channels, 
                    H - image height, W - image width.
        
        Returns:
            A numpy array with the principal components of the original images.
        """
        # Check correct format of input images
        images = check_image_format(images)

        # Reshape to match expected format of PCA
        images = images.transpose(0, 2, 3, 1)
        H, W, C = images[0].shape

        # Compute principal components of each image
        principal_images = []
        for image in images:
            reshaped = image.reshape(-1, C)
            principal_image = self.pca.transform(reshaped)
            principal_image = principal_image.reshape(H, W, self.n_components)
            principal_images.append(principal_image)
        
        # Reformat output to (N,C,H,W)
        return np.stack(principal_images).transpose(0, 3, 1, 2)


class SamplePCA():
    """ Class to perform PCA on samples.
    
    Wraps ImagePCA above for compact api of three modalities per sample.

    Attributes:
        n_components: The number of principal components to compute.
        pca_R: The PCA model object of the reflectance mode.
        pca_S: The PCA model object of the scattering mode.
        pca_T: The PCA model object of the transmittance mode.
    """

    def __init__(self, n_components: int = 3):
        """ Initializes an sample PCA model instance.
        
        Args:
            n_components: The number of principal components to compute.
        """
        # Set number of output channels (generally 3 for RGB visualization)
        self.n_components = n_components
        self.pca_R = ImagePCA(n_components=n_components)
        self.pca_S = ImagePCA(n_components=n_components)
        self.pca_T = ImagePCA(n_components=n_components)
    
    def fit(self, samples: np.ndarray):
        """ Fit image PCA model to sample data.

        Args: 
            samples: Input samples on which to fit the pca model to. 
                     Expects input samples in format (N,M,C,H,W) or (M,C,H,W),
                     N - number of samples, M - number of modalities, 
                     C - number of channels, H - image height, W - image width.
        """
        # Check correct format of input samples
        samples = check_sample_format(samples, allow_multiple=True)

        # Fit model
        self.pca_R.fit(samples[:,0])
        self.pca_S.fit(samples[:,1])
        self.pca_T.fit(samples[:,2])

    def transform(self, samples: np.ndarray) -> np.ndarray:
        """ Transform samples to principal components.
        
        Args: 
            samples: Input samples on which to fit the pca model to. 
                     Expects input samples in format (N,M,C,H,W) or (M,C,H,W),
                     N - number of samples, M - number of modalities, 
                     C - number of channels, H - image height, W - image width.
        
        Returns:
            A numpy array with the principal components of the original samples.
        """
        # Check correct format of input images
        samples = check_sample_format(samples, allow_multiple=True)
        
        # Transform sample to principal components
        samples_principal = np.zeros((samples.shape[0], samples.shape[1], self.n_components, samples.shape[3], samples.shape[4]))
        samples_principal[:,0] = self.pca_R.transform(samples[:,0])
        samples_principal[:,1] = self.pca_S.transform(samples[:,1])
        samples_principal[:,2] = self.pca_T.transform(samples[:,2])
        
        return samples_principal