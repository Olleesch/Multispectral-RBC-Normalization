import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale


class ImagePCA():
    """ 
    Class to perform PCA on image channels 
    
    Basically wraps sklearn.decomposition.PCA with code to ensure
    correct formatting and reshaping of image channels. 
    """
    def __init__(self, n_components=3):
        # Set number of output channels (generally 3 for RGB visualization)
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
    
    def fit(self, images):
        """ 
        Fit image PCA model to image data 
        
        Expects input images in format (N,C,H,W) or (C,H,W). 
        """
        # Check correct format of input images
        if len(images.shape) == 3:
            images = np.expand_dims(images, 0)
        assert len(images.shape) == 4, "Error: Invalid image shape"

        # Reshape to match expected format of PCA
        images = images.transpose(0, 2, 3, 1)
        _, _, C = images[0].shape
        data = images.reshape(-1, C)

        # Fit model
        self.pca.fit(data)

    def transform(self, images):
        """ 
        Transform image to principal components 
        
        Expects input images in format (N,C,H,W) or (C,H,W). 
        """
        # Check correct format of input images
        if len(images.shape) == 3:
            images = np.expand_dims(images, 0)
        assert len(images.shape) == 4, "Error: Invalid image shape"

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