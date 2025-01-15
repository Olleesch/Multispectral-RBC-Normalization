import os
import glob
import numpy as np
import tifffile as tiff

from pathlib import Path
from torch.utils.data import Dataset
from scipy.ndimage import median_filter
from joblib import Parallel, delayed



def check_image_format(images: np.ndarray) -> np.ndarray:
    """ Check valid format of input images. """
    if len(images.shape) == 3:
        images = np.expand_dims(images, 0)
    assert len(images.shape) == 4, \
        "Error: Invalid image shape, expected format (N,C,H,W) or (C,H,W), " + \
        "N - number of images, C - number of channels, H - image height, W - image width."
    return images


def check_sample_format(samples: np.ndarray, allow_multiple: bool = True) -> np.ndarray:
    """ Check valid format of input samples. """
    if allow_multiple:
        if len(samples.shape) == 4:
            samples = np.expand_dims(samples, 0)
        assert len(samples.shape) == 5, \
            "Error: Invalid sample shape, expected format (N,M,C,H,W) or (M,C,H,W), " + \
            "N - number of samples, M - number of modalities, C - number of channels, H - image height, W - image width."
    else:
        assert len(samples.shape) == 4, \
            "Error: Invalid sample shape, expected format (M,C,H,W), " + \
            "M - number of modalities, C - number of channels, H - image height, W - image width."
    return samples


""" Pre-processing """

def min_max_scale(image: np.ndarray) -> np.ndarray:
    """ Min max scale image in range [0,1] """
    min = np.min(image)
    max = np.max(image)
    image_norm = (image - min) / (max - min)
    return image_norm


def preprocess(sample: np.ndarray) -> np.ndarray:
    """ Pre-process sample.

    Args:
        sample: A numpy array with the sample data, format (M,C,H,W),
                M - number of modalities, C - number of channels, 
                H - image height, W - image width.
    
    Returns:
        The processed sample.
    """
    # Check correct format of input sample
    sample = check_sample_format(sample, allow_multiple=False)

    # Parameters
    n_edge_pixels = 5
    filter_kernel_size = 3

    # Remove a few edge pixels
    processed_sample = sample[:, :, n_edge_pixels:-n_edge_pixels, n_edge_pixels:-n_edge_pixels]

    def process_channel(channel):
        """ Process channel by applying median filter and min-max scaling. """
        channel = median_filter(channel, size=filter_kernel_size)
        channel = median_filter(channel, size=filter_kernel_size)
        channel = min_max_scale(channel)
        return channel
    
    # Process each channel in parallel
    results = Parallel(n_jobs=-1)(
        delayed(lambda i, c: (i, c, process_channel(processed_sample[i, c])))(i, c) 
        for i in range(sample.shape[0]) 
        for c in range(sample.shape[1])
    )

    # Assign results to the original arrays
    for i, c, channel in results:
        processed_sample[i, c] = channel
    
    return processed_sample


""" Dataset """

class Dataset(Dataset):
    """ Dataset class.
    
    Data consists of samples gathered from images specified by multiple 'dataset root paths'. Each dataset root path points to a dataset directory, 
    where there must be a subdirectory 'img_raw' that contains tiff images of three modalities indicated by subscripts '_R.tiff', '_S.tiff', '_T.tiff'. 
    Note that if the data is not in this format, a script is required to transform the data to this specific format. Such a script for the 'sample_modale.mat'
    data is provided in 'format_mat_dataset.py'. 

    Attributes: 
        image_paths_R: A list of paths to the reflectance mode raw images.
        image_paths_S: A list of paths to the scattering mode raw images.
        image_paths_T: A list of paths to the transmittance mode raw images.
    """
    def __init__(self, dataset_paths: list[Path] | list[str]):
        """ Initializes a dataset instance.
        
        Args:
            dataset_paths: A list of dataset root paths from which to get image data.
        """
        # Get image paths from dataset folders
        self.image_paths_R = [Path(path) for dataset_path in dataset_paths for path in glob.glob(os.path.join(dataset_path / 'img_raw', '*_R.tiff'))]
        self.image_paths_S = [Path(path) for dataset_path in dataset_paths for path in glob.glob(os.path.join(dataset_path / 'img_raw', '*_S.tiff'))]
        self.image_paths_T = [Path(path) for dataset_path in dataset_paths for path in glob.glob(os.path.join(dataset_path / 'img_raw', '*_T.tiff'))]

        # Check that there are an equal number of images per modality
        assert len(self.image_paths_R) == len(self.image_paths_S) and len(self.image_paths_S) == len(self.image_paths_T), \
            "Error: Image paths inconsistent, check dataset folders. "

    def __len__(self) -> int:
        """ Get length of dataset. """
        return len(self.image_paths_R)

    def __getitem__(self, idx: int) -> dict:
        """ Get sample from dataset index.

        Args:
            idx: Global dataset index.
        
        Returns:
            A dict containing the sample and the paths of the raw images on the form:

            {'sample': np.ndarray,
             'paths': list[Path]}
        """
        # Get raw image paths
        path_R = self.image_paths_R[idx]
        path_S = self.image_paths_S[idx]
        path_T = self.image_paths_T[idx]
        paths = [path_R, path_S, path_T]

        # Read images
        image_R = tiff.imread(path_R).astype(np.float32)
        image_S = tiff.imread(path_S).astype(np.float32)
        image_T = tiff.imread(path_T).astype(np.float32)

        # Construct sample
        sample = np.array([image_R, image_S, image_T])

        # Pre-process sample
        sample = preprocess(sample)
        
        # Construct result dict
        sample_dict = {
            "sample": sample,
            "paths": paths
        }
        return sample_dict


class NormalizedDataset(Dataset):
    """ Dataset class for normalized samples.
    
    Data consists of samples gathered from images specified by multiple 'dataset root paths'. Each dataset root path points to a dataset directory, 
    where there must be two subdirectories: 
        'img_raw' that contains the raw tiff images of three modalities indicated by subscripts '_R.tiff', '_S.tiff', '_T.tiff', 
        'img_norm' that contains the normalized tiff images of three modalities indicated by subscripts '_R_norm.tiff', '_S_norm.tiff', '_T_norm.tiff'
            obtained by the main pipeline.
    Note that this dataset requires that we have already ran the main pipeline on all dataset root paths. 

    Attributes: 
        image_paths_R: A list of paths to the reflectance mode raw images.
        image_paths_S: A list of paths to the scattering mode raw images.
        image_paths_T: A list of paths to the transmittance mode raw images.
        image_paths_R_norm: A list of paths to the reflectance mode normalized images.
        image_paths_S_norm: A list of paths to the scattering mode normalized images.
        image_paths_T_norm: A list of paths to the transmittance mode normalized images.
    """
    def __init__(self, dataset_paths: list[Path] | list[str]):
        """ Initializes a dataset instance.
        
        Args:
            dataset_paths: A list of dataset root paths from which to get image data.
        """
        # Get image paths from data folders
        self.image_paths_R = [Path(path) for dataset_path in dataset_paths for path in glob.glob(os.path.join(dataset_path / 'img_raw', '*_R.tiff'))]
        self.image_paths_S = [Path(path) for dataset_path in dataset_paths for path in glob.glob(os.path.join(dataset_path / 'img_raw', '*_S.tiff'))]
        self.image_paths_T = [Path(path) for dataset_path in dataset_paths for path in glob.glob(os.path.join(dataset_path / 'img_raw', '*_T.tiff'))]
        self.image_paths_R_norm = [Path(path) for dataset_path in dataset_paths for path in glob.glob(os.path.join(dataset_path / 'img_normalized', '*_R_norm.tiff'))]
        self.image_paths_S_norm = [Path(path) for dataset_path in dataset_paths for path in glob.glob(os.path.join(dataset_path / 'img_normalized', '*_S_norm.tiff'))]
        self.image_paths_T_norm = [Path(path) for dataset_path in dataset_paths for path in glob.glob(os.path.join(dataset_path / 'img_normalized', '*_T_norm.tiff'))]

        # Check that there are an equal number of images per modality and an equal number of raw and normalized images
        assert len(self.image_paths_R) == len(self.image_paths_S) and len(self.image_paths_S) == len(self.image_paths_T) and \
            len(self.image_paths_R_norm) == len(self.image_paths_S_norm) and len(self.image_paths_S_norm) == len(self.image_paths_T_norm) and \
            len(self.image_paths_R) == len(self.image_paths_R_norm), \
            "Error: Image paths inconsistent, check dataset folders. "

    def __len__(self) -> int:
        """ Get length of dataset. """
        return len(self.image_paths_R)

    def __getitem__(self, idx: int) -> dict:
        """ Get sample from dataset index.

        Args:
            idx: Global dataset index.
        
        Returns:
            A dict containing the raw and normalized sample and the sample name (from originating dataset) on the form:

            {'sample': np.ndarray,
             'sample_norm': np.ndarray,
             'name': str}
        """
        # Get raw image paths
        path_R = self.image_paths_R[idx]
        path_S = self.image_paths_S[idx]
        path_T = self.image_paths_T[idx]

        # Read raw images
        image_R = tiff.imread(path_R).astype(np.float32)
        image_S = tiff.imread(path_S).astype(np.float32)
        image_T = tiff.imread(path_T).astype(np.float32)

        # Construct raw sample and pre-process it
        sample = preprocess(np.array([image_R, image_S, image_T]))

        # Get normalized image paths
        path_R_norm = self.image_paths_R_norm[idx]
        path_S_norm = self.image_paths_S_norm[idx]
        path_T_norm = self.image_paths_T_norm[idx]

        # Read normalized images
        image_R_norm = tiff.imread(path_R_norm).astype(np.float32)
        image_S_norm = tiff.imread(path_S_norm).astype(np.float32)
        image_T_norm = tiff.imread(path_T_norm).astype(np.float32)
        
        # Construct normalized sample (note that we do not pre-process normalized sample)
        sample_norm = np.array([image_R_norm, image_S_norm, image_T_norm])

        # Construct result dict
        sample_dict = {
            "sample": sample,
            "sample_norm": sample_norm,
            "name": str(path_R.stem)[:-2]
        }
        return sample_dict