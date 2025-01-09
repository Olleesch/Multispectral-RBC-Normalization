import os
import glob
import numpy as np
import tifffile as tiff

from pathlib import Path
from torch.utils.data import Dataset
from scipy.ndimage import median_filter
from joblib import Parallel, delayed


""" Pre-processing """

def min_max_scale(image):
    """ Min max scale image in range [0,1] """
    min = np.min(image)
    max = np.max(image)
    image_norm = (image - min) / (max - min)
    return image_norm


def preprocess(sample):
    n_edge_pixels = 5
    filter_kernel_size = 3

    # Remove a few edge pixels
    processed_sample = sample[:, :, n_edge_pixels:-n_edge_pixels, n_edge_pixels:-n_edge_pixels]

    def process_channel(channel):
        channel = median_filter(channel, size=filter_kernel_size)
        channel = median_filter(channel, size=filter_kernel_size)
        channel = min_max_scale(channel)
        return channel
        
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
    def __init__(self, dataset_paths: list[Path] | list[str]):
        # Get image paths from data folders
        self.image_paths_R = [Path(path) for dataset_path in dataset_paths for path in glob.glob(os.path.join(dataset_path / 'img_raw', '*_R.tiff'))]
        self.image_paths_S = [Path(path) for dataset_path in dataset_paths for path in glob.glob(os.path.join(dataset_path / 'img_raw', '*_S.tiff'))]
        self.image_paths_T = [Path(path) for dataset_path in dataset_paths for path in glob.glob(os.path.join(dataset_path / 'img_raw', '*_T.tiff'))]

        assert len(self.image_paths_R) == len(self.image_paths_S) and len(self.image_paths_S) == len(self.image_paths_T), \
            "Error: Image paths inconsistent, check dataset folders. "

        # Set median filter size (to remove salt pepper noise)
        self.median_filter_size = 3

    def __len__(self):
        return len(self.image_paths_R)

    def __getitem__(self, idx):
        # Read image
        path_R = self.image_paths_R[idx]
        path_S = self.image_paths_S[idx]
        path_T = self.image_paths_T[idx]
        paths = [path_R, path_S, path_T]

        image_R = tiff.imread(path_R).astype(np.float32)
        image_S = tiff.imread(path_S).astype(np.float32)
        image_T = tiff.imread(path_T).astype(np.float32)

        sample = np.array([image_R, image_S, image_T])
        sample = preprocess(sample)
        
        sample_dict = {
            "sample": sample,
            "paths": paths
        }
        return sample_dict


class NormalizedDataset(Dataset):
    def __init__(self, dataset_paths: list[Path] | list[str]):
        # Get image paths from data folders
        self.image_paths_R_norm = [Path(path) for dataset_path in dataset_paths for path in glob.glob(os.path.join(dataset_path / 'img_normalized', '*_R_norm.tiff'))]
        self.image_paths_S_norm = [Path(path) for dataset_path in dataset_paths for path in glob.glob(os.path.join(dataset_path / 'img_normalized', '*_S_norm.tiff'))]
        self.image_paths_T_norm = [Path(path) for dataset_path in dataset_paths for path in glob.glob(os.path.join(dataset_path / 'img_normalized', '*_T_norm.tiff'))]
        
        self.image_paths_R = [Path(path) for dataset_path in dataset_paths for path in glob.glob(os.path.join(dataset_path / 'img_raw', '*_R.tiff'))]
        self.image_paths_S = [Path(path) for dataset_path in dataset_paths for path in glob.glob(os.path.join(dataset_path / 'img_raw', '*_S.tiff'))]
        self.image_paths_T = [Path(path) for dataset_path in dataset_paths for path in glob.glob(os.path.join(dataset_path / 'img_raw', '*_T.tiff'))]

        assert len(self.image_paths_R) == len(self.image_paths_S) and len(self.image_paths_S) == len(self.image_paths_T), \
            "Error: Image paths inconsistent, check dataset folders. "

        self.median_filter_size = 3

    def __len__(self):
        return len(self.image_paths_R)

    def __getitem__(self, idx):
        # Read image
        path_R_norm = self.image_paths_R_norm[idx]
        path_S_norm = self.image_paths_S_norm[idx]
        path_T_norm = self.image_paths_T_norm[idx]

        image_R_norm = tiff.imread(path_R_norm).astype(np.float32)
        image_S_norm = tiff.imread(path_S_norm).astype(np.float32)
        image_T_norm = tiff.imread(path_T_norm).astype(np.float32)
        
        sample_norm = np.array([image_R_norm, image_S_norm, image_T_norm])
        
        path_R = self.image_paths_R[idx]
        path_S = self.image_paths_S[idx]
        path_T = self.image_paths_T[idx]

        image_R = tiff.imread(path_R).astype(np.float32)
        image_S = tiff.imread(path_S).astype(np.float32)
        image_T = tiff.imread(path_T).astype(np.float32)

        sample = preprocess(np.array([image_R, image_S, image_T]))

        sample_dict = {
            "sample": sample,
            "sample_norm": sample_norm,
            "name": str(path_R.stem)[:-2]
        }
        
        return sample_dict