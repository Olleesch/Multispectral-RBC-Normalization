import os
import glob
import numpy as np
import tifffile as tiff
from pathlib import Path
from torch.utils.data import Dataset
from scipy.ndimage import median_filter


def min_max_scale(image):
    """ Min max scale image in range [0,1] """
    min = np.min(image)
    max = np.max(image)
    image_norm = (image - min) / (max - min)
    return image_norm


class Dataset(Dataset):
    def __init__(self, dataset_paths: list[Path] | list[str]):
        # Get image paths from data folders
        self.image_paths_R = [path for dataset_path in dataset_paths for path in glob.glob(os.path.join(dataset_path, '*_R.tiff'))]
        self.image_paths_S = [path for dataset_path in dataset_paths for path in glob.glob(os.path.join(dataset_path, '*_S.tiff'))]
        self.image_paths_T = [path for dataset_path in dataset_paths for path in glob.glob(os.path.join(dataset_path, '*_T.tiff'))]

        # image_paths_R = [glob.glob(os.path.join(dataset_path, '*_R.tiff')) for dataset_path in dataset_paths]
        # image_paths_S = [glob.glob(os.path.join(dataset_path, '*_S.tiff')) for dataset_path in dataset_paths]
        # image_paths_T = [glob.glob(os.path.join(dataset_path, '*_T.tiff')) for dataset_path in dataset_paths]

        # image_paths_R_2 = glob.glob(os.path.join(dataset_path_2, '*_R.tiff'))
        # image_paths_S_2 = glob.glob(os.path.join(dataset_path_2, '*_S.tiff'))
        # image_paths_T_2 = glob.glob(os.path.join(dataset_path_2, '*_T.tiff'))

        # self.image_paths_R = image_paths_R_1 + image_paths_R_2
        # self.image_paths_S = image_paths_S_1 + image_paths_S_2
        # self.image_paths_T = image_paths_T_1 + image_paths_T_2

        assert len(self.image_paths_R) == len(self.image_paths_S) and len(self.image_paths_S) == len(self.image_paths_T), \
            "Error: Image paths inconsistent, check dataset folders. "

        # Set median filter size (to remove salt pepper noise)
        self.median_filter_size = 3

    def __len__(self):
        return len(self.image_paths_R)

    def __getitem__(self, idx):
        # Read image
        image_R = tiff.imread(self.image_paths_R[idx]).astype(np.float32)
        image_S = tiff.imread(self.image_paths_S[idx]).astype(np.float32)
        image_T = tiff.imread(self.image_paths_T[idx]).astype(np.float32)

        # Filter image to remove noise
        for c in range(image_R.shape[0]):
            image_R[c] = median_filter(image_R[c], size=self.median_filter_size)
            image_S[c] = median_filter(image_S[c], size=self.median_filter_size)
            image_T[c] = median_filter(image_T[c], size=self.median_filter_size)
        
        # Rescale image channels
        image_R = min_max_scale(image_R)
        image_S = min_max_scale(image_S)
        image_T = min_max_scale(image_T)

        # Return dict with reflected, scattered, and transmitted image
        image_dict = {
            'R': image_R,
            'S': image_S,
            'T': image_T
        }
        return image_dict