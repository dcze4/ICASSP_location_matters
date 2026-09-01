import h5py
import numpy as np


class HDF5StorageManager:
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path

    def store_embedding(self, key, embedding):
        """
        Add an embedding vector to an HDF5 file with the given key.
        Creates the file if it doesn't exist, otherwise appends to existing file.

        Args:
            hdf5_path: Path to the HDF5 file
            key: Key/name for storing the embedding
            embedding: Numpy array or list representing the embedding vector
        """

        with h5py.File(self.hdf5_path, "a") as f:
            # If key exists, overwrite it (or you can raise an error)
            if key in f:
                raise FileExistsError("There is already vector with this key")
            # Store the embedding
            f.create_dataset(key, data=embedding)

    def retrieve_embedding(
        self, key, normalize=False, global_mean=None, global_std=None
    ):
        with h5py.File(self.hdf5_path, "r") as f:
            result = f.get(key, None)
            embedding = result[()] if result is not None else None
            if normalize and embedding is not None:
                if global_mean is not None and global_std is not None:
                    embedding = (embedding - global_mean) / global_std
                else:
                    embedding = embedding / np.linalg.norm(embedding)
        return embedding
