import numpy as np
from tensorflow.python.framework import dtypes


def orthogonal_initializer(scale=1.0, seed=None, dtype=dtypes.float32):
    def _initializer(shape, dtype=dtype, partition_info=None):
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]
        flat_shape = (num_rows, num_cols)
        if seed is not None:
            np.random.seed(seed)
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # Pick the one with the correct shape
        q = q.reshape(shape)
        return scale * q[:shape[0], :shape[1]]
    return _initializer
