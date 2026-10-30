import numpy as np

def compare_arrays(torch_arr, dlfs_arr, atol=1e-6, rtol=1e-6):
    print(np.allclose(torch_arr, dlfs_arr, atol=atol, rtol=rtol))
    print(np.max(np.abs(torch_arr - dlfs_arr)))