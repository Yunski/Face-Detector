import numpy as np
from scipy.signal import convolve2d

def conv_diff(a):
    return np.convolve(a, [1, 0, -1], mode='same')

def conv_aggr(a):
    return np.convolve(a, [1, 2, 3, 3, 2, 1], mode='valid')

def conv2(a, f):
    a_row = np.apply_along_axis(f, axis=1, arr=a)
    return np.apply_along_axis(f, axis=0, arr=a_row)

def hog36(img, orientations, wrap180):
    # Compute gradient using tiny centered-difference filter
    gx = np.apply_along_axis(conv_diff, axis=1, arr=img)
    gx[:,0] = 0
    gx[:,-1] = 0
    gy = np.apply_along_axis(conv_diff, axis=0, arr=img)
    gy[0,:] = 0
    gy[-1,:] = 0
    gmag = np.hypot(gx, gy)

    multiplier = 0
    if wrap180:
        multiplier = orientations / np.pi
    else:
        multiplier = orientations / (2 * np.pi)

    gdir = np.mod(np.arctan2(gy, gx) * multiplier, orientations)

    # Bin by orientation
    cells = np.zeros((6, 6, orientations))
    for i in range(orientations):
        # Select out by orientation, weight by gradient magnitude
        if i == orientations - 1:
            gdir_wrap = np.mod(gdir + 1, orientations) - 1
            this_orientation = gmag * np.maximum(1-np.abs(gdir_wrap), 0)
        else:
            this_orientation = gmag * np.maximum(1-np.abs(gdir-(i+1)), 0)

        # Aggregate and downsample
        this_orientation = conv2(this_orientation, f=conv_aggr)
        N = this_orientation.size
        cells[:,:,i] = this_orientation[0:N:6, 0:N:6]
    # Create the output vector
    descriptor = np.zeros(5 * 5 * 4 * orientations)
    for block_i in range(5):
        for block_j in range(5):
            block = cells[block_i:block_i+2, block_j:block_j+2, :]
            block_unrolled = np.ndarray.flatten(block, order='F')
            norm_block_unrolled = np.linalg.norm(block_unrolled)
            if norm_block_unrolled > 0:
                block_unrolled = block_unrolled / norm_block_unrolled
            else:
                block_unrolled = 1 / block_unrolled.shape[0]
            first = (5*block_i+block_j) * 4*orientations
            last = first + 4*orientations
            descriptor[first:last] = block_unrolled
    return descriptor
