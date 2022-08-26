import numpy as np
import scipy.interpolate
from astropy.modeling import Model, Parameter

class BinEdgesTabular(Model):
    n_inputs = 1
    n_outputs = 1

    world_low = Parameter()
    world_high = Parameter()

    @staticmethod
    def evaluate(x, world_low, world_high):
        n_bins = len(world_low)
        if len(world_high) != n_bins:
            raise ValueError("low and high arrays of bin must have same number of elements.")
        pix_edges = np.arange(-0.5, n_bins+1)
        idx_pix = np.digitize(x, pix_edges, right=False) - 1
        unique_bin_idx = np.unique(idx_pix)
        output = np.empty(len(x), dtype=float)
        for i in unique_bin_idx:
            idx_x = np.where(idx_pix == i)
            model = scipy.interpolate.interp1d(pix_edges[i:i+2], np.array([world_low[i], world_high[i]]))
            output[idx_x] = model(x[idx_x])
        return output
