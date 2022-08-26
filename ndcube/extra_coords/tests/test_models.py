import numpy as np

from ..models import BinEdgesTabular


def test_bin_edges_tabular():
    x = np.array([-0.2, 3, 0, 1, 1.2])
    world_low=np.array([10, 11, 15, 16.5])
    world_high=np.array([11, 15, 16, 17])
    expected = np.array([10.3, 16.75, 10.5, 13., 13.8])
    m = BinEdgesTabular(world_low, world_high)
    output = m.evaluate(x, world_low, world_high)
    np.testing.assert_allclose(output, expected)
