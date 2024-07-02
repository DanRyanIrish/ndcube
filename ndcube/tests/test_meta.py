import copy

import numpy as np
import pytest

from ndcube.meta import NDMeta
from .helpers import assert_metas_equal


@pytest.fixture
def basic_meta_values():
    return {"a": "hello",
            "b": list(range(10, 25, 10)),
            "c": np.array([[1, 2, 3, 4], [10, 20, 30, 40], [100, 200, 300, 400]]),
            "d": list(range(3, 13, 3)),
            "e": list(range(2, 8, 2)),
            "f": "world",
            "g": ["hello", "world", "!"]
            }


@pytest.fixture
def basic_comments():
    return {"a": "Comment A",
            "b": "Comment B",
            "c": "Comment C",
            }


@pytest.fixture
def basic_axes():
    return {"b": 0,
            "c": (1, 3),
            "d": (3,),
            "e": 1,
            "f": 0,
            "g": (0, 1, 3)
            }


@pytest.fixture
def basic_data_shape():
    return (2, 3, 0, 4)


@pytest.fixture
def basic_meta(basic_meta_values, basic_comments, basic_axes):
    return NDMeta(basic_meta_values, basic_comments, basic_axes)


@pytest.fixture
def no_shape_meta():
    return NDMeta({"a": "hello"})


def test_shape(basic_meta, basic_data_shape):
    meta = basic_meta
    shape = np.asarray(basic_data_shape)
    assert all(meta.data_shape == shape)


def test_slice_axis_with_no_meta(basic_meta):
    meta = basic_meta
    output = meta.slice[:, :, 0]
    expected = copy.deepcopy(meta)
    expected._data_shape = meta._data_shape[[0, 1, 3]]
    expected._axes["c"] = (1, 2)
    expected._axes["d"] = (2,)
    expected._axes["g"] = (0, 1, 2)
    assert_metas_equal(output, expected)


def test_slice_away_independent_axis(basic_meta):
    meta = basic_meta
    item = 0
    output = meta.slice[item]
    values = dict([(key, value) for key, value in meta.items()])
    values["b"] = values["b"][0]
    values["g"] = ["world", "!"]
    del values["f"]
    comments = meta.comments
    axes = dict([(key, axis) for key, axis in meta.axes.items()])
    del axes["b"]
    del axes["f"]
    axes["c"] -= 1
    axes["d"] -= 1
    axes["e"] -= 1
    axes["g"] = (0, 2)
    shape = meta.data_shape[1:]
    expected = NDMeta(values, comments, axes)
    assert_metas_equal(output, expected)


def test_slice_away_independent_and_dependent_axis(basic_meta):
    meta = basic_meta
    item = (0, 1)
    output = meta.slice[item]
    values = dict([(key, value) for key, value in meta.items()])
    del values["f"]
    values["b"] = values["b"][0]
    values["c"] = values["c"][1]
    values["e"] = values["e"][1]
    values["g"] = "!"
    comments = meta.comments
    axes = dict([(key, axis) for key, axis in meta.axes.items()])
    del axes["b"]
    del axes["e"]
    del axes["f"]
    axes["c"] = 1
    axes["d"] = 1
    axes["g"] = 1
    shape = meta.data_shape[2:]
    expected = NDMeta(values, comments, axes)
    assert_metas_equal(output, expected)


def test_slice_dependent_axes(basic_meta):
    meta = basic_meta
    output = meta.slice[:, 1:3, :, 1]
    values = dict([(key, value) for key, value in meta.items()])
    values["c"] = values["c"][1:3, 1]
    values["d"] = values["d"][1]
    values["e"] = values["e"][1:3]
    values["g"] = values["g"][:2]
    comments = meta.comments
    axes = dict([(key, axis) for key, axis in meta.axes.items()])
    del axes["d"]
    axes["c"] = 1
    axes["g"] = (0, 1)
    shape = np.array([2, 2, 0])
    expected = NDMeta(values, comments, axes, data_shape=shape)
    assert_metas_equal(output, expected)


def test_slice_by_str(basic_meta):
    meta = basic_meta
    assert meta["a"] == "hello"
    assert meta["b"] == list(range(10, 25, 10))


def test_add1(basic_meta):
    meta = basic_meta
    name = "z"
    value = 100
    comment = "Comment E"
    meta.add(name, value, comment, None)
    assert name in meta.keys()
    assert meta[name] == value
    assert meta.comments[name] == comment
    assert meta.axes.get(name, None) is None


def test_add2(basic_meta):
    meta = basic_meta
    name = "z"
    value = list(range(2))
    axis = 0
    meta.add(name, value, None, axis)
    assert name in meta.keys()
    assert meta[name] == value
    assert meta.comments.get(name, None) is None
    assert meta.axes[name] == np.array([axis])


def test_add_overwrite(basic_meta):
    meta = basic_meta
    name = "a"
    value = "goodbye"
    meta.add(name, value, None, None, overwrite=True)
    assert meta[name] == value


def test_add_overwrite_error(basic_meta):
    meta = basic_meta
    with pytest.raises(KeyError):
        meta.add("a", "world", None, None)


def test_remove(basic_meta):
    meta = basic_meta
    name = "b"
    meta.remove(name)
    assert name not in meta.keys()
    assert name not in meta.comments.keys()
    assert name not in meta.axes.keys()


def test_rebin(basic_meta):
    meta = basic_meta
    rebinned_axes = {0, 2}
    new_shape = (1, 3, 5, 2)
    output = meta.rebin(rebinned_axes, new_shape)
    # Build expected result.
    expected = copy.deepcopy(meta)
    del expected._axes["b"]
    del expected._axes["c"]
    del expected._axes["d"]
    expected._data_shape = np.array([1, 3, 5, 2], dtype=int)
    assert_metas_equal(output, expected)