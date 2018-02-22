# -*- coding: utf-8 -*-

"""
Utilities for ndcube.
"""

import copy

import numpy as np
import astropy.units as u

from ndcube.utils import wcs

__all__ = ['wcs_axis_to_data_axis', 'data_axis_to_wcs_axis', 'select_order',
           'convert_extra_coords_dict_to_input_format', 'get_axis_number_from_axis_name']


def data_axis_to_wcs_axis(data_axis, missing_axis):
    """Converts a data axis number to the corresponding wcs axis number."""
    if data_axis is None:
        result = None
    else:
        result = len(missing_axis)-np.where(np.cumsum(
            [b is False for b in missing_axis][::-1]) == data_axis+1)[0][0]-1
    return result


def wcs_axis_to_data_axis(wcs_axis, missing_axis):
    """Converts a wcs axis number to the corresponding data axis number."""
    if wcs_axis is None:
        result = None
    else:
        if missing_axis[wcs_axis]:
            result = None
        else:
            data_ordered_wcs_axis = len(missing_axis)-wcs_axis-1
            result = data_ordered_wcs_axis-sum(missing_axis[::-1][:data_ordered_wcs_axis])
    return result


def select_order(axtypes):
    """
    Returns indices of the correct data order axis priority given a list of WCS CTYPEs.

    For example, given ['HPLN-TAN', 'TIME', 'WAVE'] it will return
    [1, 2, 0] because index 1 (time) has the lowest priority, followed by
    wavelength and finally solar-x.

    Parameters
    ----------
    axtypes: str list
        The list of CTYPEs to be modified.

    """
    order = [(0, t) if t in ['TIME', 'UTC'] else
             (1, t) if t == 'WAVE' else
             (2, t) if t == 'HPLT-TAN' else
             (axtypes.index(t) + 3, t) for t in axtypes]
    order.sort()
    result = [axtypes.index(s) for (_, s) in order]
    return result


def _format_input_extra_coords_to_extra_coords_wcs_axis(extra_coords, missing_axis,
                                                        data_shape):
    extra_coords_wcs_axis = {}
    coord_format_error = ("Coord must have three properties supplied, "
                          "name (str), axis (int), values (Quantity or array-like)."
                          " Input coord: {0}")
    coord_0_format_error = ("1st element of extra coordinate tuple must be a "
                            "string giving the coordinate's name.")
    coord_1_format_error = ("2nd element of extra coordinate tuple must be None "
                            "or an int giving the data axis "
                            "to which the coordinate corresponds.")
    coord_len_error = ("extra coord ({0}) must have same length as data axis "
                       "to which it is assigned: coord length, {1} != data axis length, {2}")
    for coord in extra_coords:
        # Check extra coord has the right number and types of info.
        if len(coord) != 3:
            raise ValueError(coord_format_error.format(coord))
        if not isinstance(coord[0], str):
            raise ValueError(coord_0_format_error.format(coord))
        if coord[1] is not None and not isinstance(coord[1], int) and \
                not isinstance(coord[1], np.int64):
            raise ValueError(coord_1_format_error)
        # Unless extra coord corresponds to a missing axis, check length
        # of coord is same is data axis to which is corresponds.
        if coord[1] is not None:
            if not missing_axis[::-1][coord[1]]:

                if len(coord[2]) != data_shape[coord[1]]:
                    raise ValueError(coord_len_error.format(coord[0], len(coord[2]),
                                                            data_shape[coord[1]]))
        # Determine wcs axis corresponding to data axis of coord
        extra_coords_wcs_axis[coord[0]] = {
            "wcs axis": data_axis_to_wcs_axis(coord[1], missing_axis),
            "value": coord[2]}
    return extra_coords_wcs_axis


def convert_extra_coords_dict_to_input_format(extra_coords, missing_axis):
        """
        Converts NDCube.extra_coords attribute to format required as input for new NDCube.

        Parameters
        ----------
        extra_coords: dict
            An NDCube.extra_coords instance.

        Returns
        -------
        input_format: `list`
            Infomation on extra coords in format required by `NDCube.__init__`.

        """
        coord_names = list(extra_coords.keys())
        result = []
        for name in coord_names:
            coord_keys = list(extra_coords[name].keys())
            if "wcs axis" in coord_keys and "axis" not in coord_keys:
                axis = wcs_axis_to_data_axis(extra_coords[name]["wcs axis"], missing_axis)
            elif "axis" in coord_keys and "wcs axis" not in coord_keys:
                axis = extra_coords[name]["axis"]
            else:
                raise KeyError("extra coords dict can have keys 'wcs axis' or 'axis'.  Not both.")
            result.append((name, axis, extra_coords[name]["value"]))
        return result


def get_axis_number_from_axis_name(axis_name, world_axis_physical_types):
    """
    Returns axis number (numpy ordering) given a substring unique to a world axis type string.

    Parameters
    ----------
    axis_name: `str`
        Name or substring of name of axis as defined by NDCube.world_axis_physical_types

    world_axis_physical_types: iterable of `str`
        Output from NDCube.world_axis_physical_types for relevant cube,
        i.e. iterable of string axis names.

    Returns
    -------
    axis_index[0]: `int`
        Axis number (numpy ordering) corresponding to axis name
    """
    axis_index = [axis_name in world_axis_type for world_axis_type in world_axis_physical_types]
    axis_index = np.arange(len(world_axis_physical_types))[axis_index]
    if len(axis_index) != 1:
        raise ValueError("User defined axis with a string that is not unique to "
                         "a physical axis type. {0} not in any of {1}".format(
                             str_axis, world_axis_types))
    return axis_index[0]


def collapse_ndcube_over_axis(cube, data_axis, how):
    """Sums a cube over one of its axes.

    Parameters
    ----------
    cube: `ndcube.ndcube.NDCube`
        Cube to being summed.

    data_axis: `int`
        The axis number (data order, not wcs order) over which to be summed.

    how: `str`
        Identifies how the axis should be collapsed.  Options are outlined in doctstring
        of _apply_reduction_over_axis().

    Returns
    -------
    new_cube: `ndcube.ndcube.NDCube`
        Summed NDcube with dimensions N-1 where N is number of dimensions of original cube.

    """
    # Get new data, uncertainty and mask as masked arrays.
    new_data, new_uncertainty, new_mask = _apply_reduction_over_axis(
        cube.data, cube.uncertainty.array, cube.mask, data_axis, how)
    # Slice WCS and extra coords at the midpoint along summing axis.
    new_wcs, new_missing_axis, new_extra_coords_wcs_axis = _get_0th_ndcube_coords(
        cube.wcs, cube.missing_axis, cube._extra_coords_wcs_axis, data_axis,
        cube.dimensions.value[data_axis])
    new_extra_coords = convert_extra_coords_dict_to_input_format(
        new_extra_coords_wcs_axis, new_missing_axis)
    # Return new NDCube
    return new_data, new_wcs, new_uncertainty, new_mask, new_extra_coords, new_missing_axis


def _apply_reduction_over_axis(data, uncertainty, mask, axis, how):
    """
    Collapses arrays described by the same mask over an axis using a user-selected function.

    The how arg describes the function to be applied.  The options are:
    'sum', 'mean'.

    Parameters
    ----------
    data: `numpy.ndarray`
        Data arrays over which function should be applied.

    uncertainty: `numpy.ndarray`
        Uncertainty for each elements in data.  Must be same shape as data.

    mask: `numpy.ndarray` of `bool` type
        Mask for the arrays. True implies a masked value.
        Must be of same shape of arrays in arrays arg.

    axis: `int`
        Axis over which function should be applied.

    how: `str`
        Describes which function should be used.  See top of docstring for options.

    Returns
    -------
    new_data: `numpy.ndarray`
        New data array collapsed over the given axis via the selected method.

    new_uncertainty: `numpy.ndarray`
        Uncertainties corresponding to new data.

    new_mask: `numpy.ndarray` of `bool`
        The new collapsed mask.

    """
    if how is "sum":
        new_data = np.ma.masked_array(data, mask).sum(axis)
        new_mask = new_data.mask
        new_data = new_data.data
        new_uncertainty = np.sqrt(
            np.ma.masked_array(uncertainty**2, mask).sum(axis)).data
    elif how is "mean":
        new_data = np.ma.masked_array(data, mask).mean(axis)
        new_mask = new_data.mask
        new_data = new_data.data
        lengths_along_axis = np.ma.masked_array(np.ones(data.shape), mask).sum(axis).data
        new_uncertainty = np.sqrt(np.ma.masked_array(
            uncertainty**2, mask).sum(axis)).data/lengths_along_axis
    else:
        raise ValueError("Unrecognized value for 'how' arg.")
    return new_data, new_uncertainty, new_mask


def _get_0th_ndcube_coords(wcs_object, missing_axis, extra_coords_wcs_axis, data_axis, len_axis):
    """
    Returns a wcs and extra_coords dict sliced at the midpoint along an axis.

    The midpoint is defined as the rounded integer index closest to the midpoint
    along the selected axis.

    Parameters
    ----------
    wcs_object: `ndcube.utils.wcs.WCS`
        The WCS object to be sliced.

    missing_axis: `list` of `bool`
        Denotes which WCS axes which are "missing" from data.  True implies missing.
        Order is same as WCS axes.

    extra_coords_wcs_axis: `dict` of `dict`
        Extra coords dictionary as defined by `ndcube.NDCube._extra_coords_wcs_axis`.

    data_axis: `int`
        Number of axis aong which midpoint is to be found.  Numbering follows the
        data orientation, i.e. reversed relative to the WCS axes' order.

    len_axis: `int`
        Length of data axis over which we are finding the midpoint.

    Returns
    -------
    new_wcs_object: `ndcube.utils.wcs.WCS`
        The WCS object to be sliced at the midpoint along the chosen axis.

    new_missing_axis: `list` of `bool`
        Denotes which WCS axes which are "missing" from data after slicing the WCS object.
        Order follows WCS axes' order.

    new_extra_coords_wcs_axis: `dict` of `dict`
        Extra coords dictionary where coords along chosen axis are sliced at the midpoint.

    """
    # Get number of dimensions.
    n_dim = sum(np.invert(np.array(missing_axis)))
    # Get int closest to midpoint index along summed axis.
    index = 0
    # Produce new WCS by slicing at median of summed axis.
    item = [slice(None)]*n_dim
    item[data_axis] = index
    item = tuple(item)
    new_wcs_object, new_missing_axis = wcs._wcs_slicer(
        wcs_object, missing_axis, item)
    # Reduce extra coords along summed axis to 0th value.
    wcs_axis = data_axis_to_wcs_axis(data_axis, missing_axis)
    # Make CDELT the width of the pre-collapsed axis.
    new_wcs_object.wcs.cdelt[wcs_axis] = new_wcs_object.wcs.cdelt[wcs_axis] * len_axis
    new_extra_coords_wcs_axis = copy.deepcopy(extra_coords_wcs_axis)
    for key in new_extra_coords_wcs_axis:
        if new_extra_coords_wcs_axis[key]["wcs axis"] == wcs_axis:
            new_extra_coords_wcs_axis[key]["value"] = \
              new_extra_coords_wcs_axis[key]["value"][index]
    # Return
    return new_wcs_object, new_missing_axis, new_extra_coords_wcs_axis
