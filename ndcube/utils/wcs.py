# Author: Ankit Baruah and Daniel Ryan <ryand5@tcd.ie>

"""
Miscellaneous WCS utilities.
"""

import re
import numbers
from copy import deepcopy
from collections import UserDict

import numpy as np
from astropy import wcs
from astropy.wcs._wcs import InconsistentAxisTypesError

from ndcube.utils import cube as utils_cube

__all__ = ['wcs_ivoa_mapping', 'append_sequence_axis_to_wcs',
           'pixel_axis_to_world_axes', 'world_axis_to_pixel_axes',
           'pixel_axis_to_physical_types', 'physical_type_to_pixel_axes',
           'physical_type_to_world_axis',
           'get_dependent_pixel_axes', 'get_dependent_array_axes',
           'get_dependent_world_axes', 'get_dependent_physical_types']


class TwoWayDict(UserDict):
    @property
    def inv(self):
        """
        The inverse dictionary.
        """
        return {v: k for k, v in self.items()}


# Define a two way dictionary to hold translations between WCS axis
# types and International Virtual Observatory Alliance vocabulary.
# See http://www.ivoa.net/documents/REC/UCD/UCDlist-20070402.html
wcs_to_ivoa = {
    "DSUN": "custom:pos.distance.sunToObserver",
    "HGLT": "custom:pos.heliographic.stonyhurst.lon",
    "CRLT": "custom:pos.heliographic.carrington.lat",
    "HGLN": "custom:pos.heliographic.stonyhurst.lat",
    "CRLN": "custom:pos.heliographic.carrington.lon",
    "HECR": "custom:pos.distance.fromSunCenter",
    "HECR": "custom:pos.distance.fromSunSurface",
    "SOLX": "custom.pos.heliocentric.x",
    "SOLY": "custom.pos.heliocentric.y",
    "SOLZ": "custom.pos.heliocentric.z",
    "HPLT": "custom:pos.helioprojective.lat",
    "HPLN": "custom:pos.helioprojective.lon",
    "HPRZ": "custom:pos.helioprojective.z",
    "DIST": "pos.distance",
    "TIME": "time",
    "WAVE": "em.wl",
    "RA--": "pos.eq.ra",
    "DEC-": "pos.eq.dec",
    "FREQ": "em.freq",
    "STOKES": "phys.polarization.stokes",
    "PIXEL": "instr.pixel",
    "XPIXEL": "custom:instr.pixel.x",
    "YPIXEL": "custom:instr.pixel.y",
    "ZPIXEL": "custom:instr.pixel.z"
}
wcs_ivoa_mapping = TwoWayDict()
for key in wcs_to_ivoa.keys():
    wcs_ivoa_mapping[key] = wcs_to_ivoa[key]


def append_sequence_axis_to_wcs(wcs_object):
    """
    Appends a 1-to-1 dummy axis to a WCS object.
    """
    dummy_number = wcs_object.naxis + 1
    wcs_header = wcs_object.to_header()
    wcs_header.append((f"CTYPE{dummy_number}", "ITER",
                       "A unitless iteration-by-one axis."))
    wcs_header.append((f"CRPIX{dummy_number}", 0.,
                       "Pixel coordinate of reference point"))
    wcs_header.append((f"CDELT{dummy_number}", 1.,
                       "Coordinate increment at reference point"))
    wcs_header.append((f"CRVAL{dummy_number}", 0.,
                       "Coordinate value at reference point"))
    wcs_header.append((f"CUNIT{dummy_number}", "pix",
                       "Coordinate value at reference point"))
    wcs_header["WCSAXES"] = dummy_number
    return WCS(wcs_header)


def _pixel_keep(wcs_object):
    """Returns the value of the _pixel_keep attribute if available
    else returns the array of all pixel dimension present.

    Parameters
    ----------
    wcs_object : `astropy.wcs.WCS` or alike object

    Returns
    -------
    list or `np.ndarray` object
    """
    if hasattr(wcs_object, "_pixel_keep"):
        return wcs_object._pixel_keep
    return np.arange(wcs_object.pixel_n_dim)


def convert_between_array_and_pixel_axes(axis, naxes):
    """Reflects axis index about center of number of axes.

    This is used to convert between array axes in numpy order and pixel axes in WCS order.
    Works in both directions.

    Parameters
    ----------
    axis: `numpy.ndarray` of `int`
        The axis number(s) before reflection.

    naxes: `int`
        The number of array axes.

    Returns
    -------
    reflected_axis: `numpy.ndarray` of `int`
        The axis number(s) after reflection.
    """
    # Check type of input.
    if not isinstance(axis, np.ndarray):
        raise TypeError("input must be of array type. Got type: {type(axis)}")
    if axis.dtype.char not in np.typecodes['AllInteger']:
        raise TypeError("input dtype must be of int type.  Got dtype: {axis.dtype})")
    # Convert negative indices to positive equivalents.
    axis[axis < 0] += naxes
    if any(axis > naxes - 1):
        raise IndexError("Axis out of range.  "
                         f"Number of axes = {naxes}; Axis numbers requested = {axes}")
    # Reflect axis about center of number of axes.
    reflected_axis = naxes - 1 - axis

    return reflected_axis


def pixel_axis_to_world_axes(pixel_axis, axis_correlation_matrix):
    """
    Retrieves the indices of the world axis physical types corresponding to a pixel axis.

    Parameters
    ----------
    pixel_axis: `int`
        The pixel axis index/indices for which the world axes are desired.

    axis_correlation_matrix: `numpy.ndarray` of `bool`
        2D boolean correlation matrix defining the dependence between the pixel and world axes.
        Format same as `astropy.wcs.BaseLowLevelWCS.axis_correlation_matrix`.

    Returns
    -------
    world_axes: `numpy.ndarray`
        The world axis indices corresponding to the pixel axis.
    """
    return np.arange(axis_correlation_matrix.shape[0])[axis_correlation_matrix[:, pixel_axis]]


def world_axis_to_pixel_axes(world_axis, axis_correlation_matrix):
    """
    Gets the pixel axis indices corresponding to the index of a world axis physical type.

    Parameters
    ----------
    world_axis: `int`
        The index of the physical type for which the pixes axes are desired.

    axis_correlation_matrix: `numpy.ndarray` of `bool`
        2D boolean correlation matrix defining the dependence between the pixel and world axes.
        Format same as `astropy.wcs.BaseLowLevelWCS.axis_correlation_matrix`.

    Returns
    -------
    pixel_axes: `numpy.ndarray`
        The pixel axis indices corresponding to the world axis.
    """
    return np.arange(axis_correlation_matrix.shape[1])[axis_correlation_matrix[world_axis]]


def pixel_axis_to_physical_types(pixel_axis, wcs):
    """
    Gets the world axis physical types corresponding to a pixel axis.

    Parameters
    ----------
    pixel_axis: `int`
        The pixel axis number(s) for which the world axis numbers are desired.

    wcs: `astropy.wcs.BaseLowLevelWCS`
        The WCS object defining the relationship between pixel and world axes.

    Returns
    -------
    physical_types: `numpy.ndarray` of `str`
        The physical types corresponding to the pixel axis.
    """
    return np.array(wcs.world_axis_physical_types)[wcs.axis_correlation_matrix[:, pixel_axis]]


def physical_type_to_pixel_axes(physical_type, wcs):
    """
    Gets the pixel axis indices corresponding to a world axis physical type.

    Parameters
    ----------
    physical_type: `int`
        The pixel axis number(s) for which the world axis numbers are desired.

    wcs: `astropy.wcs.BaseLowLevelWCS`
        The WCS object defining the relationship between pixel and world axes.

    Returns
    -------
    pixel_axes: `numpy.ndarray`
        The pixel axis indices corresponding to the physical type.
    """
    world_axis = physical_type_to_world_axis(physical_type, wcs.world_axis_physical_types)
    return world_axis_to_pixel_axes(world_axis, wcs.axis_correlation_matrix)


def physical_type_to_world_axis(physical_type, world_axis_physical_types):
    """
    Returns world axis index of a physical type based on WCS world_axis_physical_types.

    Input can be a substring of a physical type, so long as it is unique.

    Parameters
    ----------
    physical_type: `str`
        The physical type or a substring unique to a physical type.

    world_axis_physical_types: sequence of `str`
        All available physical types.  Ordering must be same as
        `astropy.wcs.BaseLowLevelWCS.world_axis_physical_types`

    Returns
    -------
    world_axis: `numbers.Integral`
        The world axis index of the physical type.
    """
    # Find world axis index described by physical type.
    widx = np.where(world_axis_physical_types == physical_type)[0]
    # If physical type does not correspond to entry in world_axis_physical_types,
    # check if it is a substring of any physical types.
    if len(widx) == 0:
        widx = [physical_type in world_axis_physical_type
                for world_axis_physical_type in world_axis_physical_types]
        widx = np.arange(len(world_axis_physical_types))[widx]
    if len(widx) != 1:
        raise ValueError(
                "Input does not uniquely correspond to a physical type."
                f" Expected unique substring of one of {world_axis_physical_types}."
                f"  Got: {physical_type}")
    # Return axes with duplicates removed.
    return widx[0]


def get_dependent_pixel_axes(pixel_axis, axis_correlation_matrix):
    """
    Find indices of all pixel axes associated with the world axes linked to the input pixel axis.

    For example, say the input pixel axis is 0 and it is associated with two world axes
    corresponding to longitude and latitude. Let's also say that pixel axis 1 is also
    associated with longitude and latitude. Thus, this function would return pixel axes 0 and 1.
    On the other hand let's say pixel axis 2 is associated with only one world axis,
    e.g. wavelength, which does not depend on any other pixel axis (i.e. it is independent).
    In that case this function would only return pixel axis 2.
    Both input and output pixel axis indices are in the WCS ordering convention
    (reverse of numpy ordering convention).
    The returned axis indices include the input axis.

    Parameters
    ----------
    wcs_axis: `int`
        Index of axis (in WCS ordering convention) for which dependent axes are desired.

    axis_correlation_matrix: `numpy.ndarray` of `bool`
        2D boolean correlation matrix defining the dependence between the pixel and world axes.
        Format same as `astropy.wcs.BaseLowLevelWCS.axis_correlation_matrix`.

    Returns
    -------
    dependent_pixel_axes: `np.ndarray` of `int`
        Sorted indices of pixel axes dependent on input axis in WCS ordering convention.
    """
    # The axis_correlation_matrix is (n_world, n_pixel) but we want to know
    # which pixel coordinates are linked to which other pixel coordinates.
    # To do this we take a column from the matrix and find if there are
    # any entries in common with all other columns in the matrix.
    world_dep = axis_correlation_matrix[:, pixel_axis:pixel_axis + 1]
    dependent_pixel_axes = np.sort(np.nonzero((world_dep & axis_correlation_matrix).any(axis=0))[0])
    return dependent_pixel_axes


def get_dependent_array_axes(array_axis, axis_correlation_matrix):
    """
    Find indices of all array axes associated with the world axes linked to the input array axis.

    For example, say the input array axis is 0 and it is associated with two world axes
    corresponding to longitude and latitude. Let's also say that array axis 1 is also
    associated with longitude and latitude. Thus, this function would return array axes 0 and 1.
    Note the the output axes include the input axis. On the other hand let's say
    array axis 2 is associated with only one world axis, e.g. wavelength,
    which does not depend on any other array axis (i.e. it is independent).
    In that case this function would only return array axis 2.
    Both input and output array axis indices are in the numpy array ordering convention
    (reverse of WCS ordering convention).
    The returned axis indices include the input axis.

    Parameters
    ----------
    array_axis: `int`
        Index of array axis (in numpy ordering convention) for which dependent axes are desired.

    axis_correlation_matrix: `numpy.ndarray` of `bool`
        2D boolean correlation matrix defining the dependence between the pixel and world axes.
        Format same as `astropy.wcs.BaseLowLevelWCS.axis_correlation_matrix`.

    Returns
    -------
    dependent_array_axes: `np.ndarray` of `int`
        Sorted indices of array axes dependent on input axis in numpy ordering convention.
    """
    naxes = axis_correlation_matrix.shape[1]
    pixel_axis = convert_between_array_and_pixel_axes(np.array([array_axis], dtype=int), naxes)[0]
    dependent_pixel_axes = get_dependent_pixel_axes(pixel_axis, axis_correlation_matrix)
    dependent_array_axes = convert_between_array_and_pixel_axes(dependent_pixel_axes, naxes)
    return np.sort(dependent_array_axes)


def get_dependent_world_axes(world_axis, axis_correlation_matrix):
    """
    Given a WCS world axis index, return indices of dependent WCS world axes.

    Both input and output axis indices are in the WCS ordering convention
    (reverse of numpy ordering convention). The returned axis indices include the input axis.

    Parameters
    ----------
    world_axis: `int`
        Index of axis (in WCS ordering convention) for which dependent axes are desired.

    axis_correlation_matrix: `numpy.ndarray` of `bool`
        2D boolean correlation matrix defining the dependence between the pixel and world axes.
        Format same as `astropy.wcs.BaseLowLevelWCS.axis_correlation_matrix`.

    Returns
    -------
    dependent_world_axes: `np.ndarray` of `int`
        Sorted indices of pixel axes dependent on input axis in WCS ordering convention.
    """
    # The axis_correlation_matrix is (n_world, n_pixel) but we want to know
    # which world coordinates are linked to which other world coordinates.
    # To do this we take a row from the matrix and find if there are
    # any entries in common with all other rows in the matrix.
    pixel_dep = axis_correlation_matrix[world_axis:world_axis + 1].T
    dependent_world_axes = np.sort(np.nonzero((pixel_dep & axis_correlation_matrix).any(axis=1))[0])
    return dependent_world_axes


def get_dependent_physical_types(physical_type, wcs):
    """
    Given a world axis physical type, return the dependent physical types including the input type.

    Parameters
    ----------
    physical_type: `str`
        The world axis physical types whose dependent physical types are desired.

    wcs: `astropy.wcs.BaseLowLevelWCS`
        The WCS object defining the relationship between pixel and world axes.

    Returns
    -------
    dependent_physical_types: `np.ndarray` of `str`
        Physical types dependent on the input physical type.
    """
    world_axis_physical_types = wcs.world_axis_physical_types
    world_axis = physical_type_to_world_axis(physical_type, world_axis_physical_types)
    dependent_world_axes = get_dependent_world_axes(world_axis, wcs.axis_correlation_matrix)
    dependent_physical_types = np.array(world_axis_physical_types)[dependent_world_axes]
    return dependent_physical_types
