"""
This file contains a set of common fixtures to get a set of different but
predicable NDCube objects.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time
from astropy.wcs import WCS

from ndcube import ExtraCoords, NDCube


@pytest.fixture
def wcs_4d():
    header = {
        'CTYPE4': 'HPLN-TAN',
        'CUNIT4': 'arcsec',
        'CDELT4': 5,
        'CRPIX4': 5,
        'CRVAL4': 0,

        'CTYPE3': 'HPLT-TAN',
        'CUNIT3': 'arcsec',
        'CDELT3': 20,
        'CRPIX3': 0,
        'CRVAL3': 0,

        'CTYPE2': 'WAVE    ',
        'CUNIT2': 'Angstrom',
        'CDELT2': 0.2,
        'CRPIX2': 0,
        'CRVAL2': 0,

        'CTYPE1': 'TIME    ',
        'CUNIT1': 'min',
        'CDELT1': 0.4,
        'CRPIX1': 0,
        'CRVAL1': 0,
        }
    return WCS(header=header)

@pytest.fixture
def wcs_3d():
    header = {
        'CTYPE1': 'WAVE    ',
        'CUNIT1': 'Angstrom',
        'CDELT1': 0.2,
        'CRPIX1': 0,
        'CRVAL1': 10,

        'CTYPE2': 'HPLT-TAN',
        'CUNIT2': 'arcsec',
        'CDELT2': 5,
        'CRPIX2': 5,
        'CRVAL2': 0,

        'CTYPE3': 'HPLN-TAN',
        'CUNIT3': 'arcsec',
        'CDELT3': 10,
        'CRPIX3': 0,
        'CRVAL3': 0,
        }

    return WCS(header=header)

@pytest.fixture
def wcs_2d_spatial():
    spatial = {
        'CTYPE1': 'HPLT-TAN',
        'CUNIT1': 'arcsec',
        'CDELT1': 2,
        'CRPIX1': 5,
        'CRVAL1': 0,

        'CTYPE2': 'HPLN-TAN',
        'CUNIT2': 'arcsec',
        'CDELT2': 4,
        'CRPIX2': 5,
        'CRVAL2': 0,
    }
    return WCS(header=spatial)

@pytest.fixture
def wcs_1d():
    spatial = {
        'CNAME1': 'spectral',
        'CTYPE1': 'WAVE',
        'CUNIT1': 'nm',
        'CDELT1': 0.5,
        'CRPIX1': 2,
        'CRVAL1': 0.5,
    }
    return WCS(header=spatial)


def data_nd(shape):
    nelem = np.product(shape)
    return np.arange(nelem).reshape(shape)


def generate_time_extra_coord(data_cube):
    shape = data_cube.shape[-1]
    lut = Time("2020-02-02T00:00:00", format="isot") + np.linspace(0, shape * 10, num=shape, endpoint=False) * u.s
    return ExtraCoords.from_lookup_tables(["extra_time"], [0], [lut])


@pytest.fixture
def ndcube_4d_simple(wcs_4d):
    shape = (5, 8, 10, 12)
    data_cube = data_nd(shape)
    return NDCube(data_cube, wcs=wcs_4d)


@pytest.fixture
def ndcube_4d_uncertainty(wcs_4d):
    shape = (5, 8, 10, 12)
    data_cube = data_nd(shape)
    uncertainty = np.sqrt(data_cube)
    return NDCube(data_cube, wcs=wcs_4d, uncertainty=uncertainty)


@pytest.fixture
def ndcube_4d_mask(wcs_4d):
    shape = (5, 8, 10, 12)
    data_cube = data_nd(shape)
    uncertainty = np.sqrt(data_cube)
    mask = data_cube % 2
    return NDCube(data_cube, wcs=wcs_4d, uncertainty=uncertainty, mask=mask)


@pytest.fixture
def ndcube_4d_extra_coords(wcs_4d):
    shape = (5, 8, 10, 12)
    data_cube = data_nd(shape)
    extra_coords = generate_time_extra_coord(data_cube)
    return NDCube(data_cube, wcs=wcs_4d, extra_coords=extra_coords)


@pytest.fixture
def ndcube_4d_unit_uncertainty(wcs_4d):
    shape = (5, 8, 10, 12)
    data_cube = data_nd(shape)
    uncertainty = np.sqrt(data_cube)
    return NDCube(data_cube, wcs=wcs_4d,
                  unit=u.J, uncertainty=uncertainty)


@pytest.fixture
def ndcube_4d(request):
    """
    This is a meta fixture for parametrizing all the 4D ndcubes.
    """
    return request.getfixturevalue("ndcube_4d_" + request.param)


@pytest.fixture
def ndcube_2d_simple(wcs_2d_spatial):
    shape = (10, 12)
    data_cube = data_nd(shape)
    return NDCube(data_cube, wcs=wcs_2d_spatial)


@pytest.fixture
def ndcube_2d(request):
    """
    This is a meta fixture for parametrizing all the 2D ndcubes.
    """
    return request.getfixturevalue("ndcube_2d_" + request.param)


@pytest.fixture
def ndcube_1d_simple(wcs_1d):
    shape = (10,)
    data_cube = data_nd(shape)
    return NDCube(data_cube, wcs=wcs_1d)
