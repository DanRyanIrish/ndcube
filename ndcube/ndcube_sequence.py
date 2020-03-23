import numpy as np

import astropy.units as u

from ndcube import utils
from ndcube.mixins.sequence_plotting import NDCubeSequencePlotMixin

__all__ = ['NDCubeSequence']


class NDCubeSequenceBase:
    """
    Class representing list of cubes.

    Parameters
    ----------
    data_list : `list`
        List of cubes.

    meta : `dict` or None
        The header of the NDCubeSequence.

    common_axis: `int` or None
        The data axis which is common between the NDCubeSequence and the Cubes within.
        For example, if the Cubes are sequenced in chronological order and time is
        one of the zeroth axis of each Cube, then common_axis should be se to 0.
        This enables the option for the NDCubeSequence to be indexed as though it is
        one single Cube.
    """

    def __init__(self, data_list, meta=None, common_axis=None, **kwargs):
        self.data = data_list
        self.meta = meta
        if common_axis is not None:
            self._common_axis = int(common_axis)
        else:
            self._common_axis = common_axis
        self._sequence_axis_name = "meta.obs.sequence"

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def _dimensions(self):
        dimensions = [len(self.data) * u.pix] + list(self.data[0].dimensions)
        if len(dimensions) > 1:
            # If there is a common axis, length of cube's along it may not
            # be the same. Therefore if the lengths are different,
            # represent them as a tuple of all the values, else as an int.
            if self._common_axis is not None:
                common_axis_lengths = [cube.data.shape[self._common_axis] for cube in self.data]
                if len(np.unique(common_axis_lengths)) != 1:
                    common_axis_dimensions = [cube.dimensions[self._common_axis]
                                              for cube in self.data]
                    dimensions[self._common_axis + 1] = u.Quantity(
                        common_axis_dimensions, unit=common_axis_dimensions[0].unit)
        return tuple(dimensions)

    @property
    def world_axis_physical_types(self):
        return tuple([self._sequence_axis_name] + list(self.data[0].world_axis_physical_types))

    def pixel_axes_to_world_types(self, *axes):
        """
        Retrieve the world axis physical types for each pixel axis.

        This differs from world_axis_physical_types in that it provides an explicit
        mapping between pixel axes and physical types, including dependent physical
        types.

        Parameters
        ----------
        axes: `int` or multiple `int`
            Axis number in numpy ordering of axes for which real world physical types
            are desired.
            axes=None implies axis names for all axes will be returned.

        Returns
        -------
        axes_names: `tuple` of `str`
            The world axis physical types corresponding to each axis.
            If more than one physical type found for an axis, that axis's entry will
            be a tuple of `str`.

        """
        # Parse user input.
        if axes == ():
            axes = tuple(range(n_dimensions))
        elif isinstance(axes, int):
            axes = (axes,)

        axes = np.array(axes)
        n_axes = len(axes)
        axes_names = np.array([None] * n_axes, dtype=object)

        # If sequence axis in axes, get names for it separately.
        if 0 in axes:
            sequence_axes_names = utils.sequence._get_axis_extra_coord_names_and_units(self.data, None)[0]
            sequence_index = np.array([axis == 0 for axis in axes])
            if sequence_axes_names:
                if isinstance(sequence_axes_names, str):
                    sequence_axes_names = [sequence_axes_names]
                else:
                    sequence_axes_names = list(sequence_axes_names)
            else:
                sequence_axes_names = []
            axes_names[sequence_index] = tuple([self._sequence_axis_name] + sequence_axes_names)
            cube_indices = np.invert(sequence_index)
            cube_axes = axes[cube_indices]
        else:
            cube_indices = np.ones(n_axes, dtype=bool)
            cube_axes = axes

        # Get world types from cube axes.
        if len(cube_axes) > 0:
            cube_axes_names = np.array(self.data[0].pixel_axes_to_world_types(*cube_axes))
            axes_names[cube_indices] = cube_axes_names

        return tuple(axes_names)

    def world_types_to_pixel_axes(self, *axes_names):
        raise NotImplementedError()

    @property
    def cube_like_dimensions(self):
        if not isinstance(self._common_axis, int):
            raise TypeError("Common axis must be set.")
        dimensions = list(self._dimensions)
        cube_like_dimensions = list(self._dimensions[1:])
        if dimensions[self._common_axis + 1].isscalar:
            cube_like_dimensions[self._common_axis] = u.Quantity(
                dimensions[0].value * dimensions[self._common_axis + 1].value, unit=u.pix)
        else:
            cube_like_dimensions[self._common_axis] = sum(dimensions[self._common_axis + 1])
        # Combine into single Quantity
        cube_like_dimensions = u.Quantity(cube_like_dimensions, unit=u.pix)
        return cube_like_dimensions

    @property
    def cube_like_world_axis_physical_types(self):
        return self.data[0].world_axis_physical_types

    def cube_like_pixel_axes_to_world_types(self, *axes):
        raise NotImplementedError()

    def cube_like_world_types_to_pixel_axes(self, *axes_names):
        raise NotImplementedError()

    def __getitem__(self, item):
        if len(self.dimensions) == 1:
            return self.data[item]
        else:
            return utils.sequence.slice_sequence(self, item)

    @property
    def index_as_cube(self):
        """
        Method to slice the NDCubesequence instance as a single cube.

        Example
        -------
        >>> # Say we have three Cubes each cube has common_axis=0 is time and shape=(3,3,3)
        >>> data_list = [cubeA, cubeB, cubeC] # doctest: +SKIP
        >>> cs = NDCubeSequence(data_list, meta=None, common_axis=0) # doctest: +SKIP
        >>> # return zeroth time slice of cubeB in via normal NDCubeSequence indexing.
        >>> cs[1,:,0,:] # doctest: +SKIP
        >>> # Return same slice using this function
        >>> cs.index_sequence_as_cube[3:6, 0, :] # doctest: +SKIP
        """
        if self._common_axis is None:
            raise ValueError("common_axis cannot be None")
        return _IndexAsCubeSlicer(self)

    @property
    def common_axis_extra_coords(self):
        if not isinstance(self._common_axis, int):
            raise ValueError("Common axis is not set.")
        # Get names and units of coords along common axis.
        axis_coord_names, axis_coord_units = utils.sequence._get_axis_extra_coord_names_and_units(
            self.data, self._common_axis)
        # Compile dictionary of common axis extra coords.
        if axis_coord_names is not None:
            return utils.sequence._get_int_axis_extra_coords(
                self.data, axis_coord_names, axis_coord_units, self._common_axis)
        else:
            return None

    @property
    def sequence_axis_extra_coords(self):
        sequence_coord_names, sequence_coord_units = \
            utils.sequence._get_axis_extra_coord_names_and_units(self.data, None)
        if sequence_coord_names is not None:
            # Define empty dictionary which will hold the extra coord
            # values not assigned a cube data axis.
            sequence_extra_coords = {}
            # Define list of None signifying unit of each coord.  It will
            # be filled in in for loop below.
            sequence_coord_units = [None] * len(sequence_coord_names)
            # Iterate through cubes and populate values of each extra coord
            # not assigned a cube data axis.
            cube_extra_coords = [cube.extra_coords for cube in self.data]
            for i, coord_key in enumerate(sequence_coord_names):
                coord_values = np.array([None] * len(self.data), dtype=object)
                for j, cube in enumerate(self.data):
                    # Construct list of coord values from each cube for given extra coord.
                    try:
                        coord_values[j] = cube_extra_coords[j][coord_key]["value"]
                        # Determine whether extra coord is a quantity by checking
                        # whether any one value has a unit. As we are not
                        # assuming that all cubes have the same extra coords
                        # along the sequence axis, we will keep checking as we
                        # move through the cubes until all cubes are checked or
                        # we have found a unit.
                        if (isinstance(cube_extra_coords[j][coord_key]["value"], u.Quantity) and
                                not sequence_coord_units[i]):
                            sequence_coord_units[i] = cube_extra_coords[j][coord_key]["value"].unit
                    except KeyError:
                        pass
                # If the extra coord is normally a Quantity, replace all
                # None occurrences in coord value array with a NaN, and
                # convert coord_values from an array of Quantities to a
                # single Quantity of length equal to number of cubes in
                # sequence.
                w_none = np.where(coord_values == None)[0]  # NOQA
                if sequence_coord_units[i]:
                    # This part of if statement is coded in an apparently
                    # round about way but necessitated because you can't
                    # put a NaN quantity into an array and keep its unit.
                    w_not_none = np.where(coord_values != None)[0]  # NOQA
                    coord_values = u.Quantity(list(coord_values[w_not_none]),
                                              unit=sequence_coord_units[i])
                    coord_values = list(coord_values.value)
                    for index in w_none:
                        coord_values.insert(index, np.nan)
                    coord_values = u.Quantity(coord_values, unit=sequence_coord_units[i]).flatten()
                else:
                    coord_values[w_none] = np.nan
                sequence_extra_coords[coord_key] = coord_values
        else:
            sequence_extra_coords = None
        return sequence_extra_coords

    def explode_along_axis(self, axis):
        """
        Separates slices of NDCubes in sequence along a given cube axis into
        (N-1)DCubes.

        Parameters
        ----------

        axis : `int`
            The axis along which the data is to be changed.
        """
        # if axis is None then set axis as common axis.
        if self._common_axis is not None:
            if self._common_axis != axis:
                raise ValueError("axis and common_axis should be equal.")
        # is axis is -ve then calculate the axis from the length of the dimensions of one cube
        if axis < 0:
            axis = len(self.dimensions[1::]) + axis
        # To store the resultant cube
        result_cubes = []
        # All slices are initially initialised as slice(None, None, None)
        result_cubes_slice = [slice(None, None, None)] * len(self[0].data.shape)
        # the range of the axis that needs to be sliced
        range_of_axis = self[0].data.shape[axis]
        for ndcube in self.data:
            for index in range(range_of_axis):
                # setting the slice value to the index so that the slices are done correctly.
                result_cubes_slice[axis] = index
                # appending the sliced cubes in the result_cube list
                result_cubes.append(ndcube.__getitem__(tuple(result_cubes_slice)))
        # creating a new sequence with the result_cubes keeping the meta and common axis as axis
        return self._new_instance(result_cubes, meta=self.meta)

    def __repr__(self):
        return (
            """NDCubeSequence
---------------------
Length of NDCubeSequence:  {length}
Shape of 1st NDCube: {shapeNDCube}
Axis Types of 1st NDCube: {axis_type}
""".format(length=self.dimensions[0], shapeNDCube=self.dimensions[1::],
                axis_type=self.world_axis_physical_types[1:]))

    @classmethod
    def _new_instance(cls, data_list, meta=None, common_axis=None):
        """
        Instantiate a new instance of this class using given data.
        """
        return cls(data_list, meta=meta, common_axis=common_axis)


class NDCubeSequence(NDCubeSequenceBase, NDCubeSequencePlotMixin):
    pass


"""
Cube Sequence Helpers
"""


class _IndexAsCubeSlicer:
    """
    Helper class to make slicing in index_as_cube sliceable/indexable like a
    numpy array.

    Parameters
    ----------
    seq : `ndcube.NDCubeSequence`
        Object of NDCubeSequence.
    """

    def __init__(self, seq):
        self.seq = seq

    def __getitem__(self, item):
        return utils.sequence._index_sequence_as_cube(self.seq, item)
