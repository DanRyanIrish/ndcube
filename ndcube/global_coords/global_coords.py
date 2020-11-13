from collections import OrderedDict
from collections.abc import Mapping

__all__ = ['GlobalCoords']


class GlobalCoords(Mapping):
    """
    A structured representation of coordinate information applicable to a whole NDCube.
    """
    def __init__(self, ndcube):
        """
        Init method.
        """
        super().__init__()
        self._ndcube = ndcube

        self._all_coords = OrderedDict()
        self._internal_coords = OrderedDict()

    @property
    def _all_coords(self):
        """
        Establish _all_coords as a property as some _internal_coords.
        """
        return self._internal_coords

    def add(self, name, physical_type, coords):
        """
        Add a new coordinate to the collection.
        """
        self._internal_coords[name] = [physical_type, coords]

    def remove(self, name):
        """
        Remove a coordinate from the collection
        """
        del self._internal_coords[name]

    @property
    def names(self):
        """
        A tuple of all the names or keys.
        """
        return tuple(self._all_coords.keys())

    @property
    def physical_types(self):
        """
        A tuple of all physical types, one per coordinate.
        """
        return tuple(item[0] for item in self._all_coords.values())

    def __getitem__(self, item):
        """
        Index the collection by a name.
        """
        return self._all_coords[item]

    def __iter__(self):
        """
        Iterate over the collection.
        """
        return iter(self._all_coords)

    def __len__(self):
        """
        Establish the length of the collection.
        """
        return len(self._all_coords)

    @_all_coords.setter
    def _all_coords(self, value):
        """
        Use setter to set _all_coords
        """
        self.__all_coords = value
