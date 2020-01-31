from bisect import bisect
from copy import deepcopy
import enum
from itertools import product
import logging

from nibabel.affines import apply_affine
from nibabel.streamlines.tractogram import (Tractogram,
                                            PerArraySequenceDict,
                                            PerArrayDict)
import numpy as np

from dipy.io.stateful_tractogram import Space
from dipy.io.utils import get_reference_info



class StatefulSurface(object):
    """ Class for stateful representation of collections of streamlines
    Object designed to be identical no matter the file format
    (trk, tck, vtk, fib, dpy). Facilitate transformation between space and
    data manipulation for each streamline / point.
    """

    def __init__(self, surface, reference, space,
                 data_per_vertex=None, data_per_triangle=None):
        """ Create a strict, state-aware, robust tractogram

        Parameters
        ----------
        surface : list or ArraySequence
            Streamlines of the tractogram
        reference : Nifti or Trk filename, Nifti1Image or TrkFile,
            Nifti1Header, trk.header (dict) or another Stateful Tractogram
            Reference that provides the spatial attributes.
            Typically a nifti-related object from the native diffusion used for
            streamlines generation
        space : Enum (dipy.io.stateful_tractogram.Space)
            Current space in which the streamlines are (vox, voxmm or rasmm)
            Typically after tracking the space is VOX, after nibabel loading
            the space is RASMM
        data_per_vertex : array
            TODO
        data_per_triangle : array
            TODO

        Notes
        -----
        Very important to respect the convention, verify that streamlines
        match the reference and are effectively in the right space.

        Any change to the number of streamlines, data_per_point or
        data_per_streamline requires particular verification.

        In a case of manipulation not allowed by this object, use Nibabel
        directly and be careful.
        """

        if isinstance(streamlines, Streamlines):
            streamlines = streamlines.copy()
        self._tractogram = Tractogram(streamlines,
                                      data_per_point=data_per_point,
                                      data_per_streamline=data_per_streamline)

        space_attributes = get_reference_info(reference)
        if space_attributes is None:
            raise TypeError('Reference MUST be one of the following:\n' +
                            'Nifti or Trk filename, Nifti1Image or TrkFile, ' +
                            'Nifti1Header or trk.header (dict)')

        (self._affine, self._dimensions,
         self._voxel_sizes, self._voxel_order) = space_attributes
        self._inv_affine = np.linalg.inv(self._affine)

        if space not in Space:
            raise ValueError('Space MUST be from Space enum, e.g Space.VOX')
        self._space = space

        if not isinstance(shifted_origin, bool):
            raise TypeError('shifted_origin MUST be a boolean')
        self._shifted_origin = shifted_origin
        logging.debug(self)

    def __str__(self):
        """ Generate the string for printing """
        text = 'Affine: \n{}'.format(
            np.array2string(self._affine,
                            formatter={'float_kind': lambda x: "%.6f" % x}))
        text += '\ndimensions: {}'.format(
            np.array2string(self._dimensions))
        text += '\nvoxel_sizes: {}'.format(
            np.array2string(self._voxel_sizes,
                            formatter={'float_kind': lambda x: "%.2f" % x}))
        text += '\nvoxel_order: {}'.format(self._voxel_order)

        text += '\nstreamline_count: {}'.format(self._get_streamline_count())
        text += '\npoint_count: {}'.format(self._get_point_count())
        text += '\ndata_per_streamline keys: {}'.format(
            self.data_per_point.keys())
        text += '\ndata_per_point keys: {}'.format(
            self.data_per_streamline.keys())

        return text

    def __len__(self):
        """ Define the length of the object """
        return self._get_streamline_count()

    @property
    def space_attributes(self):
        """ Getter for spatial attribute """
        return self._affine, self._dimensions, self._voxel_sizes, \
            self._voxel_order

    @property
    def space(self):
        """ Getter for the current space """
        return self._space

    @property
    def affine(self):
        """ Getter for the reference affine """
        return self._affine

    @property
    def dimensions(self):
        """ Getter for the reference dimensions """
        return self._dimensions

    @property
    def voxel_sizes(self):
        """ Getter for the reference voxel sizes """
        return self._voxel_sizes

    @property
    def voxel_order(self):
        """ Getter for the reference voxel order """
        return self._voxel_order

    @property
    def surface(self):
        """ Partially safe getter for streamlines """
        raise NotImplementedError()

    def get_surface_copy(self):
        """ Safe getter for streamlines (for slicing) """
        raise NotImplementedError()

    @surface.setter
    def surface(self, surf):
        """ surface streamlines. Creating a new object would be less risky.

        Parameters
        ----------
        streamlines : list or ArraySequence (list and deepcopy recommanded)
            Streamlines of the tractogram
        """
        raise NotImplementedError()

    def to_vox(self):
        """ Safe function to transform streamlines and update state """
        if self._space == Space.VOXMM:
            self._voxmm_to_vox()
        elif self._space == Space.RASMM:
            self._rasmm_to_vox()

    def to_voxmm(self):
        """ Safe function to transform streamlines and update state """
        if self._space == Space.VOX:
            self._vox_to_voxmm()
        elif self._space == Space.RASMM:
            self._rasmm_to_voxmm()

    def to_rasmm(self):
        """ Safe function to transform streamlines and update state """
        if self._space == Space.VOX:
            self._vox_to_rasmm()
        elif self._space == Space.VOXMM:
            self._voxmm_to_rasmm()

    def to_space(self, target_space):
        """ Safe function to transform streamlines to a particular space using
        an enum and update state """
        if target_space == Space.VOX:
            self.to_vox()
        elif target_space == Space.VOXMM:
            self.to_voxmm()
        elif target_space == Space.RASMM:
            self.to_rasmm()
        else:
            logging.error('Unsupported target space, please use Enum in '
                          'dipy.io.stateful_tractogram')

    def compute_bounding_box(self):
        """ Compute the bounding box of the streamlines in their current state

        Returns
        -------
        output : ndarray
            8 corners of the XYZ aligned box, all zeros if no streamlines
        """
        raise NotImplementedError()

    def is_bbox_in_vox_valid(self):
        """ Verify that the bounding box is valid in voxel space.
        Negative coordinates or coordinates above the volume dimensions
        are considered invalid in voxel space.

        Returns
        -------
        output : bool
            Are the streamlines within the volume of the associated reference
        """
        raise NotImplementedError()

    def remove_invalid_streamlines(self):
        """ Remove streamlines with invalid coordinates from the object.
        Will also remove the data_per_point and data_per_streamline.
        Invalid coordinates are any X,Y,Z values above the reference
        dimensions or below zero
        Returns
        -------
        output : tuple
            Tuple of two list, indices_to_remove, indices_to_keep
        """
        raise NotImplementedError()

    def _vox_to_voxmm(self):
        """ Unsafe function to transform streamlines """
        raise NotImplementedError()

    def _voxmm_to_vox(self):
        """ Unsafe function to transform streamlines """
        raise NotImplementedError()

    def _vox_to_rasmm(self):
        """ Unsafe function to transform streamlines """
        raise NotImplementedError()

    def _rasmm_to_vox(self):
        """ Unsafe function to transform streamlines """
        raise NotImplementedError()

    def _voxmm_to_rasmm(self):
        """ Unsafe function to transform streamlines """
        raise NotImplementedError()

    def _rasmm_to_voxmm(self):
        """ Unsafe function to transform streamlines """
        raise NotImplementedError()
