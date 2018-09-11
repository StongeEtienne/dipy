import random
import warnings

import numpy as np

from dipy.align import Bunch
from dipy.core.sphere import HemiSphere
from dipy.tracking.local.localtrack import local_tracker, pft_tracker
from dipy.tracking.local.tissue_classifier import ConstrainedTissueClassifier
from dipy.tracking import utils


# enum TissueClass (tissue_classifier.pxd) is not accessible
# from here. To be changed when minimal cython version > 0.21.
# cython 0.21 - cpdef enum to export values into Python-level namespace
# https://github.com/cython/cython/commit/50133b5a91eea348eddaaad22a606a7fa1c7c457
TissueTypes = Bunch(OUTSIDEIMAGE=-1, INVALIDPOINT=0, TRACKPOINT=1, ENDPOINT=2)


class LocalTracking(object):

    @staticmethod
    def _get_voxel_size(affine):
        """Computes the voxel sizes of an image from the affine.

        Checks that the affine does not have any shear because local_tracker
        assumes that the data is sampled on a regular grid.

        """
        lin = affine[:3, :3]
        dotlin = np.dot(lin.T, lin)
        # Check that the affine is well behaved
        if not np.allclose(np.triu(dotlin, 1), 0., atol=1e-5):
            msg = ("The affine provided seems to contain shearing, data must "
                   "be acquired or interpolated on a regular grid to be used "
                   "with `LocalTracking`.")
            raise ValueError(msg)
        return np.sqrt(dotlin.diagonal())

    def __init__(self, direction_getter, tissue_classifier, seeds, affine,
                 step_size, max_cross=None, maxlen=500, fixedstep=True,
                 return_all=True, random_seed=None, unidirectional=False,
                 randomize_forward_direction=False, initial_directions=None):
        """Creates streamlines by using local fiber-tracking.

        Parameters
        ----------
        direction_getter : instance of DirectionGetter
            Used to get directions for fiber tracking.
        tissue_classifier : instance of TissueClassifier
            Identifies endpoints and invalid points to inform tracking.
        seeds : array (N, 3)
            Points to seed the tracking. Seed points should be given in point
            space of the track (see ``affine``).
        affine : array (4, 4)
            Coordinate space for the streamline point with respect to voxel
            indices of input data. This affine can contain scaling, rotational,
            and translational components but should not contain any shearing.
            An identity matrix can be used to generate streamlines in "voxel
            coordinates" as long as isotropic voxels were used to acquire the
            data.
        step_size : float
            Step size used for tracking.
        max_cross : int or None
            The maximum number of direction to track from each seed in crossing
            voxels. By default all initial directions are tracked.
        maxlen : int
            Maximum number of steps to track from seed. Used to prevent
            infinite loops.
        fixedstep : bool
            If true, a fixed stepsize is used, otherwise a variable step size
            is used.
        return_all : bool
            If true, return all generated streamlines, otherwise only
            streamlines reaching end points or exiting the image.
        random_seed : int
            The seed for the random seed generator (numpy.random.seed and
            random.seed).
        unidirectional : bool
            If true, the tracking is performed only in the forward direction.
            The seed position will be the first point of all streamlines.
        randomize_forward_direction : bool
            If true, the forward direction is randomized (multiplied by 1
            or -1). Otherwise, the provided forward direction is used.
        initial_directions: array (N, npeaks, 3)
            Initial direction to follow from the ``seed`` position. If
            ``max_cross`` is None, one streamline will be generated per peak
            per voxel. If None, `direction_getter.initial_direction` is used.
        """

        self.direction_getter = direction_getter
        self.tissue_classifier = tissue_classifier
        self.seeds = seeds
        self.unidirectional = unidirectional
        self.randomize_forward_direction = randomize_forward_direction
        self.initial_directions = initial_directions
        if affine.shape != (4, 4):
            raise ValueError("affine should be a (4, 4) array.")
        if step_size <= 0:
            raise ValueError("step_size must be greater than 0.")
        if maxlen < 1:
            raise ValueError("maxlen must be greater than 0.")
        if initial_directions is not None:
            if seeds.shape[0] != initial_directions.shape[0]:
                raise ValueError("initial_directions and seeds must have the "
                                 + "same shape[0].")
        if (self.initial_directions is None and self.unidirectional is True and
                self.randomize_forward_direction is False):
                warnings.warn("Unidirectional tractography will be performed "
                              + "without providing initial directions nor "
                              + "randomizing extracted initial forward "
                              + "directions. This can introduce directional "
                              + "biases in the reconstructed streamlines. "
                              + "See ``initial_directions`` and "
                              + "``randomize_forward_direction`` parameters.")
        self.affine = affine
        self._voxel_size = np.ascontiguousarray(self._get_voxel_size(affine),
                                                dtype=float)
        self.step_size = step_size
        self.fixed_stepsize = fixedstep
        self.max_cross = max_cross
        self.max_length = maxlen
        self.return_all = return_all
        self.random_seed = random_seed

    def _tracker(self, seed, first_step, streamline):
        return local_tracker(self.direction_getter,
                             self.tissue_classifier,
                             seed,
                             first_step,
                             self._voxel_size,
                             streamline,
                             self.step_size,
                             self.fixed_stepsize)

    def __iter__(self):
        # Make tracks, move them to point space and return
        track = self._generate_streamlines()
        return utils.move_streamlines(track, self.affine)

    def _generate_streamlines(self):
        """A streamline generator"""

        # Get inverse transform (lin/offset) for seeds
        inv_A = np.linalg.inv(self.affine)
        lin = inv_A[:3, :3]
        offset = inv_A[:3, 3]

        F = np.empty((self.max_length + 1, 3), dtype=float)
        B = F.copy()
        for i in range(len(self.seeds)):
            s = np.dot(lin, self.seeds[i]) + offset
            # Set the random seed in numpy and random
            if self.random_seed is not None:
                s_random_seed = hash(np.abs((np.sum(s)) + self.random_seed)) \
                    % (2**32 - 1)
                random.seed(s_random_seed)
                np.random.seed(s_random_seed)
            if self.initial_directions is None:
                directions = self.direction_getter.initial_direction(s)
            else:
                directions = []
                for d in self.initial_directions[i, :, :]:
                    if np.linalg.norm(d) > 0:
                        i = self.direction_getter.sphere.find_closest(d)
                        new_d = self.direction_getter.sphere.vertices[i]
                        if np.dot(d, new_d) < 0:
                            #  The direction is flipped if the original
                            #  direction is closer to the opposite direction.
                            #  This will happen if the Sphere is an Hemisphere.
                            new_d = new_d * -1
                        directions.append(new_d)
                directions = np.array(directions)
            if self.randomize_forward_direction:
                directions = [d * random.choice([1, -1]) for d in directions]
            directions = directions[:self.max_cross]

            if len(directions) == 0 and self.return_all:
                # only the seed position
                yield [s]

            for first_step in directions:
                stepsF = stepsB = 1
                stepsF, tissue_class = self._tracker(s, first_step, F)
                if not (self.return_all or
                        tissue_class == TissueTypes.ENDPOINT or
                        tissue_class == TissueTypes.OUTSIDEIMAGE):
                    continue
                if not self.unidirectional:
                    first_step = -first_step
                    stepsB, tissue_class = self._tracker(s, first_step, B)
                    if not (self.return_all or
                            tissue_class == TissueTypes.ENDPOINT or
                            tissue_class == TissueTypes.OUTSIDEIMAGE):
                        continue
                if stepsB == 1:
                    streamline = F[:stepsF].copy()
                else:
                    parts = (B[stepsB - 1:0:-1], F[:stepsF])
                    streamline = np.concatenate(parts, axis=0)
                yield streamline


class ParticleFilteringTracking(LocalTracking):

    def __init__(self, direction_getter, tissue_classifier, seeds, affine,
                 step_size, max_cross=None, maxlen=500,
                 pft_back_tracking_dist=2, pft_front_tracking_dist=1,
                 pft_max_trial=20, particle_count=15, return_all=True,
                 random_seed=None, unidirectional=False,
                 randomize_forward_direction=False, initial_directions=None,
                 min_wm_pve_before_stopping=0):
        r"""A streamline generator using the particle filtering tractography
        method [1]_.

        Parameters
        ----------
        direction_getter : instance of ProbabilisticDirectionGetter
            Used to get directions for fiber tracking.
        tissue_classifier : instance of ConstrainedTissueClassifier
            Identifies endpoints and invalid points to inform tracking.
        seeds : array (N, 3)
            Points to seed the tracking. Seed points should be given in point
            space of the track (see ``affine``).
        affine : array (4, 4)
            Coordinate space for the streamline point with respect to voxel
            indices of input data. This affine can contain scaling, rotational,
            and translational components but should not contain any shearing.
            An identity matrix can be used to generate streamlines in "voxel
            coordinates" as long as isotropic voxels were used to acquire the
            data.
        step_size : float
            Step size used for tracking.
        max_cross : int or None
            The maximum number of direction to track from each seed in crossing
            voxels. By default all initial directions are tracked.
        maxlen : int
            Maximum number of steps to track from seed. Used to prevent
            infinite loops.
        pft_back_tracking_dist : float
            Distance in mm to back track before starting the particle filtering
            tractography. The total particle filtering tractography distance is
            equal to back_tracking_dist + front_tracking_dist.
            By default this is set to 2 mm.
        pft_front_tracking_dist : float
            Distance in mm to run the particle filtering tractography after the
            the back track distance. The total particle filtering tractography
            distance is equal to back_tracking_dist + front_tracking_dist. By
            default this is set to 1 mm.
        pft_max_trial : int
            Maximum number of trial for the particle filtering tractography
            (Prevents infinite loops).
        particle_count : int
            Number of particles to use in the particle filter.
        return_all : bool
            If true, return all generated streamlines, otherwise only
            streamlines reaching end points or exiting the image.
        random_seed : int
            The seed for the random seed generator (numpy.random.seed and
            random.seed).
        unidirectional : bool
            If true, the tracking is performed only in the forward direction.
            The seed position will be the first point of all streamlines.
        randomize_forward_direction : bool
            If true, the forward direction is randomized (multiplied by 1
            or -1). Otherwise, the provided forward direction is used.
        initial_directions: array (N, npeaks, 3)
            Initial direction to follow from the ``seed`` position. If
            ``max_cross`` is None, one streamline will be generated per peak
            per voxel. If None, `direction_getter.initial_direction` is used.
        min_wm_pve_before_stopping : int
            Minimum white matter pve (1 - tissue_classifier.include_map -
            tissue_classifier.exclude_map) to reach before allowing the
            tractography to stop.

        References
        ----------
        .. [1] Girard, G., Whittingstall, K., Deriche, R., & Descoteaux, M.
               Towards quantitative connectivity analysis: reducing
               tractography biases. NeuroImage, 98, 266-278, 2014.
        """

        if not isinstance(tissue_classifier, ConstrainedTissueClassifier):
            raise ValueError("expecting ConstrainedTissueClassifier")

        self.pft_max_nbr_back_steps = int(np.ceil(pft_back_tracking_dist
                                                  / step_size))
        self.pft_max_nbr_front_steps = int(np.ceil(pft_front_tracking_dist
                                                   / step_size))
        pft_max_steps = (self.pft_max_nbr_back_steps +
                         self.pft_max_nbr_front_steps)

        if (self.pft_max_nbr_front_steps < 0
                or self.pft_max_nbr_back_steps < 0
                or pft_max_steps < 1):
            raise ValueError("The number of PFT steps must be greater than 0.")

        if particle_count <= 0:
            raise ValueError("The particle count must be greater than 0.")

        if min_wm_pve_before_stopping < 0 or min_wm_pve_before_stopping > 1:
            raise ValueError("The min_wm_pve_before_stopping value must be "
                             "between 0 and 1.")

        self.min_wm_pve_before_stopping = min_wm_pve_before_stopping

        self.directions = np.empty((maxlen + 1, 3), dtype=float)

        self.pft_max_trial = pft_max_trial
        self.particle_count = particle_count
        self.particle_paths = np.empty((2, self.particle_count,
                                        pft_max_steps + 1, 3),
                                       dtype=float)
        self.particle_weights = np.empty(self.particle_count, dtype=float)
        self.particle_dirs = np.empty((2, self.particle_count,
                                       pft_max_steps + 1, 3), dtype=float)
        self.particle_steps = np.empty((2, self.particle_count), dtype=int)
        self.particle_tissue_classes = np.empty((2, self.particle_count),
                                                dtype=int)
        super(ParticleFilteringTracking, self).__init__(
                direction_getter=direction_getter,
                tissue_classifier=tissue_classifier,
                seeds=seeds,
                affine=affine,
                step_size=step_size,
                max_cross=max_cross,
                maxlen=maxlen,
                fixedstep=True,
                return_all=return_all,
                random_seed=random_seed,
                unidirectional=unidirectional,
                initial_directions=initial_directions)

    def _tracker(self, seed, first_step, streamline):
        return pft_tracker(self.direction_getter,
                           self.tissue_classifier,
                           seed,
                           first_step,
                           self._voxel_size,
                           streamline,
                           self.directions,
                           self.step_size,
                           self.pft_max_nbr_back_steps,
                           self.pft_max_nbr_front_steps,
                           self.pft_max_trial,
                           self.particle_count,
                           self.particle_paths,
                           self.particle_dirs,
                           self.particle_weights,
                           self.particle_steps,
                           self.particle_tissue_classes,
                           self.min_wm_pve_before_stopping)
