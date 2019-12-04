from __future__ import division, print_function, absolute_import

import numpy as np

from dipy.tracking.streamline import transform_streamlines

from dipy.utils.optpkg import optional_package
fury, have_fury, setup_module = optional_package('fury')

if have_fury:
    from fury.utils import lines_to_vtk_polydata, get_polydata_lines
    from fury.io import load_polydata, save_polydata


def save_vtk_streamlines(streamlines, filename,
                         to_lps=True, binary=False):
    """Save streamlines as vtk polydata to a supported format file.

    File formats can be VTK, FIB

    Parameters
    ----------
    streamlines : list
        list of 2D arrays or ArraySequence
    filename : string
        output filename (.vtk or .fib)
    to_lps : bool
        Default to True, will follow the vtk file convention for streamlines
        Will be supported by MITKDiffusion and MI-Brain
    binary : bool
        save the file as binary
    """
    if to_lps:
        # ras (mm) to lps (mm)
        to_lps = np.eye(4)
        to_lps[0, 0] = -1
        to_lps[1, 1] = -1
        streamlines = transform_streamlines(streamlines, to_lps)

    polydata = lines_to_vtk_polydata(streamlines, colors=False)
    save_polydata(polydata, filename, binary=binary)


def load_vtk_streamlines(filename, to_lps=True):
    """Load streamlines from vtk polydata.

    Load formats can be VTK, FIB

    Parameters
    ----------
    filename : string
        input filename (.vtk or .fib)
    to_lps : bool
        Default to True, will follow the vtk file convention for streamlines
        Will be supported by MITKDiffusion and MI-Brain

    Returns
    -------
    output :  list
         list of 2D arrays
    """
    polydata = load_polydata(filename)
    lines = get_polydata_lines(polydata)

    if to_lps:
        to_lps = np.eye(4)
        to_lps[0, 0] = -1
        to_lps[1, 1] = -1
        return transform_streamlines(lines, to_lps)

    return lines
