
import logging
import os

import nibabel
from nibabel.nifti1 import xform_codes, intent_codes, data_type_codes
import numpy as np


freesurfer_meta = {
    'AnatomicalStructurePrimary': 'CortexLeft',
    'AnatomicalStructureSecondary': 'GrayWhite',
    'GeometricType': 'Anatomical',
    'Name': 'surf/lh.white',
    'VolGeomWidth': '256',
    'VolGeomHeight': '256',
    'VolGeomDepth': '256',
    'VolGeomXsize': '1.000000',
    'VolGeomYsize': '1.000000',
    'VolGeomZsize': '1.000000',
    'VolGeomX_R': '-1.000000',
    'VolGeomX_A': '0.000000',
    'VolGeomX_S': '0.000000',
    'VolGeomY_R': '0.000000',
    'VolGeomY_A': '-0.000000',
    'VolGeomY_S': '-1.000000',
    'VolGeomZ_R': '-0.000000',
    'VolGeomZ_A': '1.000000',
    'VolGeomZ_S': '0.000000',
    'VolGeomC_R': '-46.233253',
    'VolGeomC_A': '-47.380287',
    'VolGeomC_S': '-13.806366',
    'SurfaceCenterX': '-31.191170',
    'SurfaceCenterY': '1.297012',
    'SurfaceCenterZ': '13.223328'
}


def load_gii_surface(filename):
    """ Load surface from Gifti file (.surf.gii)

    Parameters
    ----------
    filename : string
        Filename with valid extension

    Returns
    -------
    output : nibabel GiftiImage
        Return the gii surface
        None if it is not a Gifti
    """
    basename = os.path.basename(filename)
    splitted_basename = basename.split(".")
    last_extension = splitted_basename[-1].lower()
    if last_extension == "gii":
        if splitted_basename[-2].lower() != "surf":
            logging.warning('Only surface Gifti are supported (.surf.gii)\n'
                            '  the provided file only have the .gii extension')
        return nibabel.load(filename)
    else:
        logging.error('Provided file does not have the Gifti (.gii) extension')
        return None


def get_vertices_from_gii(gii):
    """ Load vertices from Gifti img

    Parameters
    ----------
    gii : nibabel GiftiImage
        Gifti surface object

    Returns
    -------
    output : numpy array
        Return vertices from the surface
        None if no vertices were found
    """
    return get_data_from_intent(gii, 'NIFTI_INTENT_POINTSET')


def get_triangles_from_gii(gii):
    """ Load triangles from Gifti img

    Parameters
    ----------
    gii : nibabel GiftiImage
        Gifti surface object

    Returns
    -------
    output : numpy array
        Return triangles from the surface
        None if no triangles were found
    """
    return get_data_from_intent(gii, 'NIFTI_INTENT_TRIANGLE')


def get_normals_from_gii(gii):
    """ Load normals from Gifti img

    Parameters
    ----------
    gii : nibabel GiftiImage
        Gifti surface object

    Returns
    -------
    output : numpy array
        Return normals from the surface
        None if no normals were found
    """
    return get_data_from_intent(gii, 'NIFTI_INTENT_VECTOR')


def get_transfo_from_gii(gii):
    """ Load transformation from Gifti img

    Parameters
    ----------
    gii : nibabel GiftiImage
        Gifti surface object

    Returns
    -------
    output : 4x4 array
        Return transformation from the gifti metadata
    """
    pointset = get_first_array_from_intent(gii, 'NIFTI_INTENT_POINTSET')
    return pointset.coordsys.xform


def get_shape_metadata_from_gii(gii):
    """ Load volume from Gifti img shape

    Parameters
    ----------
    gii : nibabel GiftiImage
        Gifti surface object

    Returns
    -------
    output : numpy array (3,)
        Return dimension of the related image shape
    """
    metadata = get_metadata_from_gii(gii)
    if not metadata:
        return None

    key_x = 'VolGeomWidth'
    key_y = 'VolGeomHeight'
    key_z = 'VolGeomDepth'
    shape_keys = [key_x, key_y, key_z]
    if not np.all(np.in1d(shape_keys, list(metadata.keys()))):
        logging.warning('No Gifti metadata for dimension was found')
        return None
    return np.array([metadata[x] for x in shape_keys], dtype=np.int)


def get_xyz_ras_metadata_from_gii(gii):
    """ Load X_RAS, Y_RAS, Z_RAS metadata info from Gifti

    Parameters
    ----------
    gii : nibabel GiftiImage
        Gifti surface object

    Returns
    -------
    output : numpy array (3, 3,)
        Return X_RAS, Y_RAS, Z_RAS in a single matrix
    """
    metadata = get_metadata_from_gii(gii)
    if not metadata:
        return None

    axis_string = ("X", "Y", "Z")
    order_string = ("R", "A", "S")
    basestring = "VolGeom{}_{}"
    values = []
    try:
        for i in axis_string:
            for j in order_string:
                values.append(float(metadata[basestring.format(i, j)]))
        return np.reshape(values, (3, 3))
    except KeyError:
        logging.warning('No Gifti metadata for xyz RAS was found')
    return None


def get_cras_metadata_from_gii_metadata(gii):
    """ Load C_RAS metadata info from Gifti

    Parameters
    ----------
    gii : nibabel GiftiImage
        Gifti surface object

    Returns
    -------
    output : numpy array (3,)
        Return C_RAS in a single array
    """
    metadata = get_metadata_from_gii(gii)
    if not metadata:
        return None

    basestring = "VolGeomC_{}"
    order_string = ("R", "A", "S")
    values = []
    try:
        for j in order_string:
            values.append(float(metadata[basestring.format(j)]))
        return np.asarray(values)
    except KeyError:
        logging.warning('No Gifti metadata for cRAS was found')
    return None


def get_zooms_metadata_from_gii(gii):
    """ Load voxel size metadata info from Gifti

    Parameters
    ----------
    gii : nibabel GiftiImage
        Gifti surface object

    Returns
    -------
    output : numpy array (3,)
        Return C_RAS in a single array
    """
    metadata = get_metadata_from_gii(gii)
    if not metadata:
        return None

    basestring = "VolGeom{}size"
    axis_string = ("X", "Y", "Z")
    values = []
    try:
        for i in axis_string:
            values.append(float(metadata[basestring.format(i)]))
        return np.asarray(values)
    except KeyError:
        logging.warning('No Gifti metadata for cRAS was found')
    return None


def get_metadata_from_gii(gii):
    pointset = get_first_array_from_intent(gii, 'NIFTI_INTENT_POINTSET')
    if pointset:
        return pointset.metadata
    return None


def is_gii_xform_valid(gii):
    """ Verify if xform from Gifti img, is valid for dipy header

    Parameters
    ----------
    gii : nibabel GiftiImage
        Gifti surface object

    Returns
    -------
    output : bool
        True xform metadata is compliant with dipy
        False otherwise
    """
    pointset = get_first_array_from_intent(gii, 'NIFTI_INTENT_POINTSET')
    current_space = xform_codes.niistring[pointset.coordsys.dataspace]
    xform_space = xform_codes.niistring[pointset.coordsys.xformspace]

    current_space_is_valid = is_xform_valid_for_gii(current_space)
    xform_space_is_valid = is_xform_valid_for_gii(xform_space)

    if current_space == xform_space:
        logging.warning('Both space is set %s, no transform', xform_space)
        return False
    elif current_space_is_valid and xform_space_is_valid:
        return True
    else:
        if not current_space_is_valid:
            logging.warning('Unsupported current space %s', current_space)
        if not xform_space_is_valid:
            logging.warning('Unsupported xform space %s', xform_space)
        return False


def get_data_from_intent(gii, intent):
    """ Wrapper over Nibabel function, to directly get data from a Gifti

    Parameters
    ----------
    gii : nibabel GiftiImage
        Gifti surface object
    intent : NIFTI_INTENT enum
        Nifti intent code (see nibabel.nifti1.intent_codes)

    Returns
    -------
    output : numpy array
        Data numpy array from the intent
        None if no related array was found
    """
    gii_array = get_first_array_from_intent(gii, intent)
    if gii_array is not None:
        return gii_array.data
    else:
        return None


def get_first_array_from_intent(gii, intent):
    """ Wrapper over Nibabel function, to access the Gifti array

    Parameters
    ----------
    gii : nibabel GiftiImage
        Gifti surface object
    intent : NIFTI_INTENT enum
        Nifti intent code (see nibabel.nifti1.intent_codes)

    Returns
    -------
    output : nibabel GiftiDataArray
        Gifti first array from the intent (nibabel GiftiDataArray)
        None if no related array was found
    """
    intent_nii = intent_codes.label[intent]
    arrays_list = gii.get_arrays_from_intent(intent_nii)
    intent_label = intent_codes.label[intent]
    if len(arrays_list) == 0:
        logging.error('No %s were found in the Gifti', intent_label)
        return None
    elif len(arrays_list) > 1:
        logging.warning('Multiple %s array were found in the Gifti \n'
                        'only the first one will be used', intent_label)
    return arrays_list[0]


def is_xform_valid_for_gii(xform_space):
    """ Verify if a specific Gifti xform_space is compliant with dipy

    Parameters
    ----------
    xform_space : NIFTI_XFORM enum
        Nifti xform code

    Returns
    -------
    output : bool
        True xform_space is compliant with dipy
        False otherwise
    """
    supported_space = ["NIFTI_XFORM_SCANNER_ANAT", "NIFTI_XFORM_ALIGNED_ANAT"]
    return xform_codes.niistring[xform_space] in supported_space
