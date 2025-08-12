from __future__ import absolute_import, division

import numpy as np
from shapely.geometry import box, Polygon


def center_error(rects1, rects2):
    r"""Center error.
    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
    """
    centers1 = rects1[..., :2] + (rects1[..., 2:] - 1) / 2
    centers2 = rects2[..., :2] + (rects2[..., 2:] - 1) / 2
    errors = np.sqrt(np.sum(np.power(centers1 - centers2, 2), axis=-1))

    return errors

def normalized_center_error(rects1, rects2):
    r"""Center error normalized by the size of ground truth.
    Args:
        rects1 (numpy.ndarray): prediction box. An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): groudn truth box. An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
    """
    centers1 = rects1[..., :2] + (rects1[..., 2:] - 1) / 2
    centers2 = rects2[..., :2] + (rects2[..., 2:] - 1) / 2
    errors = np.sqrt(np.sum(np.power((centers1 - centers2)/np.maximum(np.array([[1.,1.]]), rects2[:, 2:]), 2), axis=-1))

    return errors


def rect_iou(rects1, rects2, bound=None):
    r"""Intersection over union.
    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        bound (numpy.ndarray): A 4 dimensional array, denotes the bound
            (min_left, min_top, max_width, max_height) for ``rects1`` and ``rects2``.
    """
    assert rects1.shape == rects2.shape
    if bound is not None:
        # bounded rects1
        rects1[:, 0] = np.clip(rects1[:, 0], 0, bound[0])
        rects1[:, 1] = np.clip(rects1[:, 1], 0, bound[1])
        rects1[:, 2] = np.clip(rects1[:, 2], 0, bound[0] - rects1[:, 0])
        rects1[:, 3] = np.clip(rects1[:, 3], 0, bound[1] - rects1[:, 1])
        # bounded rects2
        rects2[:, 0] = np.clip(rects2[:, 0], 0, bound[0])
        rects2[:, 1] = np.clip(rects2[:, 1], 0, bound[1])
        rects2[:, 2] = np.clip(rects2[:, 2], 0, bound[0] - rects2[:, 0])
        rects2[:, 3] = np.clip(rects2[:, 3], 0, bound[1] - rects2[:, 1])

    rects_inter = _intersection(rects1, rects2)
    areas_inter = np.prod(rects_inter[..., 2:], axis=-1)

    areas1 = np.prod(rects1[..., 2:], axis=-1)
    areas2 = np.prod(rects2[..., 2:], axis=-1)
    areas_union = areas1 + areas2 - areas_inter

    eps = np.finfo(float).eps
    ious = areas_inter / (areas_union + eps)
    ious = np.clip(ious, 0.0, 1.0)

    return ious


def _intersection(rects1, rects2):
    r"""Rectangle intersection.
    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
    """
    assert rects1.shape == rects2.shape
    x1 = np.maximum(rects1[..., 0], rects2[..., 0])
    y1 = np.maximum(rects1[..., 1], rects2[..., 1])
    x2 = np.minimum(rects1[..., 0] + rects1[..., 2],
                    rects2[..., 0] + rects2[..., 2])
    y2 = np.minimum(rects1[..., 1] + rects1[..., 3],
                    rects2[..., 1] + rects2[..., 3])

    w = np.maximum(x2 - x1, 0)
    h = np.maximum(y2 - y1, 0)

    return np.stack([x1, y1, w, h]).T


def poly_iou(polys1, polys2, bound=None):
    r"""Intersection over union of polygons.
    Args:
        polys1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height); or an N x 8 numpy array, each line represent
            the coordinates (x1, y1, x2, y2, x3, y3, x4, y4) of 4 corners.
        polys2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height); or an N x 8 numpy array, each line represent
            the coordinates (x1, y1, x2, y2, x3, y3, x4, y4) of 4 corners.
        bound (numpy.ndarray, optional): A 2 dimensional array, denotes the image bound
            (width, height) for ``rects1`` and ``rects2``.
    """
    assert polys1.ndim in [1, 2]
    if polys1.ndim == 1:
        polys1 = np.array([polys1])
        polys2 = np.array([polys2])
    assert len(polys1) == len(polys2)

    polys1 = _to_polygon(polys1)
    polys2 = _to_polygon(polys2)
    if bound is not None:
        bound = box(0, 0, bound[0], bound[1])
        polys1 = [p.intersection(bound) for p in polys1]
        polys2 = [p.intersection(bound) for p in polys2]
    
    eps = np.finfo(float).eps
    ious = []
    for poly1, poly2 in zip(polys1, polys2):
        area_inter = poly1.intersection(poly2).area
        area_union = poly1.union(poly2).area
        ious.append(area_inter / (area_union + eps))
    ious = np.clip(ious, 0.0, 1.0)

    return ious


def _to_polygon(polys):
    r"""Convert 4 or 8 dimensional array to Polygons
    Args:
        polys (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height); or an N x 8 numpy array, each line represent
            the coordinates (x1, y1, x2, y2, x3, y3, x4, y4) of 4 corners.
    """
    def to_polygon(x):
        assert len(x) in [4, 8]
        if len(x) == 4:
            return box(x[0], x[1], x[0] + x[2], x[1] + x[3])
        elif len(x) == 8:
            return Polygon([(x[2 * i], x[2 * i + 1]) for i in range(4)])
    
    if polys.ndim == 1:
        return to_polygon(polys)
    else:
        return [to_polygon(t) for t in polys]


def segm_iou(segm1, segm2):
    r"""Intersection over union for segmentation masks

    Args:
        segm1 (numpy.ndarray): An N x H x W numpy array, each line represent a binary mask
            (H, W).
        segm2 (numpy.ndarray): An N x H x W numpy array, each line represent a binary mask
            (H, W).
    """
    assert segm1.shape == segm2.shape

    segm1 = segm1.astype(bool)
    segm2 = segm2.astype(bool)

    ious = np.zeros(segm1.shape[0])

    for i in range(segm1.shape[0]):
        # Compute intersection and union
        intersection = np.logical_and(segm1[i], segm2[i]).sum()
        union = np.logical_or(segm1[i], segm2[i]).sum()
        
        eps = np.finfo(float).eps
        ious[i] = intersection / (union + eps)

    ious = np.clip(ious, 0.0, 1.0)

    return ious

def segm_iou_vec(segm1, segm2):
    r"""Intersection over union.

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        bound (numpy.ndarray): A 4 dimensional array, denotes the bound
            (min_left, min_top, max_width, max_height) for ``rects1`` and ``rects2``.
    """
    assert segm1.shape == segm2.shape

    segm1 = segm1.astype(bool)
    segm2 = segm2.astype(bool)

    ious = np.zeros(segm1.shape[0])

    # Compute intersection and union for each pair of masks
    intersection = np.logical_and(segm1, segm2).sum(axis=(1, 2))
    union = np.logical_or(segm1, segm2).sum(axis=(1, 2))

    eps = np.finfo(float).eps

    # Avoid division by zero: IoU is zero when union is zero
    ious = intersection / (union + eps)

    ious = np.clip(ious, 0.0, 1.0)
    
    return ious


def compute_barycenters(segm):
    """
    Computes the barycenter (centroid) of a binary mask.
    
    Parameters:
    - binary_mask: numpy.ndarray, binary mask of shape (H, W)
    
    Returns:
    - barycenter: tuple (cy, cx) representing the coordinates of the barycenter.
                  cy is the y-coordinate (row index) and cx is the x-coordinate (column index).
    """
    #if not segm.any():
    #    raise ValueError("The binary mask contains no foreground pixels.")

    # Ensure the mask is binary
    segm = segm.astype(bool)

    barycenters = []
    for i in range(segm.shape[0]):

        # Get the coordinates of all non-zero (foreground) pixels
        if np.any(segm[i]):
            y_coords, x_coords = np.nonzero(segm[i])

            # Compute the barycenter
            cy = y_coords.mean()
            cx = x_coords.mean()
        else:
            cy, cx = -1., -1.

        barycenters.append([cx, cy])

    return np.array(barycenters)

def normalized_center_error_segm(segm1, segm2):
    r"""Center error normalized by the size of ground truth for segmentations.

    Args:
        segm1 (numpy.ndarray): An N x H x W numpy array, each line represent a binary mask
            (H, W).
        segm2 (numpy.ndarray): An N x H x W numpy array, each line represent a binary mask
            (H, W).
    """
    assert segm1.shape == segm2.shape

    centers1 = compute_barycenters(segm1)
    centers2 = compute_barycenters(segm2)

    rects2 = []
    for i in range(segm2.shape[0]):
        if np.any(segm2[i]):
            rects2.append(_rect_from_mask(segm2[i]))
        else:
            rects2.append(np.zeros(4))
    rects2 = np.array(rects2)

    errors = np.sqrt(np.sum(np.power((centers1 - centers2)/np.maximum(np.array([[1.,1.]]), rects2[:, 2:]), 2), axis=-1))

    return errors

def _rect_from_mask(mask):
    '''
    create an axis-aligned rectangle from a given binary mask
    mask in created as a minimal rectangle containing all non-zero pixels
    '''
    x_ = np.sum(mask, axis=0)
    y_ = np.sum(mask, axis=1)
    x0 = np.min(np.nonzero(x_))
    x1 = np.max(np.nonzero(x_))
    y0 = np.min(np.nonzero(y_))
    y1 = np.max(np.nonzero(y_))
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]