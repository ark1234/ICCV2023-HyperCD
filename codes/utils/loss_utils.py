
'''
==============================================================

    0-------------------------------0
    |       Loss Functions          |
    0-------------------------------0

==============================================================

    Compute chamfer distance loss L1/L2

==============================================================
'''

import torch
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
from models.utils import fps_subsample

from geoopt.manifolds import PoincareBall as PoincareBallParent
from geoopt.manifolds.stereographic.math import _lambda_x, arsinh, tanh

MIN_NORM = 1e-15


chamfer_dist = chamfer_3DDist()


from hyptorch import nn as hypnn
from hyptorch.pmath import dist_matrix

def chamfer(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2)


def chamfer_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.clamp(d1, min=1e-9)
    d2 = torch.clamp(d2, min=1e-9)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    # import pdb; pdb.set_trace()
    return (d1 + d2) / 2


def chamfer_single_side(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(d1)
    return d1


def chamfer_single_side_sqrt(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.clamp(d1, min=1e-9)
    d2 = torch.clamp(d2, min=1e-9)
    d1 = torch.mean(torch.sqrt(d1))
    return d1


def get_loss(pcds_pred, partial, gt, sqrt=True):
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    Pc, P1, P2, P3 = pcds_pred

    gt_2 = fps_subsample(gt, P2.shape[1])
    gt_1 = fps_subsample(gt_2, P1.shape[1])
    gt_c = fps_subsample(gt_1, Pc.shape[1])

    cdc = CD(Pc, gt_c)
    cd1 = CD(P1, gt_1)
    cd2 = CD(P2, gt_2)
    cd3 = CD(P3, gt)

    partial_matching = PM(partial, P3)

    loss_all = (cdc + cd1 + cd2 + cd3 + partial_matching) * 1e3
    losses = [cdc, cd1, cd2, cd3, partial_matching]
    return loss_all, losses, [gt_2, gt_1, gt_c]


def get_loss1(pcds_pred, partial, gt, sqrt=True):
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    Pc, P1, P2, P3 = pcds_pred

    gt_2 = fps_subsample(gt, P2.shape[1])
    gt_1 = fps_subsample(gt_2, P1.shape[1])
    gt_c = fps_subsample(gt_1, Pc.shape[1])

    # cdc = CD(Pc, gt_c)
    cdc = calc_cd_like_hyperV2(Pc, gt_c)
    # cd1 = CD(P1, gt_1)
    cd1 = calc_cd_like_hyperV2(P1, gt_1)
    # cd2 = CD(P2, gt_2)
    cd2 = calc_cd_like_hyperV2(P2, gt_2)
    # cd3 = CD(P3, gt)
    cd3 = calc_cd_like_hyperV2(P3, gt)

    # partial_matching = PM(partial, P3)
    partial_matching = calc_cd_one_side_like_hyperV2(partial, P3)

    loss_all = (cdc + cd1 + cd2 + cd3 + partial_matching) * 1e3
    losses = [cdc, cd1, cd2, cd3, partial_matching]
    return loss_all, losses, [gt_2, gt_1, gt_c]




def calc_cd_like_hyperV2(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    # cham_loss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
    # dist1, dist2, idx1, idx2 = cham_loss(array1, array2)
    # dist1 = torch.clamp(dist1, min=1e-9)
    # dist2 = torch.clamp(dist2, min=1e-9)
    # d1 = torch.sqrt(dist1)
    # d2 = torch.sqrt(dist2)
    d1 = arcosh(1+ 1 * d1)
    d2 = arcosh(1+ 1 * d2)
    # print(d1.shape)
    # print(d2.shape)

    return torch.mean(d1) + torch.mean(d2)



def calc_cd_one_side_like_hyperV2(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    # cham_loss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
    # dist1, dist2, idx1, idx2 = cham_loss(array1, array2)
    # dist1 = torch.clamp(dist1, min=1e-9)
    # dist2 = torch.clamp(dist2, min=1e-9)
    # d1 = torch.sqrt(dist1)
    # d2 = torch.sqrt(dist2)
    d1 = arcosh(1+ 1 * d1)
    # d2 = arcosh(1+d2)
    # print(d1.shape)
    # print(d2.shape)

    return torch.mean(d1)



# distances = distances.clamp(-1 + eps, 1 - eps)
def arcosh(x, eps=1e-5):  # pragma: no cover
    # x = x.clamp(-1 + eps, 1 - eps)
    # x = x.clamp(1,)
    x = torch.clamp(x, min=1 + eps)
    return torch.log(x + torch.sqrt(1 + x) * torch.sqrt(x - 1))











