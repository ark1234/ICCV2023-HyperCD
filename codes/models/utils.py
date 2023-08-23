import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
# import numpy as np
# from glob import glob
# # sys.path.insert(0,'/PointFlow/')
import torch
from torch import nn, einsum
from pointnet2_ops.pointnet2_utils import furthest_point_sample, \
    gather_operation, ball_query, three_nn, three_interpolate, grouping_operation
from pointnet2_ops import pointnet2_utils

class Conv1d(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size=1,
                 stride=1,
                 if_bn=True,
                 activation_fn=torch.relu):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel,
                              out_channel,
                              kernel_size,
                              stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm1d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out


class Conv2d(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size=(1, 1),
                 stride=(1, 1),
                 if_bn=True,
                 activation_fn=torch.relu):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel,
                              out_channel,
                              kernel_size,
                              stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out


class MLP(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Linear(last_channel, out_channel))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Linear(last_channel, layer_dims[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)


class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)


class MLP_Res(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out




def dilated_ball_query(dist, h, base_radius, max_radius):
    '''
    Density-dilated ball query
    Inputs:
        dist[B, M, N]: distance matrix 
        h(float): bandwidth
        base_radius(float): minimum search radius
        max_radius(float): maximum search radius
    Returns:
        radius[B, M, 1]: search radius of point
    '''

    # kernel density estimation (Eq. 8)
    sigma = 1
    gauss = torch.exp(-(dist)/(2*(h**2)*(sigma**2))) # K(x-x_i/h), [B, M, N]
    kd_dist = torch.sum(gauss, dim=-1).unsqueeze(-1) # kernel distance, [B, M, 1]

    # normalization
    kd_score = kd_dist / (torch.max(kd_dist, dim=1)[0].unsqueeze(-1) + 1e-9) # [B, M, 1]
    radius = base_radius + (max_radius - base_radius)*kd_score # kd_score -> max, base_radius -> max_radius

    return radius 

def get_dist(src, dst):
    """
    Calculate the Euclidean distance between each point pair in two point clouds.
    Inputs:
        src[B, M, 3]: point cloud 1
        dst[B, N, 3]: point cloud 2
    Return: 
        dist[B, M, N]: distance matrix
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def sample_and_group(xyz, points, npoint, nsample, radius, use_xyz=True):
    """
    Args:
        xyz: Tensor, (B, 3, N)
        points: Tensor, (B, f, N)
        npoint: int
        nsample: int
        radius: float
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, npoint)
        new_points: Tensor, (B, 3 | f+3 | f, npoint, nsample)
        idx_local: Tensor, (B, npoint, nsample)
        grouped_xyz: Tensor, (B, 3, npoint, nsample)

    """
    xyz_flipped = xyz.permute(0, 2, 1).contiguous()  # (B, N, 3)
    new_xyz = gather_operation(xyz,
                               furthest_point_sample(xyz_flipped,
                                                     npoint))  # (B, 3, npoint)
    
    dist = get_dist(xyz, xyz_flipped.permute(0, 2, 1)) # disntance matrix, [B, M, N]
    radius = dilated_ball_query(dist, h=0.1, base_radius=radius, max_radius=radius*2)
    # print(radius.mean().item())

    # print(radius.item())
    # exit() 



    idx = ball_query(radius.mean().item(), nsample, xyz_flipped,
                     new_xyz.permute(0, 2,
                                     1).contiguous())  # (B, npoint, nsample)
    grouped_xyz = grouping_operation(xyz, idx)  # (B, 3, npoint, nsample)
    grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, nsample)

    if points is not None:
        grouped_points = grouping_operation(points,
                                            idx)  # (B, f, npoint, nsample)
        if use_xyz:
            new_points = torch.cat([grouped_xyz, grouped_points], 1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    """
    Args:
        xyz: Tensor, (B, 3, nsample)
        points: Tensor, (B, f, nsample)
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, 1)
        new_points: Tensor, (B, f|f+3|3, 1, nsample)
        idx: Tensor, (B, 1, nsample)
        grouped_xyz: Tensor, (B, 3, 1, nsample)
    """
    b, _, nsample = xyz.shape
    device = xyz.device
    new_xyz = torch.zeros((1, 3, 1), dtype=torch.float,
                          device=device).repeat(b, 1, 1)
    grouped_xyz = xyz.reshape((b, 3, 1, nsample))
    idx = torch.arange(nsample,
                       device=device).reshape(1, 1, nsample).repeat(b, 1, 1)
    if points is not None:
        if use_xyz:
            new_points = torch.cat([xyz, points], 1)
        else:
            new_points = points
        new_points = new_points.unsqueeze(2)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


class PointNet_SA_Module(nn.Module):
    def __init__(self,
                 npoint,
                 nsample,
                 in_channel,
                 mlp,
                 if_bn=True,
                 group_all=False,
                 use_xyz=True,
                 if_idx=False,
                 radius=0.005):
        """
        Args:
            npoint: int, number of points to sample
            nsample: int, number of points in each local region
            radius: float
            in_channel: int, input channel of features(points)
            mlp: list of int,
        """
        super(PointNet_SA_Module, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.radius = radius
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.if_idx = if_idx
        if use_xyz:
            in_channel += 3

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp:
            self.mlp_conv.append(Conv2d(last_channel, out_channel,
                                        if_bn=if_bn))
            last_channel = out_channel

        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz, points):
        """
        Args:
            xyz: Tensor, (B, 3, N)
            points: Tensor, (B, f, N)

        Returns:
            new_xyz: Tensor, (B, 3, npoint)
            new_points: Tensor, (B, mlp[-1], npoint)
        """
        if self.group_all:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(
                xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(
                xyz, points, self.npoint, self.nsample, self.radius,
                self.use_xyz)

        new_points = self.mlp_conv(new_points)
        new_points = torch.max(new_points, 3)[0]

        if self.if_idx:
            return new_xyz, new_points, idx
        else:
            return new_xyz, new_points


class PointNet_FP_Module(nn.Module):
    def __init__(self,
                 in_channel,
                 mlp,
                 use_points1=False,
                 in_channel_points1=None,
                 if_bn=True):
        """
        Args:
            in_channel: int, input channel of points2
            mlp: list of int
            use_points1: boolean, if use points
            in_channel_points1: int, input channel of points1
        """
        super(PointNet_FP_Module, self).__init__()
        self.use_points1 = use_points1

        if use_points1:
            in_channel += in_channel_points1

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp:
            self.mlp_conv.append(Conv1d(last_channel, out_channel,
                                        if_bn=if_bn))
            last_channel = out_channel

        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Args:
            xyz1: Tensor, (B, 3, N)
            xyz2: Tensor, (B, 3, M)
            points1: Tensor, (B, in_channel, N)
            points2: Tensor, (B, in_channel, M)

        Returns:MLP_CONV
            new_points: Tensor, (B, mlp[-1], N)
        """
        dist, idx = three_nn(
            xyz1.permute(0, 2, 1).contiguous(),
            xyz2.permute(0, 2, 1).contiguous())
        dist = torch.clamp_min(dist, 1e-10)  # (B, N, 3)
        recip_dist = 1.0 / dist
        norm = torch.sum(recip_dist, 2, keepdim=True).repeat((1, 1, 3))
        weight = recip_dist / norm
        interpolated_points = three_interpolate(points2, idx,
                                                weight)  # B, in_channel, N

        if self.use_points1:
            new_points = torch.cat([interpolated_points, points1], 1)
        else:
            new_points = interpolated_points

        new_points = self.mlp_conv(new_points)
        return new_points


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # B, N, M
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def query_knn(nsample, xyz, new_xyz, include_self=True):
    """Find k-NN of new_xyz in xyz"""
    pad = 0 if include_self else 1
    sqrdists = square_distance(new_xyz, xyz)  # B, S, N
    idx = torch.argsort(sqrdists, dim=-1, descending=False)[:, :,
                                                            pad:nsample + pad]
    return idx.int()


def sample_and_group_knn(xyz, points, npoint, k, use_xyz=True, idx=None):
    """
    Args:
        xyz: Tensor, (B, 3, N)
        points: Tensor, (B, f, N)
        npoint: int
        nsample: int
        radius: float
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, npoint)
        new_points: Tensor, (B, 3 | f+3 | f, npoint, nsample)
        idx_local: Tensor, (B, npoint, nsample)
        grouped_xyz: Tensor, (B, 3, npoint, nsample)

    """
    xyz_flipped = xyz.permute(0, 2, 1).contiguous()  # (B, N, 3)
    new_xyz = gather_operation(xyz,
                               furthest_point_sample(xyz_flipped,
                                                     npoint))  # (B, 3, npoint)
    if idx is None:
        idx = query_knn(k, xyz_flipped, new_xyz.permute(0, 2, 1).contiguous())
    grouped_xyz = grouping_operation(xyz, idx)  # (B, 3, npoint, nsample)
    grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, k)

    if points is not None:
        grouped_points = grouping_operation(points,
                                            idx)  # (B, f, npoint, nsample)
        if use_xyz:
            new_points = torch.cat([grouped_xyz, grouped_points], 1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


class PointNet_SA_Module_KNN(nn.Module):
    def __init__(self,
                 npoint,
                 nsample,
                 in_channel,
                 mlp,
                 if_bn=True,
                 group_all=False,
                 use_xyz=True,
                 if_idx=False):
        """
        Args:
            npoint: int, number of points to sample
            nsample: int, number of points in each local region
            radius: float
            in_channel: int, input channel of features(points)
            mlp: list of int,
        """
        super(PointNet_SA_Module_KNN, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.if_idx = if_idx
        if use_xyz:
            in_channel += 3

        last_channel = in_channel
        self.mlp_conv = []
        # print(mlp[:-1])
        # exit()
        for out_channel in mlp[:-1]:
            # print(out_channel)
            self.mlp_conv.append(Conv2d(last_channel, out_channel,
                                        if_bn=if_bn))
            last_channel = out_channel
        # exit()
        self.mlp_conv.append(
            Conv2d(last_channel, mlp[-1], if_bn=False, activation_fn=None))
        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz, points, idx=None):
        """
        Args:
            xyz: Tensor, (B, 3, N)
            points: Tensor, (B, f, N)

        Returns:
            new_xyz: Tensor, (B, 3, npoint)
            new_points: Tensor, (B, mlp[-1], npoint)
        """
        if self.group_all:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(
                xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_knn(
                xyz, points, self.npoint, self.nsample, self.use_xyz, idx=idx)

        # print(new_points.shape)
        new_points = self.mlp_conv(new_points)
        # print(new_points.shape)

        new_points = torch.max(new_points, 3)[0]
        # print(new_points.shape)
        # exit()

        if self.if_idx:
            return new_xyz, new_points, idx
        else:
            return new_xyz, new_points


def fps_subsample(pcd, n_points=2048):
    """
    Args
        pcd: (b, 16384, 3)

    returns
        new_pcd: (b, n_points, 3)
    """
    if pcd.shape[1] == n_points:
        return pcd
    elif pcd.shape[1] < n_points:
        raise ValueError(
            'FPS subsampling receives a larger n_points: {:d} > {:d}'.format(
                n_points, pcd.shape[1]))
    new_pcd = gather_operation(
        pcd.permute(0, 2, 1).contiguous(),
        furthest_point_sample(pcd, n_points))
    new_pcd = new_pcd.permute(0, 2, 1).contiguous()
    return new_pcd


def get_nearest_index(target, source, k=1, return_dis=False):
    """
    Args:
        target: (bs, 3, v1)
        source: (bs, 3, v2)
    Return:
        nearest_index: (bs, v1, 1)
    """
    inner = torch.bmm(target.transpose(1, 2), source)  # (bs, v1, v2)
    s_norm_2 = torch.sum(source**2, dim=1)  # (bs, v2)
    t_norm_2 = torch.sum(target**2, dim=1)  # (bs, v1)
    d_norm_2 = s_norm_2.unsqueeze(1) + t_norm_2.unsqueeze(
        2) - 2 * inner  # (bs, v1, v2)
    nearest_dis, nearest_index = torch.topk(d_norm_2,
                                            k=k,
                                            dim=-1,
                                            largest=False)
    if not return_dis:
        return nearest_index
    else:
        return nearest_index, nearest_dis


def indexing_neighbor(x, index):
    """
    Args:
        x: (bs, dim, num_points0)
        index: (bs, num_points, k)
    Return:
        feature: (bs, dim, num_points, k)
    """
    batch_size, num_points, k = index.size()

    id_0 = torch.arange(batch_size).view(-1, 1, 1)

    x = x.transpose(2, 1).contiguous()  # (bs, num_points, num_dims)
    feature = x[id_0, index]  # (bs, num_points, k, num_dims)
    feature = feature.permute(0, 3, 1,
                              2).contiguous()  # (bs, num_dims, num_points, k)

    return feature


class vTransformer(nn.Module):
    def __init__(self,
                 in_channel,
                 dim=256,
                 n_knn=16,
                 pos_hidden_dim=64,
                 attn_hidden_multiplier=4):
        super(vTransformer, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)

        self.pos_mlp = nn.Sequential(nn.Conv2d(3, pos_hidden_dim, 1),
                                     nn.BatchNorm2d(pos_hidden_dim), nn.ReLU(),
                                     nn.Conv2d(pos_hidden_dim, dim, 1))

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier), nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1))

        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, x, pos):
        """feed forward of transformer
        Args:
            x: Tensor of features, (B, in_channel, n)
            pos: Tensor of positions, (B, 3, n)

        Returns:
            y: Tensor of features with attention, (B, in_channel, n)
        """

        identity = x

        x = self.linear_start(x)
        b, dim, n = x.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)
        key = self.conv_key(x)  # (B, dim, N)
        value = self.conv_value(x)
        query = self.conv_query(x)

        key = grouping_operation(key, idx_knn)  # (B, dim, N, k)
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos.reshape(
            (b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # (B, 3, N, k)
        pos_embedding = self.pos_mlp(pos_rel)  # (B, dim, N, k)

        attention = self.attn_mlp(qk_rel + pos_embedding)
        attention = torch.softmax(attention, -1)

        # knn value is correct
        value = grouping_operation(value,
                                   idx_knn) + pos_embedding  # (B, dim, N, k)

        agg = einsum('b c i j, b c i j -> b c i', attention,
                     value)  # (B, dim, N)
        y = self.linear_end(agg)  # (B, in_dim, N)

        return y + identity






#diffConvNet
class Conv1x1(nn.Module):
    '''
    1x1 1d convolution
    '''
    def __init__(self, in_channels, out_channels, act=nn.GELU(), bias_=False): # nn.LeakyReLU(negative_slope=0.2)
        super(Conv1x1, self).__init__()
        self.conv = nn.Sequential(      
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias_),
                nn.BatchNorm1d(out_channels)
            )
        self.act = act
        nn.init.xavier_normal_(self.conv[0].weight.data)
    
    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        x = self.conv(x)
        
        x = x.transpose(1, 2).contiguous()
        if self.act is not None:
            return self.act(x)
        else:
            return x
        
class PositionEncoder(nn.Module):
    def __init__(self, out_channel, radius, k=20):
        super(PositionEncoder, self).__init__()
        self.k = k

        self.xyz2feature = nn.Sequential(
                            nn.Conv2d(9, out_channel//8, kernel_size=1),
                            nn.BatchNorm2d(out_channel//8),
                            nn.GELU()
                            )
        
        self.mlp = nn.Sequential(
                            Conv1x1(out_channel//8, out_channel//4),
                            Conv1x1(out_channel//4, out_channel, act=None)   
                            )
        
        self.qg = pointnet2_utils.QueryAndGroup(radius, self.k)
    
    def forward(self, centroid, xyz, radius, dist):
        point_feature, _ = sample_and_group(radius, self.k, xyz, xyz, centroid, dist) # [B, N, k, 3]
        
        points = centroid.unsqueeze(2).repeat(1, 1, self.k, 1) # [B, N, k, 3]
        
        variance = point_feature - points # [B, N, k, 3]
        
        point_feature = torch.cat((points, point_feature, variance), dim=-1) # [B, N, k, 9]

        point_feature = point_feature.permute(0, 3, 1, 2).contiguous() # [B, 9, N, k]
        
        point_feature = self.xyz2feature(point_feature) # [B, 9, N, k]
        
        point_feature = torch.max(point_feature, dim=-1)[0].transpose(1,2) # [B, N, C]        
        
        point_feature = self.mlp(point_feature) # [B, N, C']
        
        return point_feature

class MaskedAttention(nn.Module):
    def __init__(self, in_channels, hid_channels=128):
        super().__init__()
        if not hid_channels:
            hid_channels = 1
        self.conv_q = Conv1x1(in_channels+3, hid_channels, act=None)# map query (key points) to another linear space  
        self.conv_k = Conv1x1(in_channels+3, hid_channels, act=None)# map key (neighbor points) to another linear space

    def forward(self, cent_feat, feat, mask):
        '''
        Inputs:
            cent_feat: [B, M, C+3]
            feat: [B, N, C+3]
            mask: [B, M, N]

        Returns:
            adj: [B, M, N]
        '''
        q = self.conv_q(cent_feat) # [B, M, C+3] -> [B, M, C_int]

        k = self.conv_k(feat) # [B, N, C+3] -> [B, N, C_int]
        
        adj = torch.bmm(q, k.transpose(1, 2)) # [B, M, C_int] * [B, C_int, N] -> [B, M, N]

        # masked self-attention: masking all non-neighbors (Eq. 9)
        adj = adj.masked_fill(mask < 1e-9, -1e9)
        adj = torch.softmax(adj, dim=-1)

        # balanced renormalization (Eq. 11)
        adj = torch.sqrt(mask + 1e-9) * torch.sqrt(adj + 1e-9) - 1e-9

        adj = F.normalize(adj, p=1, dim=1) # [B, M, N]
        adj = F.normalize(adj, p=1, dim=-1) # [B, M, N]
        
        return adj

def dilated_ball_query(dist, h, base_radius, max_radius):
    '''
    Density-dilated ball query
    Inputs:
        dist[B, M, N]: distance matrix 
        h(float): bandwidth
        base_radius(float): minimum search radius
        max_radius(float): maximum search radius
    Returns:
        radius[B, M, 1]: search radius of point
    '''

    # kernel density estimation (Eq. 8)
    sigma = 1
    gauss = torch.exp(-(dist)/(2*(h**2)*(sigma**2))) # K(x-x_i/h), [B, M, N]
    kd_dist = torch.sum(gauss, dim=-1).unsqueeze(-1) # kernel distance, [B, M, 1]

    # normalization
    kd_score = kd_dist / (torch.max(kd_dist, dim=1)[0].unsqueeze(-1) + 1e-9) # [B, M, 1]
    radius = base_radius + (max_radius - base_radius)*kd_score # kd_score -> max, base_radius -> max_radius

    return radius 

class diffConv(nn.Module):
    def __init__(self, in_channels, out_channels, base_radius, bottleneck=4):
        super().__init__()
        self.conv_v = Conv1x1(2*in_channels, out_channels, act=None)
        self.mat = MaskedAttention(in_channels, in_channels//bottleneck)
        self.pos_conv = PositionEncoder(out_channels, np.sqrt(base_radius)) 
        self.base_radius = base_radius # squared radius

    def forward(self, x, xyz, cent_num):
        '''
        Inputs:
            x[B, N, C]: point features
            xyz[B, N, 3]: points
            cent_num(int): number of key points

        Returns:
            x[B, M, C']: updated point features 
            centroid[B, M, 3]: sampled features
        '''
        batch_size, point_num = xyz.size(0), xyz.size(1)

        if cent_num < point_num:
            # random sampling
            idx = np.arange(point_num)
            idx = idx[:cent_num] 
            idx = torch.from_numpy(idx).unsqueeze(0).repeat(batch_size, 1).int().to(xyz.device)

            # gathering
            centroid = index_points(xyz, idx) # [B, M, 3]
            cent_feat = index_points(x, idx) # [B, M, C]
        else:
            centroid = xyz.clone()
            cent_feat = x.clone()

        dist = get_dist(centroid, xyz) # disntance matrix, [B, M, N]

        radius = dilated_ball_query(dist, h=0.1, base_radius=self.base_radius, max_radius=self.base_radius*2)
            
        mask = (dist < radius).float()         

        # get attentive mask (adjacency matrix)
        emb_cent = torch.cat((cent_feat, centroid), dim=-1)
        emb_x = torch.cat((x, xyz), dim=-1)
        adj = self.mat(emb_cent, emb_x, mask) # [B, M, N]
        
        # inner-group attention
        smoothed_x = torch.bmm(adj, x) # [B, M, N] * [B, N, C] -> [B, M, C]
        variation = smoothed_x - cent_feat # [B, M, C] -> [B, M, C]
        
        x = torch.cat((variation, cent_feat), dim=-1) # [B, M, C] -> [B, M, 2C]
        x = self.conv_v(x) # [B, M, 2C] -> [B, M, C']
        
        pos_emb = self.pos_conv(centroid, xyz, radius, dist)

        # feature fusion
        x = x + pos_emb
        x = F.gelu(x)
        
        return x, centroid

class Attention_block(nn.Module):
    '''
    attention U-Net is taken from https://github.com/tiangexiang/CurveNet/blob/main/core/models/curvenet_util.py.
    '''
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm1d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g = g.transpose(1,2)
        x = x.transpose(1,2)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = F.gelu(g1+x1)
        psi = self.psi(psi)
        psi = psi.transpose(1,2)

        return psi

class PointFeaturePropagation(nn.Module):
    def __init__(self, in_channel1, in_channel2, out_channel):
        super(PointFeaturePropagation, self).__init__()
        in_channel = in_channel1 + in_channel2
        self.conv = nn.Sequential(  
                                  Conv1x1(in_channel, in_channel//2), 
                                  Conv1x1(in_channel//2, in_channel//2),
                                  Conv1x1(in_channel//2, out_channel)
                                    )
        self.att = Attention_block(in_channel1, in_channel2, in_channel2)

    def forward(self, xyz1, xyz2, feat1, feat2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, M, 3]
            feat1: input points data, [B, N, C']
            feat2: input points data, [B, M, C]
        Return:
            new_points: upsampled points data, [B, N, C+C']
        """
        dists, idx = pointnet2_utils.three_nn(xyz1, xyz2)
        dist_recip = 1.0 / (dists + 1e-6) # [B, N, 3]
        norm = torch.sum(dist_recip, dim=-1, keepdim=True) # [B, N, 1]
        weight = dist_recip / norm # [B, N, 1]
        int_feat = pointnet2_utils.three_interpolate(feat2.transpose(1,2).contiguous(), idx, weight).transpose(1,2)
        
        psix = self.att(int_feat, feat1)
        feat1 = feat1 * psix
        
        if feat1 is not None:
            cat_feat = torch.cat((feat1, int_feat), dim=-1) # [B, N, C'], [B, N, C] -> [B, N, C + C']
        else:
            cat_feat = int_feat # [B, N, C]
        cat_feat = self.conv(cat_feat) # [B, N, C + C'] -> [B, N, C']
        
        return cat_feat


def get_dist(src, dst):
    """
    Calculate the Euclidean distance between each point pair in two point clouds.
    Inputs:
        src[B, M, 3]: point cloud 1
        dst[B, N, 3]: point cloud 2
    Return: 
        dist[B, M, N]: distance matrix
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points[B, N, C]: input point features
        idx[B, M]: sample index data
    Return:
        new_points[B, M, C]: quried point features
    """
    new_points = pointnet2_utils.gather_operation(points.transpose(1,2).contiguous(), idx).transpose(1,2).contiguous()
    return new_points

def sample_and_group(radius, k, xyz, feat, centroid, dist):
    """
    Input:
        radius[B, M, 1]: search radius of each key point
        k(int): max number of samples in local region
        xyz[B, N, 3]: query points
        centroid[B, M, 3]: key points
        dist[B, M, N]: distance matrix
        feat[B, N, D]: input points features
    Return:
        cent_feat[B, M, D]: grouped features
        idx[B, M, k]: indices of selected neighbors
    """
    device = xyz.device
    B, N, _ = xyz.shape
    _, M, _ = centroid.shape
    
    idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, M, 1])
    
    idx[dist > radius] = N
    idx = idx.sort(dim=-1)[0][:, :, :k]
    group_first = idx[:, :, 0].view(B, M, 1).repeat([1, 1, k])
    mask = (idx == N)
    idx[mask] = group_first[mask]
    
    torch.cuda.empty_cache()
    idx = idx.int().contiguous()

    feat = feat.transpose(1,2).contiguous()
    cent_feat = pointnet2_utils.grouping_operation(feat, idx)
    cent_feat = cent_feat.transpose(1,2).transpose(-1, -2).contiguous()
    torch.cuda.empty_cache()
    
    return cent_feat, idx




