# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import torch
from pytorch3d.ops import knn_points
from pytorch3d.renderer.implicit.harmonic_embedding import HarmonicEmbedding

class PointsDecoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim, points_number, points_k=4):
        super().__init__()
        # encoder
        n_harmonic_dir = 4
        embedding_dim = n_harmonic_dir * 2 * 3 + 3
        self.pos_encoder = HarmonicEmbedding(n_harmonic_dir)
        # model
        self.points_k = points_k
        self.points_querier = DynamicPointsQuerier(in_dim, points_number)
        # nerf decoder
        self.feature_layers = torch.nn.Sequential(
            torch.nn.Linear(32+32, 128, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128, bias=True),
        )
        self.density_layers = torch.nn.Sequential(
            torch.nn.Linear(128, 1, bias=True),
            torch.nn.Softplus(beta=10.0)
        )
        self.rgb_layers = torch.nn.Sequential(
            torch.nn.Linear(128+embedding_dim, 64, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(64, out_dim, bias=True),
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, coordinates, directions, points_position, tex_tplanes):
        N, M, C = coordinates.shape
        selector = ((coordinates > -1.0) & (coordinates < 1.0)).all(dim=-1)
        # query feature
        sampled_tex_features = sample_from_planes(
            coordinates, tex_tplanes, return_type='mean'
        )
        # sampled_nerf_features = plane_features
        sampled_points_features, distances = self.points_querier(
            coordinates, points_position, K=self.points_k
        )
        sampled_nerf_features = torch.cat([sampled_tex_features, sampled_points_features], dim=-1)
        # nerf forward
        feat = self.feature_layers(sampled_nerf_features)
        raw_densities = self.density_layers(feat) * selector[..., None]
        densities = 1 - (-raw_densities).exp()
        # directions
        rays_embedding = torch.nn.functional.normalize(directions, dim=-1)
        rays_embedding = self.pos_encoder(rays_embedding)
        rgb = self.rgb_layers(torch.cat([feat, rays_embedding], dim=-1))
        rgb[..., :3] = self.sigmoid(rgb[..., :3])
        rgb[..., :3] = rgb[..., :3]*(1 + 2*0.001) - 0.001 
        return densities, rgb, {'densities': densities, 'distances': distances}


class DynamicPointsQuerier(torch.nn.Module):
    def __init__(self, in_dim, points_number, outputs_dim=32, points_dim=32):
        super().__init__()
        # self.bin_size = bin_size
        self.points_dim = in_dim
        self.outputs_dim = outputs_dim
        self.points_number = points_number
        n_harmonic_functions = 4
        embedding_dim = n_harmonic_functions * 2 * 3 + 3
        self.pos_encoder = HarmonicEmbedding(n_harmonic_functions)
        # params
        self.points_feature = torch.nn.Embedding(points_number, points_dim)
        self.point_layers = torch.nn.Sequential(
            torch.nn.Linear(in_dim+embedding_dim, outputs_dim*2, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(outputs_dim*2, outputs_dim, bias=True),
        )

    def forward(self, coordinates, points_position, K):
        # query feature
        dist, idx, nn = knn_points(
            coordinates.float(), points_position.float(), K=K, return_nn=True
        )
        points_features = self.points_feature(idx)
        points_relative_pos = coordinates[:, :, None] - nn
        points_relative_pos = torch.nn.functional.normalize(points_relative_pos, dim=-1)
        points_relative_pos_embed = self.pos_encoder(points_relative_pos)
        # mlp processing
        points_features = torch.cat([points_features, points_relative_pos_embed], dim=-1)
        points_features = self.point_layers(points_features)
        points_weights = 1 / dist
        points_weights = points_weights / points_weights.sum(dim=1, keepdim=True)
        sample_points_features = (points_features * points_weights.unsqueeze(-1)).sum(dim=2)
        # have_points = dist < self.bin_size
        # all_points_features = points_features.new_zeros(
        #     points_features.shape[0], points_features.shape[1], K, self.outputs_dim
        # )
        # merge feature
        # points_weights[~have_points] = 1e-12
        return sample_points_features, dist



def sample_from_planes(coords, tplanes, return_type='concat', mode='bilinear'):
    # coords: N, M, 3; tplanes: N, 3, channel, H, W
    def project_onto_planes(planes, coordinates):
        """
        Does a projection of a 3D point onto a batch of 2D planes,
        returning 2D plane coordinates.
        Takes plane axes of shape n_planes, 3, 3
        # Takes coordinates of shape N, M, 3
        # returns projections of shape N*n_planes, M, 2
        """
        N, M, C = coordinates.shape
        n_planes, _, _ = planes.shape
        coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
        inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
        projections = torch.bmm(coordinates, inv_planes)
        return projections[..., :2]

    tri_plane_axis = torch.tensor([
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
            [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
        ],dtype=torch.float32, device=tplanes.device
    )
    N, n_planes, C, H, W = tplanes.shape
    _, M, _ = coords.shape
    tplanes = tplanes.reshape(N*n_planes, C, H, W)
    projected_coordinates = project_onto_planes(tri_plane_axis, coords).unsqueeze(1)
    output_features = torch.nn.functional.grid_sample(
        tplanes, projected_coordinates.float(), mode=mode, padding_mode='zeros', align_corners=False
    ).permute(0, 3, 2, 1).view(N, n_planes, M, C)
    if return_type == 'concat':
        sampled_features = torch.cat(output_features.split(1, dim=1), dim=-1).squeeze(dim=1).contiguous()
    elif return_type == 'mean':
        sampled_features = output_features.mean(dim=1).contiguous()
    else:
        raise NotImplementedError
    return sampled_features

