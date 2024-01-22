import torch

class TPlanesEnc(torch.nn.Module):
    def __init__(self, feature_dim, plane_size, return_type='concat'):
        super().__init__()
        assert return_type in ['concat', 'mean']
        self.return_type = return_type

        self.tplanes = torch.nn.Parameter(
            torch.zeros(1, 3, feature_dim, plane_size, plane_size), requires_grad=True
        )
        torch.nn.init.uniform_(self.tplanes, -1e-2, 1e-2)

    def __str__(self) -> str:
        string = 'TriPlanes Size: {}.'.format(self.tplanes.shape[1:])
        return string

    def forward(self, coords):
        tplanes = self.tplanes.expand(coords.shape[0], -1, -1, -1, -1)
        features = sample_from_planes(coords, tplanes, return_type=self.return_type)
        return features


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


if __name__ == '__main__':
    from tqdm.rich import tqdm
    coords = torch.rand(4, 10, 3).cuda()
    tplanes = torch.rand(4, 3, 32, 256, 256).cuda()
    for i in tqdm(range(6000)):
        sampled_features = sample_from_planes(coords, tplanes, return_type='concat')
    print(sampled_features.shape)

    coords = torch.rand(4, 10, 3).cuda()
    encoder = TPlanesEnc(feature_dim=32, plane_size=256).cuda()
    print(encoder)
    for i in tqdm(range(6000)):
        sampled_features = encoder(coords, return_type='mean')
    print(sampled_features.shape)
