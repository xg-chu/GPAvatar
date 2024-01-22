import torch
import nvdiffrast.torch

class TPlanesEnc(torch.nn.Module):
    def __init__(self, feature_dim, plane_size, return_type='concat'):
        super().__init__()
        assert return_type in ['concat', 'mean']
        self.return_type = return_type

        self.tplanes = torch.nn.Parameter(
            torch.zeros(3, plane_size, plane_size, feature_dim), requires_grad=True
        )
        torch.nn.init.uniform_(self.tplanes, -1e-2, 1e-2)

    def __str__(self) -> str:
        string = 'TriPlanes Size: {}.'.format(self.tplanes.shape[1:])
        return string

    def forward(self, coords):
        # coords: N, M, 3; tplanes: N, 3, channel, H, W
        coords = coords * 0.5 + 0.5
        batch_size = coords.shape[0]
        coords = coords.reshape(-1, 3)
        decomposed_coords = torch.stack([
                coords[:, :, None, [0, 1]],
                coords[:, :, None, [0, 2]],
                coords[:, :, None, [2, 1]],
            ], dim=0,
        )  # 3xNx1x2
        if decomposed_coords.shape[1] > 4000000:
            output_features = []
            for i in range(0, decomposed_coords.shape[1], 4000000):
                output_features.append(
                    nvdiffrast.torch.texture(
                        self.tplanes, decomposed_coords[:, i:i+4000000].contiguous(),
                        mip_level_bias=None, boundary_mode="clamp", max_mip_level=0,
                    ).squeeze(dim=2)
                )
            output_features = torch.cat(output_features, dim=1)
        else:
            output_features = nvdiffrast.torch.texture(
                self.tplanes, decomposed_coords,
                mip_level_bias=None, boundary_mode="clamp", max_mip_level=0,
            ).squeeze(dim=2)  # 3xNMx1xC -> 3xNMxC -> NMx3xC -> NxMx3xC
        output_features = output_features.permute(1, 0, 2).view(batch_size, -1, 3, output_features.shape[-1])
        if self.return_type == 'concat':
            sampled_features = output_features.reshape(output_features.shape[0], output_features.shape[1], -1).contiguous()
        elif self.return_type == 'mean':
            sampled_features = output_features.mean(dim=-1).contiguous()
        else:
            raise NotImplementedError
        return sampled_features


def sample_from_planes(coords, tplanes, return_type='mean', channel_last=False):
    if not channel_last:
        assert tplanes.shape[1] == 3
        tplanes = tplanes.permute(0, 1, 3, 4, 2).contiguous()
    # coords: N, M, 3; tplanes: N, 3, H, W, channel
    coords = coords * 0.5 + 0.5
    batch_size = coords.shape[0]
    decomposed_coords = torch.stack([
            coords[:, :, None, [0, 1]],
            coords[:, :, None, [0, 2]],
            coords[:, :, None, [2, 1]],
        ], dim=1,
    ).reshape(-1, coords.shape[1], 1, 2)  # Nx3xMx1x2 -> N3xMx1x2
    tplanes = tplanes.reshape((-1,)+tplanes.shape[2:])
    if decomposed_coords.shape[1] > 4000000:
        output_features = []
        for i in range(0, decomposed_coords.shape[1], 4000000):
            output_features.append(
                nvdiffrast.torch.texture(
                    tplanes, decomposed_coords[:, i:i+4000000].contiguous(),
                    mip_level_bias=None, boundary_mode="clamp", max_mip_level=0,
                ).squeeze(dim=2) # N3xMx1xC -> N3xMxC -> Nx3xMxC
            )
        output_features = torch.cat(output_features, dim=1)
    else:
        output_features = nvdiffrast.torch.texture(
            tplanes, decomposed_coords,
            mip_level_bias=None, boundary_mode="clamp", max_mip_level=0,
        ).squeeze(dim=2)  # N3xMx1xC -> N3xMxC -> Nx3xMxC
    output_features = output_features.view(batch_size, 3, output_features.shape[1], output_features.shape[2])
    # Nx3xMxC -> NxMx3xC
    output_features = output_features.permute(0, 2, 3, 1)
    if return_type == 'concat':
        sampled_features = output_features.reshape(output_features.shape[0], output_features.shape[1], -1).contiguous()
    elif return_type == 'mean':
        sampled_features = output_features.mean(dim=-1).contiguous()
    else:
        raise NotImplementedError
    return sampled_features


if __name__ == '__main__':
    from tqdm.rich import tqdm
    coords = torch.rand(4, 10, 3).cuda()
    tplanes = torch.rand(4, 3, 256, 256, 32).cuda()
    for i in tqdm(range(6000)):
        sampled_features = sample_from_planes(coords, tplanes, return_type='mean')
    print(sampled_features.shape)

    coords = torch.rand(4, 10, 3).cuda()
    encoder = TPlanesEnc(feature_dim=32, plane_size=256).cuda()
    print(encoder)
    for i in tqdm(range(6000)):
        sampled_features = encoder(coords)
    print(sampled_features.shape)
