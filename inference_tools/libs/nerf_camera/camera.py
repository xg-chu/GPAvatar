# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import torch
import nerfacc

class NeRFCamera(torch.nn.Module):
    # Please refer to https://pytorch3d.org/docs/cameras for coordinates system.
    def __init__(self, calib, camera_size, pts_per_ray, imp_pts_per_ray=None):
        super().__init__()
        if isinstance(calib, float):
            K = torch.tensor([[calib, 0.0, 0.0], [0.0, calib, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
        else:
            K = torch.tensor(calib, dtype=torch.float32)
        self.register_buffer("K", K, persistent=False)
        assert self.K.shape == (3, 3), self.K.shape
        self.camera_size = camera_size
        self.pts_per_ray = pts_per_ray
        self.imp_pts_per_ray = imp_pts_per_ray if imp_pts_per_ray else pts_per_ray
        self._init_camera_rays()

    def _init_camera_rays(self, ):
        x, y = torch.meshgrid(
            torch.linspace(1, -1, self.camera_size[0], dtype=torch.float32),
            torch.linspace(1, -1, self.camera_size[1], dtype=torch.float32),
            indexing="xy",
        )
        camera_directions = torch.nn.functional.pad(
            torch.stack([x / self.K[0][0], y / self.K[1][1]], dim=-1), (0, 1), value=1.0,
        )  # [num_rays, 3]
        self._camera_shape = camera_directions.shape[:2]
        _camera_dirs = camera_directions.reshape(-1, 3)
        self.register_buffer("_camera_dirs", _camera_dirs, persistent=False)

    def __str__(self, ):
        str_0 = 'Type: {}, Size: {}, K: {}.'.format(self._get_name(), self.camera_size, self.K.tolist())
        str_1 = 'Pts_per_ray: {}, Imp_pts_per_ray: {}.'.format(self.pts_per_ray, self.imp_pts_per_ray)
        return str_0 + '\n' + str_1

    def set_position(self, transform_matrix, ):
        # camera receive world2cam matrix.
        assert transform_matrix.dim() == 3 and transform_matrix.shape[1:] == (3, 4), 'Affine: (N, 3, 4), {}'.format(transform_matrix.shape)
        self.transform_matrix = transform_matrix
        self._batch_size = transform_matrix.shape[0]

    def _ray_marching(self, opacities, values, background=None):
        # opacities: [NM, pts_per_ray], values: [NM, pts_per_ray, 3]
        absorption = nerfacc.exclusive_prod(1 - opacities)
        weights = opacities * absorption
        accum_values = nerfacc.accumulate_along_rays(weights, values)
        accum_opacities = nerfacc.accumulate_along_rays(weights, None)
        accum_opacities.clamp_(0.0, 1.0)
        if background:
            accum_values = accum_values + (1 - accum_opacities) * background
        return accum_values, accum_opacities, weights

    def _ray_importance_sampling(self, ray_bundle, weights, noise=False):
        # origins: (NM, 3), dirs: (NM, 3), weights: (NM, pts_per_ray)
        weights = weights + 1e-5
        if weights.min() <= 0:
            raise ValueError("Negative weights provided.")
        pdfs = weights / weights.sum(dim=-1, keepdim=True)
        cdfs = torch.cumsum(pdfs, -1)
        intervals = nerfacc.RayIntervals(vals=ray_bundle['depths'])
        intervals, _ = nerfacc.importance_sampling(
            intervals, cdfs, self.imp_pts_per_ray, stratified=noise
        )
        bins = intervals.vals
        bins_start, bins_end = bins[..., :-1], bins[..., 1:]
        depths = (bins_start + bins_end) * 0.5
        coordinates = ray_bundle['origins'].unsqueeze(-2) + \
                      depths.unsqueeze(-1) * ray_bundle['dirs'].unsqueeze(-2)
        importance_ray_bundle = {
            'origins':ray_bundle['origins'], 'dirs': ray_bundle['dirs'], 'depths': depths, 
            'coords': coordinates
        }
        return importance_ray_bundle

    def _merge_hierarchical(self, depths_0, opacities_0, values_0, depths_1, opacities_1, values_1):
        # depths: (NM, pts_per_ray), opacities: (NM, pts_per_ray), values: (NM, pts_per_ray, 3)
        depths_all = torch.cat([depths_0, depths_1], dim = -1)
        values_all = torch.cat([values_0, values_1], dim = -2)
        opacities_all = torch.cat([opacities_0, opacities_1], dim = -1)
        depths_all, indices = torch.sort(depths_all, dim=-1)
        # depths_all = torch.gather(depths_all, -1, indices)
        opacities_all = torch.gather(opacities_all, -1, indices)
        values_all = torch.gather(values_all, -2, indices[..., None].expand(-1, -1, values_all.shape[-1]))
        return opacities_all, values_all, depths_all

    def two_path_render(self, nerf_query_fn, noise=False, background=None, **kwargs):
        def _batch_query(coords, dirs, nerf_query_fn, **kwargs):
            # coords: (NM, pts_per_ray, 3), dirs: (NM, 3)
            pts_per_ray = coords.shape[1]
            dirs = dirs[:, None].expand(-1, coords.shape[1], -1)
            coords = coords.reshape(self._batch_size, -1, 3)
            dirs = dirs.reshape(self._batch_size, -1, 3)
            opacities, values, other_kwargs = nerf_query_fn(coords, dirs, **kwargs)
            opacities = opacities.reshape(-1, pts_per_ray)
            values = values.reshape(-1, pts_per_ray, values.shape[-1])
            return opacities, values, other_kwargs
        # coarse sampling
        coarse_rays = self.ray_sampling(noise=noise)
        c_opacities, c_values, c_other_kwargs = _batch_query(
            coarse_rays['coords'], coarse_rays['dirs'], nerf_query_fn, **kwargs
        )
        # marching
        c_accum_values, c_accum_opacities, weights = self._ray_marching(c_opacities, c_values, background=background)
        # fine sampling
        fine_rays = self._ray_importance_sampling(coarse_rays, weights, noise=noise)
        f_opacities, f_values, f_other_kwargs = _batch_query(
            fine_rays['coords'], fine_rays['dirs'], nerf_query_fn, **kwargs
        )
        opacities, values, depths = self._merge_hierarchical(
            coarse_rays['depths'], c_opacities, c_values, 
            fine_rays['depths'], f_opacities, f_values
        )
        f_accum_values, f_accum_opacities, weights = self._ray_marching(opacities, values, background=background)
        # depths
        ray_depths = nerfacc.accumulate_along_rays(weights, depths[..., None])
        ray_depths = ray_depths + (1 - f_accum_opacities) * depths.max()
        ray_depths = (ray_depths - ray_depths.min()) / (ray_depths.max() - ray_depths.min())
        output_size = (self._batch_size, ) + self._camera_shape + (-1, )
        c_accum_values = c_accum_values.reshape(output_size).permute(0, 3, 1, 2).contiguous()
        f_accum_values = f_accum_values.reshape(output_size).permute(0, 3, 1, 2).contiguous()
        ray_depths = ray_depths.reshape(output_size).permute(0, 3, 1, 2)
        return c_accum_values, f_accum_values, {'coarse': c_other_kwargs, 'fine': f_other_kwargs, 'depth': ray_depths}


    @torch.no_grad()
    def debug(self, masks=None):
        def nerf_fn(coordinates, directions, **kwargs):
            opacities = torch.rand(coordinates.shape[0], coordinates.shape[1]).type_as(coordinates)
            values = torch.rand(coordinates.shape[0], coordinates.shape[1], 3).type_as(coordinates)
            return opacities, values, None
        self.set_position(
            transform_matrix = torch.tensor(
                [[[ 9.3255e-01, -1.5425e-01, -3.2644e-01,  8.6750e-08],
                [-3.6105e-01, -3.9842e-01, -8.4315e-01, -1.2890e-08],
                [-1.8923e-08,  9.0414e-01, -4.2724e-01,  4.0311e+00]], 
                [[ 8.8332e-01, -1.0858e-01, -4.5601e-01, -1.1263e-07],
                [-4.6876e-01, -2.0461e-01, -8.5930e-01,  9.2711e-09],
                [-3.2295e-10,  9.7280e-01, -2.3164e-01,  4.0311e+00]]
                ]
            ).cuda()
        )
        # ray_bundle = self.ray_sampling(n_pts_per_ray=32, noise=True)
        rgb, _, _ = self.render(nerf_fn, noise=True)
        return rgb

    @staticmethod
    def _jiggle_within_stratas(bin_centers: torch.Tensor) -> torch.Tensor:
        """
        Performs sampling of 1 point per bin given the bin centers.

        More specifically, it replaces each point's value `z`
        with a sample from a uniform random distribution on
        `[z - delta_-, z + delta_+]`, where `delta_-` is half of the difference
        between `z` and the previous point, and `delta_+` is half of the difference
        between the next point and `z`. For the first and last items, the
        corresponding boundary deltas are assumed zero.

        Args:
            `bin_centers`: The input points of size (..., N); the result is broadcast
                along all but the last dimension (the rows). Each row should be
                sorted in ascending order.

        Returns:
            a tensor of size (..., N) with the locations jiggled within stratas/bins.
        """
        # Get intervals between bin centers.
        mids = 0.5 * (bin_centers[..., 1:] + bin_centers[..., :-1])
        upper = torch.cat((mids, bin_centers[..., -1:]), dim=-1)
        lower = torch.cat((bin_centers[..., :1], mids), dim=-1)
        # Samples in those intervals.
        jiggled = lower + (upper - lower) * torch.rand_like(lower)
        return jiggled

    @staticmethod
    def _linspace(start: torch.Tensor, stop: torch.Tensor, num: int):
        steps = torch.arange(num, dtype=start.dtype, device=start.device) / (num - 1)
        steps = torch.broadcast_to(steps, start.shape[:-1]+(-1,))
        out = start + steps * (stop - start)
        return out


class OriginalNeRFCamera(NeRFCamera):
    def __init__(self, calib, camera_size, pts_per_ray, near, far, imp_pts_per_ray=None):
        super().__init__(calib, camera_size, pts_per_ray, imp_pts_per_ray)
        self.near, self.far = near, far
        self.render = self.two_path_render
   
    def ray_sampling(self, noise=False):
        # origons: [NM, 3], dirs: [NM, 3], depths: [NM, pts_per_ray], coords: [NM, pts_per_ray, 3]
        # transform
        R = self.transform_matrix[:, None, :3, :3]
        T = self.transform_matrix[:, None, :3, 3:]
        directions = torch.matmul(R, self._camera_dirs[None, :, :, None])[..., 0]
        origins = (-torch.matmul(R, T)[..., 0]).broadcast_to(directions.shape)
        directions = directions / torch.linalg.norm(directions, dim=-1, keepdims=True)
        origins = origins.reshape(-1, 3)
        directions = directions.reshape(-1, 3)
        # sample
        ray_start = origins.new_ones(origins.shape[0], 1) * self.near
        ray_end = origins.new_ones(origins.shape[0], 1) * self.far
        bins = self._linspace(ray_start, ray_end, self.pts_per_ray+1)
        if noise:
            bins = self._jiggle_within_stratas(bins)
        bins_start, bins_end = bins[..., :-1], bins[..., 1:]
        depths = (bins_start + bins_end) * 0.5
        # gather
        coordinates = origins.unsqueeze(-2) + depths.unsqueeze(-1) * directions.unsqueeze(-2)
        ray_bundle = {
            'origins':origins, 'dirs': directions, 'depths': depths, 
            'coords': coordinates,
        }
        return ray_bundle


class CubicNeRFCamera(NeRFCamera):
    def __init__(self, calib, camera_size, pts_per_ray, imp_pts_per_ray=None):
        super().__init__(calib, camera_size, pts_per_ray, imp_pts_per_ray)
        # {xmin, ymin, zmin, xmax, ymax, zmax}
        axis_align_box = torch.tensor([[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]])
        self.register_buffer("axis_align_box", axis_align_box, persistent=False)
        self.render = self.two_path_render
   
    def ray_sampling(self, noise=False):
        # origons: [NM, 3], dirs: [NM, 3], depths: [NM, pts_per_ray], coords: [NM, pts_per_ray, 3]
        # transform
        R = self.transform_matrix[:, None, :3, :3]
        T = self.transform_matrix[:, None, :3, 3:]
        directions = torch.matmul(R, self._camera_dirs[None, :, :, None])[..., 0]
        origins = (-torch.matmul(R, T)[..., 0]).broadcast_to(directions.shape)
        directions = directions / torch.linalg.norm(directions, dim=-1, keepdims=True)
        origins = origins.reshape(-1, 3)
        directions = directions.reshape(-1, 3)
        # sample
        ray_start, ray_end, hit = nerfacc.ray_aabb_intersect(
            origins.reshape(-1, 3), directions.reshape(-1, 3), self.axis_align_box, miss_value=-1.0
        )
        bins = self._linspace(ray_start, ray_end, self.pts_per_ray+1)
        if noise:
            bins = self._jiggle_within_stratas(bins)
        bins_start, bins_end = bins[..., :-1], bins[..., 1:]
        depths = (bins_start + bins_end) * 0.5
        # gather
        coordinates = origins.unsqueeze(-2) + depths.unsqueeze(-1) * directions.unsqueeze(-2)
        ray_bundle = {
            'origins':origins, 'dirs': directions, 'depths': depths, 
            'coords': coordinates
        }
        return ray_bundle



if __name__ == '__main__':
    cameras = CubicNeRFCamera(2.7778, [256, 256], 48, 24).cuda()
    print(cameras)
    coords = cameras.debug()
    print(coords.shape)
