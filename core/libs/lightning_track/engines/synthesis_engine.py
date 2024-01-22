import torch
import torchvision
from tqdm.rich import tqdm
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

from engines.FLAME import FLAMEDense, FLAMETex
from utils.renderer_utils import Texture_Renderer

class Synthesis_Engine:
    def __init__(self, device='cuda', lazy_init=True):
        self._device = device

    def init_model(self, calibration_results):
        # camera params
        self.verts_scale = calibration_results['verts_scale']
        self.focal_length = calibration_results['focal_length'].to(self._device)
        self.principal_point = calibration_results['principal_point'].to(self._device)
        # build flame
        self.flame_model = FLAMEDense(n_shape=100, n_exp=50).to(self._device)
        self.flame_texture = FLAMETex(n_tex=50).to(self._device)
        self.mesh_render = Texture_Renderer(
            tuv=self.flame_texture.get_tuv(), flame_mask=self.flame_texture.get_face_mask(), device=self._device
        )
        print('Done.')

    def _build_cameras_kwargs(self, batch_size, image_size=512):
        screen_size = torch.tensor(
            [image_size, image_size], device=self._device
        ).float()[None].repeat(batch_size, 1)
        cameras_kwargs = {
            'principal_point': self.principal_point.repeat(batch_size, 1), 'focal_length': self.focal_length, 
            'image_size': screen_size, 'device': self._device,
        }
        return cameras_kwargs

    def texture_optimize(self, batch_data, steps=[100, 70, 30]):
        # ['frames', 'bbox', 'mica_shape', 'emoca_expression', 'emoca_pose', 'transform_matrix']
        batch_size = len(batch_data['frames'])
        batch_data['frames'] = batch_data['frames'] / 255.0
        # build camera
        transform_matrix = batch_data['transform_matrix']
        rotation, translation = transform_matrix[:, :3, :3], transform_matrix[..., :3, 3]
        # flame params
        batch_data['emoca_pose'][..., :3] *= 0
        vertices, _, _ = self.flame_model(
            shape_params=batch_data['mica_shape'], 
            expression_params=batch_data['emoca_expression'],
            pose_params=batch_data['emoca_pose']
        )
        vertices = vertices * self.verts_scale
        # optimize
        sh_params = torch.nn.Parameter(torch.zeros(batch_size, 9, 3).float().to(self._device))
        texture_params = torch.nn.Parameter(torch.zeros(batch_size, 50).to(self._device))
        params = [
            {'params': [texture_params], 'lr': 0.02, 'name': ['tex']},
            {'params': [sh_params], 'lr': 0.05, 'name': ['sh']},
        ]
        optimizer = torch.optim.Adam(params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sum(steps), gamma=0.5)
        for sidx, this_steps in enumerate(steps):
            this_image_size = {0: 64, 1:128, 2:512}[sidx]
            cameras_kwargs = self._build_cameras_kwargs(batch_size, image_size=this_image_size)
            cameras = PerspectiveCameras(R=rotation, T=translation, **cameras_kwargs)
            tqdm_queue = tqdm(range(this_steps), desc='', leave=True, miniters=1, ncols=80, colour='#95bb72')
            gt_images = torchvision.transforms.functional.resize(batch_data['frames'], this_image_size, antialias=True) 
            gt_images = torchvision.transforms.functional.gaussian_blur(gt_images, [7, 7], sigma=[9.0, 9.0])
            for k in tqdm_queue:
                albedos = self.flame_texture(texture_params, image_size=this_image_size)
                lights = sh_params.expand(batch_size, -1, -1)
                pred_images, masks_all, masks_face = self.mesh_render(vertices, albedos, cameras=cameras, lights=lights, image_size=this_image_size)
                # torchvision.utils.save_image([pred_images[0], gt_images[0]], './debug.jpg', nrow=1)
                # loss_head = pixel_loss(pred_images, gt_images, mask=masks_all) * 350
                loss_face = pixel_loss(pred_images, gt_images, mask=masks_face) * 350
                loss_norm = torch.sum(texture_params ** 2) * 0.04
                all_loss = (loss_face + loss_norm)
                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()
                scheduler.step()
                tqdm_queue.set_description(f'Loss(Texture): {all_loss.item():.4f}')
        texture_results = {'tex_params': texture_params.detach().cpu(), 'sh_params': sh_params.detach().cpu(), }
        return texture_results, torchvision.utils.make_grid(torch.cat([gt_images[:4], pred_images[:4].clamp(0, 1)]).cpu(), nrow=4)
        

    def synthesis_optimize(self, track_frames, batch_data, texture_code, steps=[50, 30, 20], visualize=False):
        # ['frames', 'bbox', 'mica_shape', 'emoca_expression', 'emoca_pose', 'transform_matrix', 'lmks_dense']
        batch_size = len(track_frames)
        batch_data['emoca_pose'][..., :3] *= 0
        batch_data['frames'] = batch_data['frames'] / 255.0
        cameras_kwargs = self._build_cameras_kwargs(batch_size)
        # build params
        transform_matrix = batch_data['transform_matrix']
        rotation, translation = transform_matrix[:, :3, :3], transform_matrix[..., :3, 3]
        tex_params = torch.nn.Parameter(texture_code['tex_params'], requires_grad=False)
        sh_params = torch.nn.Parameter(texture_code['sh_params'])
        translation = torch.nn.Parameter(translation)
        ori_rotation = matrix_to_rotation_6d(rotation)
        rotation = torch.nn.Parameter(ori_rotation.clone())
        expression_codes = torch.nn.Parameter(batch_data['emoca_expression'])
        params = [
            {'params': [sh_params], 'lr': 0.001, 'name': ['sh']},
            {'params': [expression_codes], 'lr': 0.025, 'name': ['exp']},
            {'params': [rotation], 'lr': 0.02, 'name': ['r']},
            {'params': [translation], 'lr': 0.03, 'name': ['t']},
        ]
        optimizer = torch.optim.Adam(params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sum(steps), gamma=0.1)
        # run
        for sidx, this_steps in enumerate(steps):
            this_image_size = {0: 128, 1:256, 2:512}[sidx]
            cameras_kwargs = self._build_cameras_kwargs(batch_size, image_size=this_image_size)
            albedos = self.flame_texture(tex_params, image_size=this_image_size)
            gt_images = torchvision.transforms.functional.resize(batch_data['frames'], this_image_size, antialias=True) 
            gt_images = torchvision.transforms.functional.gaussian_blur(gt_images, [7, 7], sigma=[9.0, 9.0])
            gt_lmks_68 = batch_data['lmks'] / 512.0 * this_image_size
            gt_lmks_dense = batch_data['lmks_dense'][:, self.flame_model.mediapipe_idx] / 512.0 * this_image_size
            for idx in range(this_steps):
                # build flame params
                # flame params
                vertices, pred_lmk_68, pred_lmk_dense = self.flame_model(
                    shape_params=batch_data['mica_shape'], 
                    expression_params=expression_codes,
                    pose_params=batch_data['emoca_pose']
                )
                vertices = vertices*self.verts_scale
                pred_lmk_68, pred_lmk_dense = pred_lmk_68*self.verts_scale, pred_lmk_dense*self.verts_scale
                cameras = PerspectiveCameras(
                    R=rotation_6d_to_matrix(rotation), T=translation, **cameras_kwargs
                )
                pred_lmk_68 = cameras.transform_points_screen(
                    pred_lmk_68, R=rotation_6d_to_matrix(rotation), T=translation
                )[..., :2]
                pred_lmk_dense = cameras.transform_points_screen(
                    pred_lmk_dense, R=rotation_6d_to_matrix(rotation), T=translation
                )[..., :2]
                # vis_i = torchvision.utils.draw_keypoints(gt_images[0].to(torch.uint8), pred_lmk_dense[0:1, EYE_LMKS_0], colors="red", radius=1.5)
                # vis_i = torchvision.utils.draw_keypoints(vis_i.to(torch.uint8), pred_lmk_dense[0:1, EYE_LMKS_1], colors="blue", radius=1.5)
                # torchvision.utils.save_image(vis_i.float(), './debug.jpg')
                # import ipdb; ipdb.set_trace()
                pred_images, mask_all, mask_face = self.mesh_render(vertices, albedos, cameras=cameras, lights=sh_params, image_size=this_image_size)
                loss_face = pixel_loss(pred_images, gt_images, mask=mask_face) * 350 
                loss_head = pixel_loss(pred_images, gt_images, mask=mask_all) * 350 
                loss_lmk_68 = lmk_loss(pred_lmk_68, gt_lmks_68, this_image_size) * 1000
                loss_lmk_oval = oval_lmk_loss(pred_lmk_68, gt_lmks_68, this_image_size) * 2000
                loss_lmk_dense = lmk_loss(pred_lmk_dense, gt_lmks_dense, this_image_size) * 5000
                loss_lmk_mouth = mouth_lmk_loss(pred_lmk_dense, gt_lmks_dense, this_image_size) * 6000
                loss_lmk_eye_closure = eye_closure_lmk_loss(pred_lmk_dense, gt_lmks_dense, this_image_size) * 1000
                loss_exp_norm = torch.sum(expression_codes ** 2) * 0.02
                all_loss = (loss_face + loss_head) + loss_exp_norm + \
                           loss_lmk_68 + loss_lmk_oval + \
                           loss_lmk_dense + loss_lmk_mouth + loss_lmk_eye_closure 
                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()
                scheduler.step()
        if visualize:
            vis_images_0 = gt_images[:12].clone()
            vis_images_1 = gt_images[:12].clone()
            pred_images = pred_images[:12].detach()
            vis_masks = mask_face[:12].expand(-1, 3, -1, -1).detach()
            vis_images_1[vis_masks] = pred_images[vis_masks]
            vis_images = torch.cat(
                [vis_images_0[:, None], vis_images_1[:, None]], dim=1
            ).reshape(-1, 3, this_image_size, this_image_size)
            visualization = torchvision.utils.make_grid(vis_images, nrow=4)
                # torchvision.utils.save_image(visualization, 'debug.jpg')
        else:
            visualization = None
        # gather results
        synthesis_results = {}
        transform_matrix = torch.cat(
            [rotation_6d_to_matrix(rotation), translation[:, :, None]], dim=-1
        )
        for idx, name in enumerate(track_frames):
            synthesis_results[name] = {
                'tex_params': tex_params[idx].detach().float().cpu(),
                'sh_params': sh_params[idx].detach().float().cpu(),
                'bbox': batch_data['bbox'][idx].detach().float().cpu(),
                'mica_shape': batch_data['mica_shape'][idx].detach().float().cpu(),
                'emoca_expression': expression_codes[idx].detach().float().cpu(),
                'emoca_pose': batch_data['emoca_pose'][idx].detach().float().cpu(),
                'transform_matrix': transform_matrix[idx].detach().float().cpu()
            }
        return synthesis_results, visualization


def lmk_loss(opt_lmks, target_lmks, image_size, lmk_mask=None):
    size = torch.tensor([1 / image_size, 1 / image_size], device=opt_lmks.device).float()[None, None, ...]
    diff = torch.pow(opt_lmks - target_lmks, 2)
    if lmk_mask is None:
        return (diff * size).mean()
    else:
        return (diff * size * lmk_mask).mean()


def oval_lmk_loss(opt_lmks, target_lmks, image_size, lmk_mask=None):
    size = torch.tensor([1 / image_size, 1 / image_size], device=opt_lmks.device).float()[None, None, ...]
    oval_ids = [i for i in range(17)]
    diff = torch.pow(opt_lmks[:, oval_ids, :] - target_lmks[:, oval_ids, :], 2)
    if lmk_mask is None:
        return (diff * size).mean()
    else:
        return (diff * size * lmk_mask).mean()


def eye_closure_lmk_loss(opt_lmks, target_lmks, image_size, lmk_mask=None):
    size = torch.tensor([1 / image_size, 1 / image_size], device=opt_lmks.device).float()[None, None, ...]
    UPPER_EYE_LMKS = [29, 30, 31, 45, 46, 47,]
    LOWER_EYE_LMKS = [23, 24, 25, 39, 40, 41,]
    diff_opt = opt_lmks[:, UPPER_EYE_LMKS, :] - opt_lmks[:, LOWER_EYE_LMKS, :]
    diff_target = target_lmks[:, UPPER_EYE_LMKS, :] - target_lmks[:, LOWER_EYE_LMKS, :]
    diff = torch.pow(diff_opt - diff_target, 2)
    if lmk_mask is None:
        return (diff * size).mean()
    else:
        return (diff * size * lmk_mask).mean()


def mouth_lmk_loss(opt_lmks, target_lmks, image_size, lmk_mask=None):
    size = torch.tensor([1 / image_size, 1 / image_size], device=opt_lmks.device).float()[None, None, ...]
    MOUTH_IDS = [
        65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 
        75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 
        85, 86, 87, 88, 89, 91, 92, 93, 94, 95, 
        96, 97, 98, 99, 100, 101, 102, 103, 104
    ]
    diff = torch.pow(opt_lmks[:, MOUTH_IDS, :] - target_lmks[:, MOUTH_IDS, :], 2)
    if lmk_mask is None:
        return (diff * size).mean()
    else:
        return (diff * size * lmk_mask).mean()


def pixel_loss(opt_img, target_img, mask=None):
    if mask is None:
        mask = torch.ones_like(opt_img).type_as(opt_img)
    n_pixels = torch.sum((mask[:, 0, ...] > 0).int()).detach().float()
    loss = (mask * (opt_img - target_img)).abs()
    loss = torch.sum(loss) / n_pixels
    return loss
