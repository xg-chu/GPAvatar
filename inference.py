#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import sys
import argparse
from copy import deepcopy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
sys.path.append('./')

import torch
import torchvision
from pytorch3d.renderer.cameras import look_at_view_transform
from core.data.InferenceLightning import InferenceLightning
from core.data.tools import unnorm_transform, perspective_input
from core.utils.registry import MODEL_REGISTRY
from core.utils.visualize import Mesh_Renderer
from core.tools.config import pretty_dict
from core.utils.utils import tqdm, to_gpu, vis_depth, list_all_files

class InfEngine:
    def __init__(self, ckpt_path):
        ckpt_dict = torch.load(ckpt_path, map_location='cpu')
        self.device = 'cuda'
        self.model_path = ckpt_path
        model = MODEL_REGISTRY.get(ckpt_dict['config_dict']['MODEL']['MODEL_NAME'])(
            ckpt_dict['config_dict'], ckpt_dict['data_meta_info']
        ).eval()
        model.load_state_dict(ckpt_dict['state_dict'], strict=True)
        self.config_dict = model.config_dict
        print(pretty_dict(self.config_dict))
        self.model = model.to(self.device)
        self.output_path = os.path.join(
            'results', self.config_dict['TRAIN']['GENERAL']['EXP_STR'].split('_')[0],
        )
        self.meta_info = ckpt_dict['data_meta_info']
        print(pretty_dict(self.meta_info))
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        _water_mark_meta = torchvision.io.read_image(
            'demos/gpavatar_logo.png', mode=torchvision.io.ImageReadMode.RGB_ALPHA
        ).float()/255.0
        _scale = 320.0 / max(_water_mark_meta.shape)
        _tgt_size = (int(_water_mark_meta.shape[1] * _scale), int(_water_mark_meta.shape[2] * _scale))
        _water_mark_meta = self.resize(_water_mark_meta, tgt_size=_tgt_size)
        self._water_mark = _water_mark_meta[None, :3].to(self.device)
        self._water_mark_alpha = _water_mark_meta[None, 3:4].to(self.device).expand(-1, 3, -1, -1)

    def build_dataset(self, data_path, if_video=False):
        self.dataname = os.path.basename(data_path)
        if if_video:
            files = [f for f in os.listdir(data_path) if '.mp4' in f]
            if len(files):
                f_path = os.path.join(data_path, files[0])
                print('Read video from {}'.format(f_path))
                self._d_images, _, _ = torchvision.io.read_video(f_path, pts_unit='sec')
                self._d_images = self._d_images.permute(0, 3, 1, 2).float().to(self.device)/255.0
            self.dataset = InferenceLightning(data_path, self.config_dict['DATASET_AUGMENT'], self.meta_info, if_video=if_video)
        else:
            self.dataset = InferenceLightning(data_path, self.config_dict['DATASET_AUGMENT'], self.meta_info, if_video=if_video)
            self.dataset.slice(30)
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=1, pin_memory=True, drop_last=True, shuffle=False, 
        )
        self.dataloader = iter(dataloader)

    def build_camera(self, transforms, angle):
        distance = transforms[..., 3].square().sum(dim=-1).sqrt()[0].item() * 1.0
        # print(distance)
        R, T = look_at_view_transform(distance, 5, angle, device=self.device) # D, E, A
        # T[:, 1] = transforms[:, 1, 3]
        rotate_trans = torch.cat([R, T[:, :, None]], dim=-1)
        return rotate_trans

    def get_input_images(self, image_path):
        if os.path.exists(os.path.join(image_path, 'tracked_result.pth')):
            print('Run with tracked data: {}'.format(os.path.join(image_path, 'tracked_result.pth')))
            # frame_key = os.path.basename(image_path).split('.')[0]
            result = torch.load(os.path.join(image_path, 'tracked_result.pth'))
            input_images, feature_image, feature_shape, frame_key, frame_trans = \
                result['input_images'], result['feature_image'], result['feature_shape'], result['frame_key'], result['frame_trans']
            assert frame_key == os.path.basename(image_path).split('.')[0], 'Frame key not match!'
        else:
            print('Run with online tracking...')
            from core.libs.lightning_track import TrackEngine
            track_engine = TrackEngine(focal_length=self.meta_info['focal_length'], device=self.device)
            frame_key = os.path.basename(image_path).split('.')[0]
            image_paths = list_all_files(image_path)
            image_paths = [i for i in image_paths if ('jpg' in i or 'png' in i) and 'tracked' not in i]
            input_images = [self.read_image(p) for p in image_paths]
            input_images, lightning_results = track_engine.track_images(
                input_images, vis_path=os.path.join(image_path, 'tracked_visualization.jpg')
            )
            if lightning_results.keys() == []:
                raise Exception('Track failed!')
            input_images = torch.stack(input_images)
            frame_trans = torch.cat([
                torch.tensor(lightning_results[str(i)]['transform_matrix'][None]) for i in range(len(input_images))
            ])
            feature_image = torch.stack([
                perspective_input(
                    input_images[i][None].to(self.device), 
                    torch.tensor(lightning_results[str(i)]['transform_matrix'][None]).to(self.device), 
                    self.meta_info, fill=0.0
                )[0]
                for i in range(len(input_images))
            ])
            feature_shape = torch.stack([
                    torch.tensor(lightning_results[str(i)]['mica_shape']) for i in range(len(input_images))
                ]
            ).mean(dim=0)
            torch.save({
                'input_images': input_images.cpu(), 'frame_key': frame_key,
                'feature_image': feature_image.cpu(), 'feature_shape': feature_shape.cpu(), 'frame_trans': frame_trans
            }, os.path.join(image_path, 'tracked_result.pth'))
        input_images = input_images[None].to(self.device)
        feature_image, feature_shape = feature_image[None].to(self.device), feature_shape[None].to(self.device)
        return input_images, feature_image, feature_shape, frame_key, frame_trans

    def run_video(self, image_path, frame_num, rotation_angle=0, show_driver=True):
        results = []
        of_image, f_image, f_shape, frame_key, _ = self.get_input_images(image_path)
        frame_num = min(frame_num, len(self.dataloader))
        for f_idx in tqdm(range(frame_num)):
            batch_data = next(self.dataloader)
            if rotation_angle > 0:
                angle = -rotation_angle+rotation_angle*2/frame_num*f_idx
                batch_data['d_transforms'] = self.build_camera(deepcopy(batch_data['d_transforms']), angle)
            points = self.model.get_point_cloud(
                batch_data['d_poses'].to(self.device), batch_data['d_expressions'].to(self.device), 
                f_shape.to(self.device)
            )
            batch_data['f_images'] = f_image.clone()
            batch_data['d_points'] = points
            batch_data = to_gpu(batch_data, self.device)
            val_results = self.model.forward_inference(**batch_data)
            result_img = val_results['gen_sr'].clamp(0, 1)
            _mark_patch = result_img[..., -self._water_mark.shape[-2]:, -self._water_mark.shape[-1]:]
            _mark_patch[self._water_mark_alpha>0.5] = _mark_patch[self._water_mark_alpha>0.5] * 0.5 + \
                                                      self._water_mark[self._water_mark_alpha>0.5] * 0.5
            result_img[..., -self._water_mark.shape[-2]:, -self._water_mark.shape[-1]:] = _mark_patch
            if show_driver:
                if hasattr(self, '_d_images'):
                    d_images = self._d_images[f_idx:f_idx+1]
                else:
                    if not hasattr(self, 'mesh_renderer'):
                        self.mesh_renderer = Mesh_Renderer(
                            512, faces=self.model.flame_model.get_faces(), device=self.device
                        )
                    d_images, d_alphas = self.mesh_renderer(
                        points, transform_matrix=batch_data['d_transforms'],
                        focal_length=self.model.data_meta_info['focal_length'],
                        principal_point=self.model.data_meta_info['principal_point'],
                    )
                    d_alphas[d_alphas > 0.1] = 1.0
                    d_alphas[d_alphas <= 0.1] = 0.0
                    d_images = d_images * d_alphas / 255.0
                of_images = self.merge_multiple_images(of_image, (512, 512))
                img_list = [
                    of_images, d_images, result_img, vis_depth(val_results['depth'])
                ]
            else:
                img_list = [result_img, vis_depth(val_results['depth'])]
            img_list = [self.resize(i, (512, 512)) for i in img_list]
            grid_image = torch.cat(img_list)
            grid_image = torchvision.utils.make_grid(grid_image, nrow=grid_image.shape[0], padding=0, pad_value=0)[None].cpu()
            results.append(grid_image)
        results = unnorm_transform(torch.cat(results), self.config_dict['DATASET_AUGMENT']).to(torch.uint8).permute(0, 2, 3, 1)
        video_path = os.path.join(
            self.output_path, 
            '{}_{}{}.mp4'.format(
                frame_key, self.dataname, f'_rotate_{rotation_angle}' if rotation_angle > 0 else '', 
            )
        )
        torchvision.io.write_video(video_path, results, fps=25.0)
        print('Video saved to {}'.format(video_path))

    def run_images(self, image_path, ):
        results = []
        of_image, f_image, f_shape, frame_key, _ = self.get_input_images(image_path)
        for batch_data in self.dataloader:
            points = self.model.get_point_cloud(
                batch_data['d_poses'].to(self.device), batch_data['d_expressions'].to(self.device), 
                f_shape.to(self.device)
            )
            batch_data['f_images'] = f_image.clone()
            batch_data['d_points'] = points
            batch_data = to_gpu(batch_data, self.device)
            val_results = self.model.forward_inference(**batch_data)
            result_img = val_results['gen_sr'].clamp(0, 1)
            _mark_patch = result_img[..., -self._water_mark.shape[-2]:, -self._water_mark.shape[-1]:]
            _mark_patch[self._water_mark_alpha>0.5] = _mark_patch[self._water_mark_alpha>0.5] * 0.5 + \
                                                      self._water_mark[self._water_mark_alpha>0.5] * 0.5
            result_img[..., -self._water_mark.shape[-2]:, -self._water_mark.shape[-1]:] = _mark_patch
            if 'd_images' in batch_data.keys():
                img_list = [of_image[:, 0], batch_data['d_images'], result_img, vis_depth(val_results['depth'])]
            else:
                img_list = [of_image[:, 0], result_img, vis_depth(val_results['depth'])]
            img_list = [self.resize(i, (512, 512)) for i in img_list]
            grid_image = torch.cat(img_list)
            grid_image = torchvision.utils.make_grid(grid_image, nrow=grid_image.shape[0], padding=0, pad_value=0)[None].cpu()
            results.append(grid_image)
        results = unnorm_transform(torch.cat(results), self.config_dict['DATASET_AUGMENT']) / 255.0
        torchvision.utils.save_image(
            results, os.path.join(self.output_path, f'{frame_key}_{self.dataname}.jpg'), nrow=2
        )

    @staticmethod
    def read_image(image_path):
        image = torchvision.io.read_image(image_path, mode=torchvision.io.ImageReadMode.RGB).float()/255.0
        image = torchvision.transforms.functional.resize(image, 512, antialias=True)
        image = torchvision.transforms.functional.center_crop(image, 512)
        return image

    @staticmethod
    def resize(frames, tgt_size=(256, 256)):
        frames = torchvision.transforms.functional.resize(
            frames, tgt_size, antialias=True
        )
        return frames

    @staticmethod
    def merge_multiple_images(batch_images, tgt_size, nrow=2):
        if batch_images.shape[1] == 1:
            input_feature_image = batch_images[:, 0]
        else:
            input_feature_image = []
            for images in batch_images:
                images = images[:4]
                if images.shape[0] < 4:
                    images = torch.cat([
                            images, 
                            images.new_zeros(4-images.shape[0], images.shape[1], images.shape[2], images.shape[3])
                        ], dim=0
                    )
                input_feature_image.append(torchvision.utils.make_grid(images, nrow=2, padding=0))
            input_feature_image = torch.stack(input_feature_image, dim=0)
        input_feature_image = torchvision.transforms.functional.resize(input_feature_image, tgt_size, antialias=True)
        return input_feature_image


if __name__ == "__main__":
    # build args
    parser = argparse.ArgumentParser()
    parser.add_argument('--driver', required=True, type=str)
    parser.add_argument('--input', default=None, type=str)
    parser.add_argument('--resume', '-r', required=True, type=str)
    parser.add_argument('--frame_num', default=150, type=int)
    parser.add_argument('--rotate', default=0, type=int)
    parser.add_argument('--if_video', '-v', action='store_true')
    args = parser.parse_args()
    print(args)
    # set devices
    test_engine = InfEngine(args.resume)
    test_engine.build_dataset(args.driver, args.if_video)
    if args.if_video:
        test_engine.run_video(args.input, args.frame_num, args.rotate)
    else:
        test_engine.run_images(args.input)
