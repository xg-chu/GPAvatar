#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import sys
import torch
import random
import torchvision
from tqdm.rich import tqdm

from .mica import MICAEngine
from .emoca import EMOCAEngine
from .lightning_engine import Lightning_Engine
from .synthesis_engine import Synthesis_Engine
from .human_matting import StyleMatteEngine as HumanMattingEngine

class TrackEngine:
    def __init__(self, focal_length=8.0, device='cuda'):
        random.seed(42)
        self._device = device
        # paths and data engine
        self.matting_engine = HumanMattingEngine(device=device)
        self.mica_engine = MICAEngine(device=device, lazy_init=True)
        self.emoca_engine = EMOCAEngine(device=device, lazy_init=True)
        self.lightning_engine = Lightning_Engine(device=device, lazy_init=True)
        self.synthesis_engine = Synthesis_Engine(device=device, lazy_init=True)
        calibration_results = {
            'focal_length':torch.tensor([focal_length]), 
            'principal_point': torch.tensor([0., 0.]), 'verts_scale': 5.0
        }
        self.calibration_results = calibration_results

    def build_video(self, video_path, output_path, matting=True, background=0.0):
        from utils.lmdb_utils import LMDBEngine
        video_name = os.path.basename(video_path).split('.')[0]
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(os.path.join(output_path, 'img_lmdb')):
            lmdb_engine = LMDBEngine(os.path.join(output_path, 'img_lmdb'), write=True)
            video_reader = torchvision.io.VideoReader(src=video_path)
            meta_data = video_reader.get_metadata()['video']
            for fidx, frame_data in tqdm(enumerate(iter(video_reader)), total=int(meta_data['fps'][0]*meta_data['duration'][0]), ncols=80, colour='#95bb72'):
                if meta_data['fps'][0] > 50:
                    if fidx % 2 == 0:
                        continue
                frame, pts = frame_data['data'], frame_data['pts']
                frame = torchvision.transforms.functional.resize(frame, 512, antialias=True) 
                frame = torchvision.transforms.functional.center_crop(frame, 512)
                if matting:
                    frame = self.matting_engine(
                        frame/255.0, return_type='matting', background_rgb=background
                    ).cpu()*255.0
                lmdb_engine.dump(f'{video_name}_{fidx}', payload=frame, type='image')
            lmdb_engine.random_visualize(os.path.join(output_path, 'img_lmdb', 'visualize.jpg'))
            lmdb_engine.close()
            return meta_data['fps'][0]
        else:
            video_reader = torchvision.io.VideoReader(src=video_path)
            meta_data = video_reader.get_metadata()['video']
            return meta_data['fps'][0]

    def track_base(self, image_tensor, ):
        # EMOCA
        # sys.stdout = open(os.devnull, 'w')
        # sys.stdout = sys.__stdout__
        mica_result = self.mica_engine.process_frame(image_tensor)
        emoca_result = self.emoca_engine.process_frame(image_tensor)
        if emoca_result is not None and mica_result is not None:
            for key in list(emoca_result.keys()):
                if isinstance(emoca_result[key], torch.Tensor):
                    emoca_result[key] = emoca_result[key].float().cpu().numpy()
        else:
            return None
        base_result = {**mica_result, **emoca_result}
        return base_result

    def track_lightning(self, base_result, lmdb_engine=None, vis_path=None):
        self.lightning_engine.init_model(self.calibration_results, image_size=512)
        base_result = {k: v for k, v in base_result.items() if v is not None}
        mini_batchs = self.build_minibatch(list(base_result.keys()))
        if lmdb_engine is not None:
            batch_frames = torch.stack([lmdb_engine[key] for key in mini_batchs[0]][:20]).to(self._device).float()
        else:
            batch_frames = None
        lightning_results = {}
        for mini_batch in tqdm(mini_batchs, ncols=80, colour='#95bb72'):
            mini_batch_emoca = [base_result[key] for key in mini_batch]
            mini_batch_emoca = torch.utils.data.default_collate(mini_batch_emoca)
            mini_batch_emoca = {k: v.to(self._device) for k, v in mini_batch_emoca.items()}
            lightning_result, visualization = self.lightning_engine.lightning_optimize(
                mini_batch, mini_batch_emoca, batch_frames=batch_frames
            )
            batch_frames = None
            if visualization is not None:
                torchvision.utils.save_image(visualization, os.path.join(vis_path, 'lightning.jpg'))
            lightning_results.update(lightning_result)
        for fkey in lightning_results:
            for key in list(lightning_results[fkey].keys()):
                if isinstance(lightning_results[fkey][key], torch.Tensor):
                    lightning_results[fkey][key] = lightning_results[fkey][key].float().cpu().numpy()
        return lightning_results

    def track_synthesis(self, base_result, lightning_result, lmdb_engine, vis_path=None):
        self.synthesis_engine.init_model(self.calibration_results)
        lightning_result = {k: v for k, v in lightning_result.items() if v is not None}
        # texture
        tex_frames = random.sample(list(lmdb_engine.keys()), 16)
        tex_batch = [lightning_result[key] for key in tex_frames]
        tex_batch = torch.utils.data.default_collate(tex_batch)
        tex_batch = {k: v.to(self._device) for k, v in tex_batch.items()}
        tex_batch['frames'] = torch.stack([lmdb_engine[k] for k in tex_frames]).to(self._device).float()
        texture_result, visualization = self.synthesis_engine.texture_optimize(tex_batch)
        torchvision.utils.save_image(visualization, os.path.join(vis_path, 'texture.jpg'))
        # tracking
        mini_batchs = self.build_minibatch(list(lightning_result.keys()), batch_size=32)
        synthesis_results = {}
        for bidx, mini_batch in enumerate(tqdm(mini_batchs, ncols=80, colour='#95bb72')):
            mini_batch_lightning = [lightning_result[key] for key in mini_batch]
            mini_batch_lightning = torch.utils.data.default_collate(mini_batch_lightning)
            mini_batch_lightning = {k: v.to(self._device) for k, v in mini_batch_lightning.items()}
            mini_batch_lightning['lmks'] = torch.stack([torch.tensor(base_result[key]['lmks']) for key in mini_batch]).to(self._device).float()
            mini_batch_lightning['lmks_dense'] = torch.stack([torch.tensor(base_result[key]['lmks_dense']) for key in mini_batch]).to(self._device).float()
            mini_batch_lightning['frames'] = torch.stack([lmdb_engine[key] for key in mini_batch]).to(self._device).float()
            mini_batch_texture = {
                'tex_params': texture_result['tex_params'].mean(dim=0, keepdim=True).expand(len(mini_batch), -1).to(self._device).float(),
                'sh_params': texture_result['sh_params'].mean(dim=0, keepdim=True).expand(len(mini_batch), -1, -1).to(self._device).float(),
            }
            synthesis_result, visualization = self.synthesis_engine.synthesis_optimize(
                mini_batch, mini_batch_lightning, mini_batch_texture, visualize=bidx==0
            )
            if visualization is not None:
                torchvision.utils.save_image(visualization, os.path.join(vis_path, 'synthesis.jpg'))
            synthesis_results.update(synthesis_result)
        for fkey in synthesis_results:
            for key in list(synthesis_results[fkey].keys()):
                if isinstance(synthesis_results[fkey][key], torch.Tensor):
                    synthesis_results[fkey][key] = synthesis_results[fkey][key].float().cpu().numpy()
        return synthesis_results

    @staticmethod
    def build_minibatch(all_frames, batch_size=256):
        # random.shuffle(all_frames)
        # all_frames = sorted(all_frames, key=lambda x: int(x.split('_')[-1]))
        all_mini_batch, mini_batch = [], []
        for frame_name in all_frames:
            mini_batch.append(frame_name)
            if len(mini_batch) % batch_size == 0:
                all_mini_batch.append(mini_batch)
                mini_batch = []
        if len(mini_batch):
            all_mini_batch.append(mini_batch)
        return all_mini_batch
