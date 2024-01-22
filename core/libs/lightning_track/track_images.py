#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import torch
import random
import torchvision

from .engines.mica import MICAEngine
from .engines.emoca import EMOCAEngine
from .lightning_engine import Lightning_Engine
from .engines.human_matting import StyleMatteEngine as HumanMattingEngine

class TrackEngine:
    def __init__(self, focal_length=12.0, device='cuda'):
        random.seed(42)
        self._device = device
        # paths and data engine
        self.matting_engine = HumanMattingEngine(device=device)
        self.mica_engine = MICAEngine(device=device, lazy_init=True)
        self.emoca_engine = EMOCAEngine(device=device, lazy_init=True)
        self.lightning_engine = Lightning_Engine(device=device, lazy_init=True)
        calibration_results = {
            'focal_length':torch.tensor([focal_length]), 
            'principal_point': torch.tensor([0., 0.]), 'verts_scale': 5.0
        }
        self.calibration_results = calibration_results

    def track_images(self, images, matting=True, background=0.0, vis_path=None):
        if images[0].max() > 2.0:
            raise Warning('Images should be in [0, 1].')
        assert type(images) == list
        # build video data
        if matting:
            images = [
                self.matting_engine(
                    image, return_type='matting', background_rgb=background
                ).cpu()
                for image in images
            ]
        base_results = {}
        for idx, image in enumerate(images):
            base_results[str(idx)] = self.track_base(image.clone()*255.0)
        # track lightning
        lightning_results = self.track_lightning(
            base_results, 
            batch_frames=torch.stack(images).to(self._device).float(), vis_path=vis_path
        )
        return images, lightning_results

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

    def track_lightning(self, base_result, batch_frames=None, vis_path=None):
        self.lightning_engine.init_model(self.calibration_results, image_size=512)
        base_result = {k: v for k, v in base_result.items() if v is not None}
        mini_batchs = self.build_minibatch(list(base_result.keys()))
        #     batch_frames = torch.stack([lmdb_engine[key] for key in mini_batchs[0]][:20]).to(self._device).float()
        # else:
            # batch_frames = None
        lightning_results = {}
        for mini_batch in mini_batchs:
            mini_batch_emoca = [base_result[key] for key in mini_batch]
            mini_batch_emoca = torch.utils.data.default_collate(mini_batch_emoca)
            mini_batch_emoca = {k: v.to(self._device) for k, v in mini_batch_emoca.items()}
            lightning_result, visualization = self.lightning_engine.lightning_optimize(
                mini_batch, mini_batch_emoca, batch_frames=batch_frames
            )
            batch_frames = None
            if visualization is not None:
                torchvision.utils.save_image(visualization, vis_path)
            lightning_results.update(lightning_result)
        for fkey in lightning_results:
            for key in list(lightning_results[fkey].keys()):
                if isinstance(lightning_results[fkey][key], torch.Tensor):
                    lightning_results[fkey][key] = lightning_results[fkey][key].float().cpu().numpy()
        return lightning_results

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

