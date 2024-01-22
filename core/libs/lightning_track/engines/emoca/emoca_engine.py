#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import math
import torch
import numpy as np
import torchvision
import torch.nn as nn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import mediapipe
# import face_alignment

from .emoca_v2 import EMOCAV2

FLAME_SCALE = 5.0

class EMOCAEngine(nn.Module):
    def __init__(self, device='cuda', lazy_init=True):
        super().__init__()
        self.device = device
        if not lazy_init:
            self._init_model()

    def _init_model(self, ):
        # emocav2
        emoca_model = EMOCAV2().to(self.device).eval()
        self.emoca_model = emoca_model
        # landmarks
        # self.lmks_model = face_alignment.FaceAlignment(
        #     face_alignment.LandmarksType.TWO_D, device=self.device
        # )
        self.dense_lmks_model = mediapipe.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        # print('EMOCAEngine(Trimmed EMOCAV2) Init Done.')

    @staticmethod
    def _crop_frame(frame, landmark):
        min_xy = landmark.min(dim=0)[0]
        max_xy = landmark.max(dim=0)[0]
        box = torch.tensor([min_xy[0], min_xy[1], max_xy[0], max_xy[1]])
        size = int((box[2]+box[3]-box[0]-box[1])/2*1.25)
        center = torch.tensor([(box[0]+box[2])/2.0, (box[1]+box[3])/2.0])
        frame = torchvision.transforms.functional.crop(
            frame.float(), 
            top=int(center[1]-size/2), left=int(center[0]-size/2),
            height=size, width=size,
        )
        frame = torchvision.transforms.functional.resize(frame, size=224, antialias=True)
        bbox = [center[0]-size/2, center[1]-size/2, center[0]+size/2, center[1]+size/2]
        # torchvision.utils.save_image(frame/255.0, './debug.jpg')
        return frame, bbox

    def process_frame(self, frame):
        if not hasattr(self, 'emoca_model'):
            self._init_model()
        # mediapipe
        lmk_image = frame.permute(1, 2, 0)
        # lmks, scores, detected_faces = self.lmks_model.get_landmarks_from_image(
        #     lmk_image, return_landmark_score=True, return_bboxes=True
        # )
        # if lmks is None:
        #     return None
        lmk_image = lmk_image.to(torch.uint8).cpu().numpy()
        lmks_dense = self.dense_lmks_model.process(lmk_image)
        if lmks_dense.multi_face_landmarks is None:
            return None
        else:
            lmks_dense = lmks_dense.multi_face_landmarks[0].landmark
            lmks_dense = np.array(list(map(lambda l: np.array([l.x, l.y]), lmks_dense)))
            lmks_dense[:, 0] = lmks_dense[:, 0] * lmk_image.shape[1]
            lmks_dense[:, 1] = lmks_dense[:, 1] * lmk_image.shape[0]
            lmks_dense = torch.tensor(lmks_dense)
        croped_frame, bbox = self._crop_frame(frame, lmks_dense)
        bbox = torch.tensor(
            [bbox[0]/frame.shape[2], bbox[1]/frame.shape[1], bbox[2]/frame.shape[2], bbox[3]/frame.shape[1]]
        )
        # please input normed frame
        croped_frame = croped_frame.to(self.device)[None]/255.0
        emoca_result = self.emoca_model.encode(croped_frame)
        results = {
            'emoca_expression': emoca_result['exp'], 
            'emoca_pose': emoca_result['pose'],
            'lmks_dense': lmks_dense,
            # 'lmks': torch.tensor(lmks[0]), 'bbox': bbox
        }
        for key in results.keys():
            if isinstance(results[key], torch.Tensor):
                results[key] = results[key].detach().float().cpu()
        return results

