import os
import inspect
import cv2 
import torch

from .mica import MICA
from .insightface import FaceAnalysis, Face, face_align


class MICAEngine(torch.nn.Module):
    def __init__(self, device='cuda', lazy_init=True):
        super().__init__()
        self.device = device
        if not lazy_init:
            self._init_model()

    def _init_model(self, ):
        # emocav2
        mica_model = MICA().to(self.device).eval()
        self.mica_model = mica_model
        # landmarks
        _abs_script_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        _ckpt_path = os.path.join(_abs_script_path, 'assets')
        retina_face = FaceAnalysis(ckpt_path=_ckpt_path)
        retina_face.prepare(ctx_id=0, det_size=(512, 512), det_thresh=0.4)
        self.retina_face = retina_face

    def _crop_frame(self, frame, ):
        # bboxes, kpss = self.retina_face.det_model.detect(frame, max_num=0, metric='default')
        faces = self.retina_face.get(frame)
        if faces is None or len(faces) == 0:
            return None
        bbox, kps, det_score, lmks = faces[0].bbox, faces[0].kps, faces[0].det_score, faces[0].landmark_3d_68[..., :2]
        face = Face(bbox=bbox, kps=kps, det_score=det_score)
        aimg = face_align.norm_crop(frame, landmark=face.kps)
        blob = cv2.dnn.blobFromImages([aimg], 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        images = torch.tensor(blob[0])[None]

        size = int((bbox[2]+bbox[3]-bbox[0]-bbox[1])/2*1.25)
        center = torch.tensor([(bbox[0]+bbox[2])/2.0, (bbox[1]+bbox[3])/2.0])
        bbox = [center[0]-size/2, center[1]-size/2, center[0]+size/2, center[1]+size/2]
        bbox = torch.tensor(
            [bbox[0]/frame.shape[1], bbox[1]/frame.shape[0], bbox[2]/frame.shape[1], bbox[3]/frame.shape[0]]
        )
        return {'arcface_image': images, 'bbox': bbox, 'kps': kps, 'lmks': lmks,}

    def process_frame(self, frame):
        if not hasattr(self, 'mica_model'):
            self._init_model()
        # image = cv2.imread('carell.jpg')
        # image = torchvision.io.read_image('carell.jpg').permute(1, 2, 0).numpy()
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame = frame.permute(1, 2, 0).numpy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        insightface_res = self._crop_frame(frame)
        if insightface_res is None:
            return None
        mica_shape = self.mica_model(
            insightface_res['arcface_image'].to(self.device).float()
        )[0, :100]
        results = {
            'mica_shape': mica_shape,
            'bbox': insightface_res['bbox'],
            'kps': insightface_res['kps'],
            'lmks': insightface_res['lmks'],
        }
        for key in results.keys():
            if isinstance(results[key], torch.Tensor):
                results[key] = results[key].detach().float().cpu()
        return results
