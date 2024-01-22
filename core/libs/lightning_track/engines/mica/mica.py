import os
import inspect
import torch
import torch.nn.functional as F

from .models.arcface import Arcface
from .models.generator import Generator


class MICA(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.arcface = Arcface()
        self.flameModel = Generator(512, 300, 300, 3)
        # load model
        _abs_script_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        model_path = os.path.join(_abs_script_path, 'assets', 'mica_clean.pth')
        checkpoint = torch.load(model_path)
        if 'arcface' in checkpoint:
            self.arcface.load_state_dict(checkpoint['arcface'])
        if 'flameModel' in checkpoint:
            self.flameModel.load_state_dict(checkpoint['flameModel'])

    def forward(self, arcface_imgs):
        identity_code = F.normalize(self.arcface(arcface_imgs))
        pred_shape_code = self.flameModel(identity_code)
        return pred_shape_code
