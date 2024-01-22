import os
import torch
import inspect
from .models.DecaEncoder import ResnetEncoder

class EMOCAV2(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.params_list = [100, 50, 50, 6, 3, 27] # n_shape, n_tex, n_exp, n_pose, n_cam, n_light
        self.params_keys = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        # 1) build coarse encoder
        self.E_flame = ResnetEncoder(outsize=sum(self.params_list))
        # 2) add expression decoder
        self.E_expression = ResnetEncoder(50) # exp length
        self.E_expression.encoder.load_state_dict(self.E_flame.encoder.state_dict())

        _abs_script_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        _ckpt_path = os.path.join(_abs_script_path, 'assets', 'emoca_v2.pth')
        ckpt = torch.load(_ckpt_path, map_location='cpu')
        trimmed_ckpt = {}
        for key in list(ckpt.keys()):
            if 'E_flame' in key or 'E_expression' in key:
                trimmed_ckpt[key.replace('deca.', '')] = ckpt[key]
        self.load_state_dict(trimmed_ckpt, strict=True)

    @torch.no_grad()
    def encode(self, images):
        deca_code = self.E_flame(images)
        exp_deca_code = self.E_expression(images)
        codedict = self.decompose_code(deca_code, exp_deca_code)
        return codedict

    def decompose_code(self, deca_code, expdeca_code):
        codedict = {}
        start = 0
        for idx, pa in enumerate(self.params_list):
            codedict[self.params_keys[idx]] = deca_code[:, start:start + pa]
            start = start + pa
        # codedict['light'] = codedict['light'].reshape(deca_code.shape[0], 9, 3)
        codedict['exp'] = expdeca_code
        del codedict['light'], codedict['tex']
        for key in codedict.keys():
            codedict[key] = codedict[key][0]
        return codedict
