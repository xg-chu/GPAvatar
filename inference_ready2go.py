#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import torch
import torchvision
from inference_tools import gpavatar_r2g

### ------------------- fast run  ------------------- ###
model = gpavatar_r2g().cuda()
model.build_avatar(inp_track=None) # should be online lightning track results

render_res = model(expression=torch.zeros(1, 50).cuda(), pose=torch.zeros(1, 6).cuda())
torchvision.utils.save_image(render_res, 'debug.jpg')

### ------------ run with inp&tgt image ------------- ###
def read_image(image_path):
    image = torchvision.io.read_image(image_path, mode=torchvision.io.ImageReadMode.RGB).float()/255.0
    image = torchvision.transforms.functional.resize(image, 512, antialias=True)
    image = torchvision.transforms.functional.center_crop(image, 512)
    return image
def warp_inp_image(input_images, inp_results):
    from core.data.tools import perspective_input
    frame_trans = torch.cat([torch.tensor(inp_results[str(i)]['transform_matrix'][None]) for i in range(len(input_images))])
    feature_image = torch.stack([
        perspective_input(
            input_images[i][None].cuda(), torch.tensor(inp_results[str(i)]['transform_matrix'][None]).cuda(), 
            {'focal_length': 12.0, 'principal_point': [0.0, 0.0]}, fill=0.0
        )[0]
        for i in range(len(input_images))
    ])
    feature_shape = torch.stack([torch.tensor(inp_results[str(i)]['mica_shape']) for i in range(len(input_images))]).mean(dim=0)
    return {
        'feature_image': feature_image.cpu(), 'feature_shape': feature_shape.cpu(), 'frame_trans': frame_trans.cpu()
    }

## track part
from core.libs.lightning_track import TrackEngine
track_engine = TrackEngine(focal_length=12.0, device='cuda')
inp_image = read_image('demos/examples/art0/0.jpg')
tgt_image = read_image('demos/examples/art1/1.jpg')
inp_images, inp_results = track_engine.track_images([inp_image]) # error may occur
if inp_results.keys() == []:
    raise Exception('Track failed!')
inp_track = warp_inp_image(inp_images, inp_results)

tgt_images, tgt_result = track_engine.track_images([tgt_image]) # error may occur
if tgt_result.keys() == []:
    raise Exception('Track failed!')
tgt_result = tgt_result['0']
## model part
model = gpavatar_r2g().cuda()
model.build_avatar(inp_track=inp_track) # should be online lightning track results
render_res = model(
    expression=torch.tensor(tgt_result['emoca_expression'][None]).cuda(), 
    pose=torch.tensor(tgt_result['emoca_pose'][None]).cuda(),
    transform_matrix= torch.tensor(tgt_result['transform_matrix'][None]).cuda()
)[0].cpu()
torchvision.utils.save_image([inp_image, tgt_image, render_res], 'debug.jpg', padding=0)
