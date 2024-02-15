<h1 align="center"><b><img src="./demos/gpavatar_logo.png" width="350"/></b></h1>
<h3 align="center">
    <a href='https://arxiv.org/abs/2401.10215'><img src='https://img.shields.io/badge/ArXiv-PDF-red'></a> &nbsp; 
    <a href='https://xg-chu.github.io/project_gpavatar/'><img src='https://img.shields.io/badge/Project-Page-blue'></a> &nbsp; 
    <a href='https://www.youtube.com/watch?v=7A3DMaB6Zk0'><img src='https://img.shields.io/badge/Youtube-Video-red'></a> &nbsp; 
</h3>


<h5 align="center">
    <a href="https://github.com/xg-chu">Xuangeng Chu</a><sup>1,2</sup>&emsp;
    <a href="https://yu-li.github.io">Yu Li</a><sup>2</sup>&emsp;
    <a href="https://ailingzeng.site">Ailing Zeng</a><sup>2</sup>&emsp;
    <a href="https://tianyu-yang.com">Tianyu Yang</a><sup>2</sup>&emsp;
    <a href="https://scholar.google.com/citations?user=Xf5_TfcAAAAJ&hl=zh-CN">Lijian Lin</a><sup>2</sup>&emsp;
    <a href="http://liuyunfei.net">Yunfei Liu</a><sup>2</sup>&emsp;
    <a href="https://www.mi.t.u-tokyo.ac.jp/harada/">Tatsuya Harada</a><sup>1,3</sup>
    <br>
    <sup>1</sup>The University of Tokyo,
    <sup>2</sup>International Digital Economy Academy (IDEA),
    <sup>3</sup>RIKEN AIP
</h5>

<h3 align="center">
🤩ICLR 2024🤩
</h3>

<div align="center"> 
    <div align="center"> 
        <b><img src="./demos/teaser.gif" alt="drawing" width="800"/></b>
    </div>
    <b>
        GPAvatar reconstructs controllable 3D head avatars from one or several images in a single forward pass.
    </b>
    <br>
    <b>
        More results can be seen from our <a href="https://xg-chu.github.io/project_gpavatar/">Project Page</a>.
    </b>
</div>

<!-- ## TO DO
We are now preparing the <b>pre-trained model and quick start materials</b> and will release it within a week. -->

## Installation
<details>
<summary><span >Install step by step</span></summary>

```
conda create -n track python=3.9
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
pip3 install mediapipe tqdm rich lmdb einops colored ninja av opencv-python scikit-image onnxruntime-gpu onnx transformers pykalman
```
</details>

## Preparation
Build resources with ```bash ./build_resources.sh```.

Download the [model checkpoint](https://github.com/xg-chu/GPAvatar/releases/download/v1.0.0/one_model.ckpt) and put it at ```checkpoints/one_model.ckpt```.

## Quick Start
Driven by images:
```
python inference.py -r ./checkpoints/one_model.ckpt --driver ./demos/drivers/pdriver --input ./demos/examples/real1
```
or driven by video:
```
python inference.py -r ./checkpoints/one_model.ckpt --driver ./demos/drivers/vdriver1 --input ./demos/examples/art1 -v
``` 

## Citation
If you find our work useful in your research, please consider citing:
```bibtex
@inproceedings{
    chu2024gpavatar,
    title   = {GPAvatar: Generalizable and Precise Head Avatar from Image(s)}, 
    author  = {Xuangeng Chu and Yu Li and Ailing Zeng and Tianyu Yang and Lijian Lin and Yunfei Liu and Tatsuya Harada},
    journal = {ICLR},
    year    = {2024},
}
```

## Acknowledgements
Some part of our work is built based on FLAME, StyleMatte, EMOCA and MICA. 
The GPAvatar Logo is designed by Caihong Ning.
We thank you for sharing their wonderful code and their wonderful work.
- **FLAME**: https://flame.is.tue.mpg.de
- **StyleMatte**: https://github.com/chroneus/stylematte
- **EMOCA**: https://github.com/radekd91/emoca
- **MICA**: https://github.com/Zielon/MICA


<!-- ## Installation
### Build environment
<details>
<summary><span >Install step by step</span></summary>

```
conda create -n track python=3.9
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
pip3 install mediapipe tqdm rich lmdb einops colored ninja av opencv-python scikit-image onnxruntime-gpu onnx transformers pykalman
```

</details>
<details>

<summary><span style="font-weight: bold;">Install with environment.yml (recommend)</span></summary>

```
conda env create -f environment.yml
```

</details>


### Build source
Check the ```build_resources.sh```.


## Fast start

Track on video:
```
python track_video.py -v demos/demo.mp4 --synthesis
```
or track all videos in a directory:
```
python track_video.py -v demos/ --no_vis
``` -->
