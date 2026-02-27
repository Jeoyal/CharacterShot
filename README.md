# [Arxiv] CharacterShot: Controllable and Consistent 4D Character Animation
[**CharacterShot: Controllable and Consistent 4D Character Animation**](https://arxiv.org/abs/2508.07409)

[Junyao Gao<sup>&Dagger;</sup>](https://jeoyal.github.io/home/), [Jiaxing Li<sup>&Dagger;</sup>](https://scholar.google.com/citations?hl=zh-CN&user=LNtaU5YAAAAJ), [Wenran Liu](https://github.com/liuwenran), [Yanhong Zeng](https://zengyh1900.github.io/), [Fei Shen](https://muzishen.github.io/), [Kai Chen](https://chenkai.site/), [Yanan Sun*](https://scholar.google.com/citations?hl=zh-CN&user=6TA1oPkAAAAJ), [Cairong Zhao*](https://vill-lab.github.io/)

(<sup>&Dagger;</sup> equal contributions, * corresponding authors)

<a href='https://arxiv.org/abs/2508.07409'><img src='https://img.shields.io/badge/arXiv-2508.07409-b31b1b.svg'></a> 
<a href='https://huggingface.co/Gaojunyao/CharacterShot'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
<a href='https://huggingface.co/datasets/Gaojunyao/Character4D'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue'></a>

**CharacterShot** supports diverse character designs and custom motion control (2D pose sequence), enabling 4D character animation in minutes and without specialized hardware.

**Your star is our fuel!  We're revving up the engines with it!**

<img src="__assets__/teaser.gif">

## News
- [2026/2/27] 🔥 We release the training/inference codes, [models](https://huggingface.co/Gaojunyao/CharacterShot) and [dataset](https://huggingface.co/datasets/Gaojunyao/Character4D) of CharacterShot!!!
- [2025/8/12] 🔥 We release the [paper](https://www.arxiv.org/abs/2503.00740) of CharacterShot!!!

## TODO List
- [x] Character4D Dataset.
- [x] Training Code.
- [x] Inference Code.
- [ ] 4D Optimization Code (expected in March).

## Get Started
CharacterShot supports: 1) 2D character animation from a character image and pose video; 2) multi-view videos generation from multi-view images of a character and pose images; 3) 4D optimization from multi-view videos.
### Clone the Repository

```
git clone git@github.com:Jeoyal/CharacterShot.git
cd ./CharacterShot
```

### Environment Setup

This script has been tested on CUDA version of 12.4.

```
conda create -n charactershot python==3.10
conda activate charactershot
pip install -r requirements.txt
pip install git+https://github.com/camenduru/simple-knn
pip install git+https://github.com/ingra14m/depth-diff-gaussian-rasterization
```

### Downloading Checkpoints

1. Download the checkpoints of 2D character animation and multi-view generation from [here](https://huggingface.co/Gaojunyao/Character2D) and [here](https://huggingface.co/Gaojunyao/CharacterShot).

2. Download [DWPose](https://huggingface.co/yzd-v/DWPose/tree/main) pretrained model:
    ```
    mkdir -p inference/dwpose/models/
    wget https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx?download=true -O inference/dwpose/models/yolox_l.onnx
    wget https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx?download=true -O inference/dwpose/models/dw-ll_ucoco_384.onnx
    ```

### Preparing Inference Samples
Construct your inference samples in the following structure:
```
├── inference/
│   ├──examples/
│       ├── 4d/
│           ├── images/
│               ├── 001/ # character images in 21 views.
│                   ├── view0.png 
│                   ├── ...
│           ├── poses/
│               ├── 001/ # pose images.
│                   ├── 0.png 
│                   ├── ... 
│       ├── 2d/
│           ├── images/
│               ├── 001.png
│               ├── ... # character images.
│           ├── poses/
│               ├── 001/ # pose images.
│                   ├── 0.png 
│                   ├── ... 
```

### Running Inference
For 2D character animation:
```
python -m inference.cli_demo_4d --image_path inference/examples/2d/images/ --func_type 2dpretrain --model_path Gaojunyao/Character2D/
```
For multi-view videos generation:
```
python -m inference.cli_demo_4d --image_path inference/examples/4d/images/ --func_type 4dfinetune --model_path Gaojunyao/CharacterShot/
```

### Training
Navigate into `./finetune` and download the checkpoints of [CogVideoX-5b-I2V](https://huggingface.co/zai-org/CogVideoX-5b-I2V).

For 2D character animation pretraining, you should prepare your own dataset into `./data/i2v/2dpretrain` and start training with:
```
bash train_2d_pretrain.sh
```
After that, to fine-tune the model for multi-view video generation, download the our proposed 4D dataset [Character4D](https://huggingface.co/datasets/Gaojunyao/Character4D) and follow the steps below to prepare cached input latents:
```
python prepare_multiview_cache.py
python convert2meta.py
```
And start training with:
```
bash train_4d_finetune.sh
```
Please set `--pose_model_path` in `train_4d_finetune.sh` to the checkpoint from the 2D pretraining stage, or continue training from [Gaojunyao/Character2D](https://huggingface.co/Gaojunyao/Character2D).

## Character4D Dataset
We construct a large-scale 4D character dataset by filtering high-quality characters from VRoid Hub, collecting a total of 13,115 characters in OBJ format. We then retarget and bind 40 diverse motions (e.g., dancing, singing, and jumping), using skeletons from Mixamo, to these characters. Next, we render all characters from 21 viewpoints in the A-pose and under various motions. Finally, we release the raw and rigged OBJ files, along with the rendered images and pose visualizations, at [this link](https://huggingface.co/datasets/Gaojunyao/Character4D).

## License and Citation
All assets and code are under the [license](./LICENSE) unless specified otherwise.

If this work is helpful for your research, please consider citing the following BibTeX entry.
```
@article{gao2025charactershot,
  title={CharacterShot: Controllable and Consistent 4D Character Animation},
  author={Gao, Junyao and Li, Jiaxing and Liu, Wenran and Zeng, Yanhong and Shen, Fei and Chen, Kai and Sun, Yanan and Zhao, Cairong},
  journal={arXiv preprint arXiv:2508.07409},
  year={2025}
}
```

## Acknowledgements
The code is built upon [CogVideo](https://github.com/THUDM/CogVideo), [WideRange4D](https://github.com/Gen-Verse/WideRange4D) and [4DGaussians]( https://github.com/hustvl/4DGaussians).
