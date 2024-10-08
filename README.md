[中文](./README_zh.md)

# DiffHarmony: Latent Diffusion Model Meets Image Harmonization

The official pytorch implementation of [DiffHarmony](https://arxiv.org/abs/2404.06139) and DiffHarmony++ (paper release soon).

Full Conference Poster is [here](./assets/poster.pdf).

## Preparation 

### enviroment

First, prepare a virtual env. You can use **conda** or anything you like. 
```shell
python 3.10
pytorch 2.2.0
cuda 12.1
xformers 0.0.24
```

Then, install requirements.
```shell
pip install -r requirements.txt
```
## dataset

Download iHarmony4 dataset from [here](https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4).

Make sure the structure is just like that:
```shell
data/iHarmony4
|- HCOCO
    |- composite_images
    |- masks
    |- real_images
    |- ...
|- HAdobe5k
|- HFlickr
|- Hday2night
|- train.jsonl
|- test.jsonl
```

The content in `train.jsonl` fit the following format
```json
{"file_name": "HAdobe5k/composite_images/a0001_1_1.jpg", "text": ""}
{"file_name": "HAdobe5k/composite_images/a0001_1_2.jpg", "text": ""}
{"file_name": "HAdobe5k/composite_images/a0001_1_3.jpg", "text": ""}
{"file_name": "HAdobe5k/composite_images/a0001_1_4.jpg", "text": ""}
...
```
All `file_name` are from the original `IHD_train.txt`. Same way with `test.jsonl` and `IHD_test.txt`.


## Training
### Train diffharmony model
```shell
sh scripts/train_diffharmony.sh
```

### Train refinement model
```shell
sh scripts/train_refinement_stage.sh
```

### Train condition vae (cvae)
```shell
sh scripts/train_cvae.sh
```

### Train diffharmony-gen and cvae-gen
Just add this in your training args:
```shell
$script
    ...
    --mode "inverse"
```
Basically it will use ground truth images as condition instead of composite images.


### (optional) online training of condition vae
refer to `scripts/train/cvae_online.py`

### (optional) train cvae with generated data
refer to `scripts/train/cvae_with_gen_data.py`

Purpose here is trying to improve cvae performance further on specific domain, i.e. our generated dataset.

## Inference 
Inference iHarmony4 dataset
```shell
sh scripts/inference.sh
```

### use diffharmony-gen and cvae-gen to augment HFlickr and Hday2night
```shell
sh scripts/inference_generate_data.sh
```
The `all_mask_metadata.jsonl` file as its name fits following format:
```json
{"file_name": "masks/f800_1.png", "text": ""}
{"file_name": "masks/f801_1.png", "text": ""}
{"file_name": "masks/f803_1.png", "text": ""}
{"file_name": "masks/f804_1.png", "text": ""}
...
```

### Make HumanHarmony dataset
First, generate some candidate composite images.

Then, use harmony classifier to select the most unharmonious images.
```shell
python scripts/misc/classify_cand_gen_data.py
```

## Evaluation
```shell
sh scripts/evaluate.sh
```
## Pretrained Models
[Baidu](https://pan.baidu.com/s/1IkF6YP4C3fsEAi0_9eCESg), code: aqqd

[Google Drive](https://drive.google.com/file/d/1rezNdcuZbwejbC9rH9S1SUuaWTGTz_wG/view?usp=drive_link)

## Citation
If you find this work useful, please consider citing:
```bibtex
@inproceedings{zhou2024diffharmony,
  title={DiffHarmony: Latent Diffusion Model Meets Image Harmonization},
  author={Zhou, Pengfei and Feng, Fangxiang and Wang, Xiaojie},
  booktitle={Proceedings of the 2024 International Conference on Multimedia Retrieval},
  pages={1130--1134},
  year={2024}
}
```
## Contact
If you have any questions, please feel free to contact me via `zhoupengfei@bupt.edu.cn` .
