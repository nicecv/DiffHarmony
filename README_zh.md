# DiffHarmony: Latent Diffusion Model Meets Image Harmonization

[DiffHarmony](https://arxiv.org/abs/2404.06139) 和 DiffHarmony++（论文即将发布）的官方 Pytorch 实现。

完整的会议海报在[这里](./assets/poster.pdf)。

## 准备

### 环境

首先，准备一个虚拟环境。你可以使用 **conda** 或其他你喜欢的工具。
```shell
python 3.10
pytorch 2.2.0
cuda 12.1
xformers 0.0.24
```

然后，安装所需的依赖。
```shell
pip install -r requirements.txt
```

## 数据集

从[这里](https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4)下载 iHarmony4 数据集。

确保目录结构如下：
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

`train.jsonl` 文件内容格式如下：
```json
{"file_name": "HAdobe5k/composite_images/a0001_1_1.jpg", "text": ""}
{"file_name": "HAdobe5k/composite_images/a0001_1_2.jpg", "text": ""}
{"file_name": "HAdobe5k/composite_images/a0001_1_3.jpg", "text": ""}
{"file_name": "HAdobe5k/composite_images/a0001_1_4.jpg", "text": ""}
...
```
所有 `file_name` 来自原始的 `IHD_train.txt`。`test.jsonl` 和 `IHD_test.txt` 也是如此。

## 训练
### 训练 diffharmony 模型
```shell
sh scripts/train_diffharmony.sh
```

### 训练 refinement 模型
```shell
sh scripts/train_refinement_stage.sh
```

### 训练 condition vae（cvae）
```shell
sh scripts/train_cvae.sh
```

### 训练 diffharmony-gen 和 cvae-gen
在你的训练参数中添加以下内容：
```shell
$script
    ...
    --mode "inverse"
```
基本上，它会使用真实图像作为条件，而不是合成图像。

### （可选）在线训练 condition vae
参考 `scripts/train/cvae_online.py`

### （可选）使用生成的数据训练 cvae
参考 `scripts/train/cvae_with_gen_data.py`

目的是进一步提高 cvae 在特定领域(即我们生成的数据集)的性能。

## 推理
推理 iHarmony4 数据集
```shell
sh scripts/inference.sh
```

### 使用 diffharmony-gen 和 cvae-gen 增强 HFlickr 和 Hday2night
```shell
sh scripts/inference_generate_data.sh
```
`all_mask_metadata.jsonl` 文件内容格式如下：
```json
{"file_name": "masks/f800_1.png", "text": ""}
{"file_name": "masks/f801_1.png", "text": ""}
{"file_name": "masks/f803_1.png", "text": ""}
{"file_name": "masks/f804_1.png", "text": ""}
...
```

### 制作 HumanHarmony 数据集
首先，生成一些候选的合成图像。

然后，使用和谐分类器选择最不和谐的图像。
```shell
python scripts/misc/classify_cand_gen_data.py
```

## 评估
```shell
sh scripts/evaluate.sh
```

## 预训练模型
[Baidu](https://pan.baidu.com/s/1IkF6YP4C3fsEAi0_9eCESg), 提取码: aqqd

[Google Drive](https://drive.google.com/file/d/1rezNdcuZbwejbC9rH9S1SUuaWTGTz_wG/view?usp=drive_link)

## 引用
如果你觉得这个工作有用，请考虑引用：
```bibtex
@inproceedings{zhou2024diffharmony,
  title={DiffHarmony: Latent Diffusion Model Meets Image Harmonization},
  author={Zhou, Pengfei and Feng, Fangxiang and Wang, Xiaojie},
  booktitle={Proceedings of the 2024 International Conference on Multimedia Retrieval},
  pages={1130--1134},
  year={2024}
}
```

## 联系
如果你有任何问题，请随时通过 `zhoupengfei@bupt.edu.cn` 联系我。
