from os.path import join as pjoin
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import numpy as np
import sys
import os
import json
from collections import defaultdict
import time
from pathlib import Path
from tqdm import tqdm
from pprint import pprint
import torchvision.transforms.functional as TF

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--data_dir", type=str, default="./data/iHarmony4")
parser.add_argument("--json_file_path", type=str, default="DataSpecs/all_test.jsonl")
parser.add_argument("--resolution",  type=int, default=256)
parser.add_argument("--use_gt_bg",  action="store_true")
parser.add_argument("--ct_max", type=int, default=10000000)
args = parser.parse_args()

# 从命令行参数获取输入目录和可选的图像大小和最大处理数
input_dir = args.input_dir
ct_max = args.ct_max
output_dir = args.output_dir
json_file_path = args.json_file_path
data_dir = args.data_dir

Path(output_dir).mkdir(parents=True, exist_ok=True)
(Path(output_dir) / "imgs").mkdir(parents=True, exist_ok=True)

# 初始化变量
psnr_total = 0
mse_total = 0
fmse_total = 0
bmse_total = 0
ds2psnr = defaultdict(list)
ds2mse = defaultdict(list)
ds2fmse = defaultdict(list)
ds2bmse = defaultdict(list)
ct = 0
html = True
htmlstr = (
    '<html><head>harmony</head><body><div style="display: flex; flex-wrap: wrap;">'
)
st = time.time()


def get_paths(comp_path):
    # .../ds_name/composite_images/xxx_x_x.jpg
    parts = comp_path.split("/")
    img_name_parts = parts[-1].split("_")
    real_path = os.path.join(*parts[:-2], "real_images", f"{img_name_parts[0]}.jpg")
    mask_path = os.path.join(
        *parts[:-2], "masks", f"{img_name_parts[0]}_{img_name_parts[1]}.png"
    )

    return {
        "real": real_path,
        "mask": mask_path,
        "comp": comp_path,
    }

comp_to_paths = get_paths

# 读取jsonl文件并存储到列表中
def read_jsonl_to_list(jsonl_file):
    data_list = []
    with open(jsonl_file, "r") as file:
        for line in file:
            # 解析每一行并将其添加到列表中
            data = json.loads(line)
            data_list.append(data)
    return data_list


json_file = read_jsonl_to_list(os.path.join(data_dir, json_file_path))


dataset_map = {
    "a": "HAdobe5k",
    "c": "HCOCO",
    "d": "Hday2night",
    "f": "HFlickr",
    "s": "SAM",
    "g": "Gen",
}

def possible_resize(image:Image.Image, resolution:int):
    if image.size != (resolution, resolution):
        image = TF.resize(image, [resolution, resolution], antialias=True)
    return image

from typing import List

ds2mask = defaultdict(list)
with open(pjoin(output_dir, "output.log"), "w") as output_file:
    # 读取JSON文件并处理图像
    for i, json_data in tqdm(enumerate(json_file), total=len(json_file)):
        all_paths = comp_to_paths(pjoin(data_dir, json_data["file_name"]))
        target_path = all_paths["real"]
        mask_path = all_paths["mask"]
        comp_path = all_paths["comp"]

        harm_path = os.path.join(input_dir, comp_path.split("/")[-1].replace(".jpg", ".png"))
        harm_name = harm_path.split("/")[-1]

        ot_file = harm_path

        dataset = dataset_map[harm_name[0]]

        try:
            if os.path.exists(ot_file):
                # import pdb ; pdb.set_trace()
                ot_img = Image.open(ot_file).convert("RGB")
                ot_img = possible_resize(ot_img, args.resolution)

                mk_img = Image.open(mask_path).convert("1")
                mk_img = possible_resize(mk_img, args.resolution)

                gt_img = Image.open(target_path).convert("RGB")
                gt_img = possible_resize(gt_img, args.resolution)

                mk_np = np.array(mk_img, dtype=np.float32)[..., np.newaxis]
                gt_np = np.array(gt_img, dtype=np.float32)
                ot_np = np.array(ot_img, dtype=np.float32)
                
                if args.use_gt_bg:
                    ot_np = ot_np * mk_np + gt_np * (1 - mk_np)

                mse_score = mse(ot_np, gt_np)
                psnr_score = psnr(gt_np, ot_np, data_range=255)

                mse_total += mse_score
                psnr_total += psnr_score

                w, h = ot_img.size
                fscore = mse(ot_np * mk_np, gt_np * mk_np) * (h * w) / (mk_np.sum())
                fmse_total += fscore

                bscore = (
                    mse(ot_np * (1 - mk_np), gt_np * (1 - mk_np))
                    * (h * w)
                    / ((1 - mk_np).sum())
                )
                bmse_total += bscore

                output_file.write(
                    f"""
        num : {ct}
        filename : {ot_file}
        image size : {ot_img.size}
        psnr : {psnr_score:.3f}
        mse : {mse_score:.3f}
        fmse : {fscore:.3f}
        bmse : {bscore:.3f}
        time : {time.time()-st:.3f}
        \n"""
                )
                if html and fscore > 1000:
                    # if html:
                    gt_save_name = (
                        output_dir + "/" + "imgs/" + "gt_" + target_path.split("/")[-1]
                    )
                    comp_save_name = (
                        output_dir + "/" + "imgs/" + "comp_" + comp_path.split("/")[-1]
                    )
                    ot_save_name = (
                        output_dir + "/" + "imgs/" + "out_" + ot_file.split("/")[-1]
                    )
                    mk_save_name = (
                        output_dir + "/" + "imgs/" + "mask_" + mask_path.split("/")[-1]
                    )

                    gt_img.save(gt_save_name)

                    comp_img = Image.open(comp_path).convert("RGB")
                    comp_img = possible_resize(comp_img, args.resolution)
                    comp_img.save(comp_save_name)

                    ot_img.save(ot_save_name)
                    mk_img.save(
                        output_dir + "/" + "imgs/" + "mask_" + mask_path.split("/")[-1]
                    )
                    htmlstr += (
                        '<img src="'
                        + "imgs/"
                        + "gt_"
                        + target_path.split("/")[-1]
                        + '" style="width: 15%;" />'
                    )
                    htmlstr += (
                        '<img src="'
                        + "imgs/"
                        + "comp_"
                        + comp_path.split("/")[-1]
                        + '" style="width: 15%;" />'
                    )
                    htmlstr += (
                        '<img src="'
                        + "imgs/"
                        + "out_"
                        + ot_file.split("/")[-1]
                        + '" style="width: 15%;" />'
                    )
                    htmlstr += (
                        '<img src="'
                        + "imgs/"
                        + "mask_"
                        + mask_path.split("/")[-1]
                        + '" style="width: 15%;" />'
                    )

                    htmlstr += (
                        '<div style="width: 10%">mse:' + "%.2f" % mse_score + "</div>"
                    )
                    htmlstr += (
                        '<div style="width: 10%">fmse:' + "%.2f" % fscore + "</div>"
                    )
                    htmlstr += (
                        '<div style="width: 10%">bmse:' + "%.2f" % bscore + "</div>"
                    )

                ds2psnr[dataset].append(psnr_score)
                ds2mse[dataset].append(mse_score)
                ds2fmse[dataset].append(fscore)
                ds2bmse[dataset].append(bscore)
                ds2mask[dataset].append(mk_np.sum() / (h * w))
                ct += 1
                if ct == ct_max:
                    break
        except Exception as e:
            print(e)

    htmlstr += "</div></body></html>"
    if html:
        with open(output_dir + "/" + "test.html", "w") as fw:
            fw.write(htmlstr)

    mask_area_range_list = [
        (0.0, 0.05),
        (0.05, 0.15),
        (0.15, 1.0),
    ]
    for mask_arae_range in mask_area_range_list:
        this_range_total_psnr = []
        this_range_total_mse = []
        this_range_total_fmse = []
        this_range_total_bmse = []
        for ds in ds2psnr:
            per_ds_total_psnr = []
            per_ds_total_mse = []
            per_ds_total_fmse = []
            per_ds_total_bmse = []
            for i in range(len(ds2mask[ds])):
                if mask_arae_range[0] <= ds2mask[ds][i] < mask_arae_range[1]:
                    per_ds_total_psnr.append(ds2psnr[ds][i])
                    per_ds_total_mse.append(ds2mse[ds][i])
                    per_ds_total_fmse.append(ds2fmse[ds][i])
                    per_ds_total_bmse.append(ds2bmse[ds][i])

                    this_range_total_psnr.append(ds2psnr[ds][i])
                    this_range_total_mse.append(ds2mse[ds][i])
                    this_range_total_fmse.append(ds2fmse[ds][i])
                    this_range_total_bmse.append(ds2bmse[ds][i])

            output_file.write(
                f"{mask_arae_range} [PSRN] {ds} : {np.sum(per_ds_total_psnr):.3f} / {len(per_ds_total_psnr)} = {np.mean(per_ds_total_psnr):.3f}\n"
            )
            output_file.write(
                f"{mask_arae_range} [MSE] {ds} : {np.sum(per_ds_total_mse):.3f} / {len(per_ds_total_mse)} = {np.mean(per_ds_total_mse):.3f}\n"
            )
            output_file.write(
                f"{mask_arae_range} [FMSE] {ds} : {np.sum(per_ds_total_fmse):.3f} / {len(per_ds_total_fmse)} = {np.mean(per_ds_total_fmse):.3f}\n"
            )
            output_file.write(
                f"{mask_arae_range} [BMSE] {ds} : {np.sum(per_ds_total_bmse):.3f} / {len(per_ds_total_bmse)} = {np.mean(per_ds_total_bmse):.3f}\n"
            )
            output_file.write("=" * 30 + "\n")

        output_file.write(
            f"{mask_arae_range} [PSRN] in total : {np.sum(this_range_total_psnr):.3f} / {len(this_range_total_psnr)} = {np.mean(this_range_total_psnr):.3f}\n"
        )
        output_file.write(
            f"{mask_arae_range} [MSE] in total : {np.sum(this_range_total_mse):.3f} / {len(this_range_total_mse)} = {np.mean(this_range_total_mse):.3f}\n"
        )
        output_file.write(
            f"{mask_arae_range} [FMSE] in total : {np.sum(this_range_total_fmse):.3f} / {len(this_range_total_fmse)} = {np.mean(this_range_total_fmse):.3f}\n"
        )
        output_file.write(
            f"{mask_arae_range} [BMSE] in total : {np.sum(this_range_total_bmse):.3f} / {len(this_range_total_bmse)} = {np.mean(this_range_total_bmse):.3f}\n"
        )
        output_file.write("=" * 30 + "\n")

    # 打印每个数据集的平均PSNR和MSE
    for ds in ds2psnr:
        output_file.write(
            f"[PSRN] of {ds} : {np.sum(ds2psnr[ds]):.3f} / {len(ds2psnr[ds])} = {np.mean(ds2psnr[ds]):.3f}\n"
        )
        output_file.write(
            f"[MSE] of {ds} : {np.sum(ds2mse[ds]):.3f} / {len(ds2mse[ds])} = {np.mean(ds2mse[ds]):.3f}\n"
        )
        output_file.write(
            f"[FMSE] of {ds} : {np.sum(ds2fmse[ds]):.3f} / {len(ds2fmse[ds])} = {np.mean(ds2fmse[ds]):.3f}\n"
        )
        output_file.write(
            f"[BMSE] of {ds} : {np.sum(ds2bmse[ds]):.3f} / {len(ds2bmse[ds])} = {np.mean(ds2bmse[ds]):.3f}\n"
        )
        output_file.write("=" * 30 + "\n")

    # 打印总体平均PSNR，MSE和FMSE
    output_file.write(
        f"""metric in total: 
total smaples {ct}
psnr : {psnr_total:.3f} / {ct} = {psnr_total / ct:.3f}
mse : {mse_total:.3f} / {ct} = {mse_total / ct:.3f}
fmse : {fmse_total:.3f} / {ct} = {fmse_total / ct:.3f}
bmse : {bmse_total:.3f} / {ct} = {bmse_total / ct:.3f}
\n"""
    )
