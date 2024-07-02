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
from typing import List
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--data_dir", type=str, default="./data/iHarmony4")
parser.add_argument("--json_file_path", type=str, default="DataSpecs/all_test.jsonl")
parser.add_argument("--resolution", type=int, default=256)
parser.add_argument("--num_processes", type=int, default=4)
parser.add_argument("--use_gt_bg", action="store_true")
args = parser.parse_args()

Path(args.output_dir).mkdir(parents=True, exist_ok=True)
(Path(args.output_dir) / "imgs").mkdir(parents=True, exist_ok=True)

# 初始化变量
count = 0
psnr_total = 0
mse_total = 0
fmse_total = 0
bmse_total = 0
ds2psnr = defaultdict(list)
ds2mse = defaultdict(list)
ds2fmse = defaultdict(list)
ds2bmse = defaultdict(list)
ds2mask = defaultdict(list)


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


def possible_resize(image: Image.Image, resolution: int):
    if image.size != (resolution, resolution):
        image = TF.resize(image, [resolution, resolution], antialias=True)
    return image


dataset_map = {
    "a": "HAdobe5k",
    "c": "HCOCO",
    "d": "Hday2night",
    "f": "HFlickr",
    "s": "SAM",
    "g": "Gen",
}


def process_fn(comp_path, lock):

    global count
    global psnr_total
    global mse_total
    global fmse_total
    global bmse_total
    global ds2psnr
    global ds2mse
    global ds2fmse
    global ds2bmse
    global ds2mask

    _output_str = ""
    _htmlstr = ""

    all_paths = comp_to_paths(comp_path)
    target_path = all_paths["real"]
    mask_path = all_paths["mask"]
    comp_path = all_paths["comp"]

    harm_path = os.path.join(
        args.input_dir, comp_path.split("/")[-1].replace(".jpg", ".png")
    )
    harm_name = harm_path.split("/")[-1]

    ot_file = harm_path

    dataset = dataset_map[harm_name[0]]

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

        with lock:
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
        with lock:
            bmse_total += bscore

        _output_str = f"""
filename : {ot_file}
image size : {ot_img.size}
psnr : {psnr_score:.3f}
mse : {mse_score:.3f}
fmse : {fscore:.3f}
bmse : {bscore:.3f}
\n"""
        if fscore > 1000:
            gt_save_name = (
                args.output_dir + "/" + "imgs/" + "gt_" + target_path.split("/")[-1]
            )
            comp_save_name = (
                args.output_dir + "/" + "imgs/" + "comp_" + comp_path.split("/")[-1]
            )
            ot_save_name = (
                args.output_dir + "/" + "imgs/" + "out_" + ot_file.split("/")[-1]
            )
            mk_save_name = (
                args.output_dir + "/" + "imgs/" + "mask_" + mask_path.split("/")[-1]
            )

            gt_img.save(gt_save_name)

            comp_img = Image.open(comp_path).convert("RGB")
            comp_img = possible_resize(comp_img, args.resolution)
            comp_img.save(comp_save_name)

            ot_img.save(ot_save_name)
            mk_img.save(
                args.output_dir + "/" + "imgs/" + "mask_" + mask_path.split("/")[-1]
            )
            _htmlstr = (
                '<img src="'
                + "imgs/"
                + "gt_"
                + target_path.split("/")[-1]
                + '" style="width: 15%;" />'
            )
            _htmlstr += (
                '<img src="'
                + "imgs/"
                + "comp_"
                + comp_path.split("/")[-1]
                + '" style="width: 15%;" />'
            )
            _htmlstr += (
                '<img src="'
                + "imgs/"
                + "out_"
                + ot_file.split("/")[-1]
                + '" style="width: 15%;" />'
            )
            _htmlstr += (
                '<img src="'
                + "imgs/"
                + "mask_"
                + mask_path.split("/")[-1]
                + '" style="width: 15%;" />'
            )

            _htmlstr += '<div style="width: 10%">mse:' + "%.2f" % mse_score + "</div>"
            _htmlstr += '<div style="width: 10%">fmse:' + "%.2f" % fscore + "</div>"
            _htmlstr += '<div style="width: 10%">bmse:' + "%.2f" % bscore + "</div>"

        with lock:
            ds2psnr[dataset].append(psnr_score)
            ds2mse[dataset].append(mse_score)
            ds2fmse[dataset].append(fscore)
            ds2bmse[dataset].append(bscore)
            ds2mask[dataset].append(mk_np.sum() / (h * w))
            count = count + 1
    else:
        _output_str = ""
        _htmlstr = ""
    return _output_str, _htmlstr


if __name__ == "__main__":
    json_file = read_jsonl_to_list(os.path.join(args.data_dir, args.json_file_path))
    lock = Lock()

    output_str = ""
    htmlstr = (
        '<html><head>harmony</head><body><div style="display: flex; flex-wrap: wrap;">'
    )
    with ThreadPoolExecutor(max_workers=args.num_processes) as executor:
        futures = [
            executor.submit(
                process_fn, pjoin(args.data_dir, json_data["file_name"]), lock
            )
            for json_data in json_file
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            _output_str, _htmlstr = future.result()
            output_str += _output_str
            htmlstr += _htmlstr
            
    print(count)

    htmlstr += "</div></body></html>"
    with open(args.output_dir + "/" + "test.html", "w") as fw:
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

            output_str += f"{mask_arae_range} [PSRN] {ds} : {np.sum(per_ds_total_psnr):.3f} / {len(per_ds_total_psnr)} = {np.mean(per_ds_total_psnr):.3f}\n"
            output_str += f"{mask_arae_range} [MSE] {ds} : {np.sum(per_ds_total_mse):.3f} / {len(per_ds_total_mse)} = {np.mean(per_ds_total_mse):.3f}\n"
            output_str += f"{mask_arae_range} [FMSE] {ds} : {np.sum(per_ds_total_fmse):.3f} / {len(per_ds_total_fmse)} = {np.mean(per_ds_total_fmse):.3f}\n"
            output_str += f"{mask_arae_range} [BMSE] {ds} : {np.sum(per_ds_total_bmse):.3f} / {len(per_ds_total_bmse)} = {np.mean(per_ds_total_bmse):.3f}\n"
            output_str += "=" * 30 + "\n"

        output_str += f"{mask_arae_range} [PSRN] in total : {np.sum(this_range_total_psnr):.3f} / {len(this_range_total_psnr)} = {np.mean(this_range_total_psnr):.3f}\n"
        output_str += f"{mask_arae_range} [MSE] in total : {np.sum(this_range_total_mse):.3f} / {len(this_range_total_mse)} = {np.mean(this_range_total_mse):.3f}\n"
        output_str += f"{mask_arae_range} [FMSE] in total : {np.sum(this_range_total_fmse):.3f} / {len(this_range_total_fmse)} = {np.mean(this_range_total_fmse):.3f}\n"
        output_str += f"{mask_arae_range} [BMSE] in total : {np.sum(this_range_total_bmse):.3f} / {len(this_range_total_bmse)} = {np.mean(this_range_total_bmse):.3f}\n"
        output_str += "=" * 30 + "\n"

    # 打印每个数据集的平均PSNR和MSE
    for ds in ds2psnr:
        output_str += f"[PSRN] of {ds} : {np.sum(ds2psnr[ds]):.3f} / {len(ds2psnr[ds])} = {np.mean(ds2psnr[ds]):.3f}\n"
        output_str += f"[MSE] of {ds} : {np.sum(ds2mse[ds]):.3f} / {len(ds2mse[ds])} = {np.mean(ds2mse[ds]):.3f}\n"
        output_str += f"[FMSE] of {ds} : {np.sum(ds2fmse[ds]):.3f} / {len(ds2fmse[ds])} = {np.mean(ds2fmse[ds]):.3f}\n"
        output_str += f"[BMSE] of {ds} : {np.sum(ds2bmse[ds]):.3f} / {len(ds2bmse[ds])} = {np.mean(ds2bmse[ds]):.3f}\n"
        output_str += "=" * 30 + "\n"

    # 打印总体平均PSNR，MSE和FMSE
    output_str += f"""metric in total: 
total smaples {count}
psnr : {psnr_total:.3f} / {count} = {psnr_total / count:.3f}
mse : {mse_total:.3f} / {count} = {mse_total / count:.3f}
fmse : {fmse_total:.3f} / {count} = {fmse_total / count:.3f}
bmse : {bmse_total:.3f} / {count} = {bmse_total / count:.3f}
\n"""

    with open(pjoin(args.output_dir, "output.log"), "w") as fw:
        fw.write(output_str)
