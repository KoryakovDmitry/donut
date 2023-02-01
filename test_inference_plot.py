import os
import os.path as osp

import cv2
import json
import torch
import numpy as np
from glob import glob
from PIL import Image
from donut import DonutModel
from donut.plot_utils import plot_bboxes
from copy import deepcopy

dir2save_results = "result/"
os.makedirs(dir2save_results, exist_ok=True)

imgs_sample = glob("dataset/dataset_nano_ziffer/test/page_0_2018.06.01_Ziffer_Invoice_Linkedin_EUR_89.95.jpg")
# imgs_sample = np.random.choice(imgs, 3, replace=False)


pretrained_model = DonutModel.from_pretrained("config/ziffer")
# pretrained_model = DonutModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

if torch.cuda.is_available():
    pretrained_model.half()
    device = torch.device("cuda")
    pretrained_model.to(device)
# else:
#     pretrained_model.encoder.to(torch.bfloat16)

pretrained_model.eval()

for img_path in imgs_sample:
    fn = osp.basename(img_path)
    fn_wout_ext = osp.splitext(fn)[0]
    dir2save_results_fn = osp.join(dir2save_results, fn_wout_ext)
    os.makedirs(dir2save_results_fn, exist_ok=True)

    dir2save_results_fn_hm = osp.join(dir2save_results_fn, "hmaps")
    os.makedirs(dir2save_results_fn_hm, exist_ok=True)
    dir2save_results_fn_hm_bb = osp.join(dir2save_results_fn, "hmaps_bboxes")
    os.makedirs(dir2save_results_fn_hm_bb, exist_ok=True)

    input_img = Image.open(img_path)
    output, super_imposed_raw_heatmap_imgs, super_imposed_imgs = pretrained_model.inference(image=input_img,
                                                                                            prompt="<s_dataset_nano_ziffer>",
                                                                                            return_attentions=True,
                                                                                            return_confs=True,
                                                                                            return_tokens=True,
                                                                                            return_max_bbox=True,
                                                                                            return_plot=True)

    for key, super_imposed_raw_heatmap_img in super_imposed_raw_heatmap_imgs.items():
        path2save = osp.join(dir2save_results_fn_hm, f"{key}_{fn_wout_ext}_hmap.jpg")
        cv2.imwrite(path2save, super_imposed_raw_heatmap_img)

    for key, super_imposed_img in super_imposed_imgs.items():
        path2save = osp.join(dir2save_results_fn_hm_bb, f"{key}_{fn_wout_ext}_bbox.jpg")
        cv2.imwrite(path2save, super_imposed_img)

    for page_n, page_res in enumerate(output["pages"]):
        img_pil_plot = plot_bboxes(input_img.copy(), page_res, plot_cof=True)
        path2save = osp.join(dir2save_results_fn, f"page_{page_n}_{fn_wout_ext}.png")
        img_pil_plot.save(path2save, "PNG")

    path2save = osp.join(dir2save_results_fn, f"{fn_wout_ext}.json")
    with open(path2save, "w", encoding="utf8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    break
