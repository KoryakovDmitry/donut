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

imgs_sample = glob("dataset/dataset_tower_company_v1/test/5.jpeg")
# imgs_sample = np.random.choice(imgs, 3, replace=False)


pretrained_model = DonutModel.from_pretrained("config/tower_company")
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
    output = pretrained_model.inference(image=input_img,
                                        prompt="<s_tower_company_v1>",
                                        return_attentions=True,
                                        return_confs=True,
                                        return_tokens=True,
                                        return_max_bbox=True,
                                        return_plot=False)
    output["pages"] = output["predictions"]
    del output["predictions"]

    print(json.dumps(output, ensure_ascii=False, indent=2))
    break
