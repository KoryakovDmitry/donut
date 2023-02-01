import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

FONT_SCALE = 20 / 4080
FONT_PATH = "donut/docs/fonts/french.ttf"


def bbox2pts(bbox):
    return [
        [bbox[0], bbox[1]],
        [bbox[2], bbox[1]],
        [bbox[2], bbox[3]],
        [bbox[0], bbox[3]],
    ]


def plot_bboxes(img_pil, res, plot_cof=True):
    font = int(np.round(FONT_SCALE * np.hypot(img_pil.size[0], img_pil.size[1])))
    pred_img_pil = Image.new(
        "RGB", (img_pil.size[0], img_pil.size[1]), (255, 255, 255)
    )
    draw = ImageDraw.Draw(img_pil)
    draw_pred = ImageDraw.Draw(pred_img_pil)

    for top_key, top_val in res.items():
        if isinstance(top_val, list):
            for n_line, item in enumerate(top_val):
                for item_key, item_val in item.items():

                    bbox = item_val[2]
                    pts = bbox2pts(bbox)
                    pts = list(map(tuple, pts))

                    text = item_val[0]

                    if plot_cof:
                        conf = np.round(float(item_val[1]), 2) * 100
                        conf = str(int(conf))
                        draw.text(
                            (pts[0][0], pts[0][1] + 10),
                            conf,
                            font=ImageFont.truetype(FONT_PATH, font),
                            # fill=self.colors[lbl],
                            fill=(0, 0, 0, 0),
                        )

                    draw.polygon(pts, outline=(255, 0, 0, 0))
                    draw_pred.polygon(pts, outline=(255, 0, 0, 0), )

                    draw_pred.text(
                        (pts[0][0], pts[0][1]),
                        text,
                        font=ImageFont.truetype(FONT_PATH, font),
                        # fill=self.colors[lbl],
                        fill=(0, 0, 0, 0),
                    )

                    draw_pred.text(
                        (pts[3][0], pts[3][1]),
                        f"{top_key}.{n_line}.{item_key}",
                        font=ImageFont.truetype(FONT_PATH, font),
                        # fill=self.colors[lbl],
                        fill=(0, 0, 0, 0),
                    )
        elif isinstance(top_val, dict):
            for sub_key, sub_val in top_val.items():
                bbox = sub_val[2]
                pts = bbox2pts(bbox)
                pts = list(map(tuple, pts))

                text = sub_val[0]

                if plot_cof:
                    conf = np.round(float(sub_val[1]), 2) * 100
                    conf = str(int(conf))
                    draw.text(
                        (pts[0][0], pts[0][1] + 10),
                        conf,
                        font=ImageFont.truetype(FONT_PATH, font),
                        # fill=self.colors[lbl],
                        fill=(0, 0, 0, 0),
                    )

                draw.polygon(pts, outline=(255, 0, 0, 0))
                draw_pred.polygon(pts, outline=(255, 0, 0, 0), )

                draw_pred.text(
                    (pts[0][0], pts[0][1]),
                    text,
                    font=ImageFont.truetype(FONT_PATH, font),
                    # fill=self.colors[lbl],
                    fill=(0, 0, 0, 0),
                )

                draw_pred.text(
                    (pts[3][0], pts[3][1]),
                    f"{top_key}.{sub_key}",
                    font=ImageFont.truetype(FONT_PATH, font),
                    # fill=self.colors[lbl],
                    fill=(0, 0, 0, 0),
                )
        else:
            bbox = top_val[2]
            pts = bbox2pts(bbox)
            pts = list(map(tuple, pts))

            text = top_val[0]

            if plot_cof:
                conf = np.round(float(top_val[1]), 2) * 100
                conf = str(int(conf))
                draw.text(
                    (pts[0][0], pts[0][1] + 10),
                    conf,
                    font=ImageFont.truetype(FONT_PATH, font),
                    # fill=self.colors[lbl],
                    fill=(0, 0, 0, 0),
                )

            draw.polygon(pts, outline=(255, 0, 0, 0))
            draw_pred.polygon(pts, outline=(255, 0, 0, 0), )

            draw_pred.text(
                (pts[0][0], pts[0][1]),
                text,
                font=ImageFont.truetype(FONT_PATH, font),
                # fill=self.colors[lbl],
                fill=(0, 0, 0, 0),
            )

            draw_pred.text(
                (pts[3][0], pts[3][1]),
                f"{top_key}",
                font=ImageFont.truetype(FONT_PATH, font),
                # fill=self.colors[lbl],
                fill=(0, 0, 0, 0),
            )

    res_img = hconcat_resize_min([img_pil, pred_img_pil], mode="pil")
    return res_img


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC, mode="pil"):
    assert mode in ("pil", "cv2"), 'use "pil" or "cv2"'

    if mode == "cv2":
        h_min = min(im.shape[0] for im in im_list)
        im_list_resize = [
            cv2.resize(
                im,
                (int(im.shape[1] * h_min / im.shape[0]), h_min),
                interpolation=interpolation,
            )
            for im in im_list
        ]
        return Image.fromarray(cv2.hconcat(im_list_resize)[:, :, ::-1])
    else:
        h_min = min(im.size[1] for im in im_list)
        im_list_resize = [
            im.resize(
                (int(im.size[0] * h_min / im.size[1]), h_min),
            )
            for im in im_list
        ]
        res_img = Image.new(
            "RGB",
            (len(im_list_resize) * im_list_resize[0].size[0], h_min),
            (255, 255, 255),
        )
        for num, im in enumerate(im_list_resize):
            res_img.paste(im, (num * im_list_resize[0].size[0], 0))

        return res_img
