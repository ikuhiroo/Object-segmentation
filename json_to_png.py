import argparse
import json
import os
import os.path as osp
import io
import base64
import warnings
from pathlib import Path
from collections import defaultdict


import numpy as np
import PIL.Image
import PIL.ImageDraw
import yaml

"""
作成したjsonファイルからsegmentationされたpngファイルを作成する
jsonフォルダ内にツールを用いて作成したjsonファイルを置く
pngファイルを置くsegmentファイルを置く

257行目で新しい画像に対するクラスラベルを設定する
"""
JSON_DIR = Path("/Users/1-10robotics/Desktop/json")
OUT_DIR = Path("/Users/1-10robotics/Desktop/segment")

def label_colormap(N=256):

    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3))
    for i in range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap


def labelcolormap(N=256):
    warnings.warn('labelcolormap is deprecated. Please use label_colormap.')
    return label_colormap(N=N)


# similar function as skimage.color.label2rgb
def label2rgb(lbl, img=None, n_labels=None, alpha=0.3, thresh_suppress=0):
    if n_labels is None:
        n_labels = len(np.unique(lbl))

    cmap = label_colormap(n_labels)
    cmap = (cmap * 255).astype(np.uint8)

    lbl_viz = cmap[lbl]
    lbl_viz[lbl == -1] = (0, 0, 0)  # unlabeled

    if img is not None:
        img_gray = PIL.Image.fromarray(img).convert('LA')
        img_gray = np.asarray(img_gray.convert('RGB'))
        # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        lbl_viz = alpha * lbl_viz + (1 - alpha) * img_gray
        lbl_viz = lbl_viz.astype(np.uint8)

    return lbl_viz


def img_b64_to_array(img_b64):
    warnings.warn('img_ba64_to_array is deprecated. '
                  'Please use img_b64_to_arr.')
    return img_b64_to_arr(img_b64)


def img_b64_to_arr(img_b64):
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_arr = np.array(PIL.Image.open(f))
    return img_arr


def img_arr_to_b64(img_arr):
    img_pil = PIL.Image.fromarray(img_arr)
    f = io.BytesIO()
    img_pil.save(f, format='PNG')
    img_bin = f.getvalue()
    img_b64 = base64.encodestring(img_bin)
    return img_b64


def polygons_to_mask(img_shape, polygons):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


def draw_label(label, img=None, label_names=None, colormap=None):
    """
    label : ラベル
    凡例を表示させない
    """
    import matplotlib.pyplot as plt

    backend_org = plt.rcParams['backend']
    plt.switch_backend('agg')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0,
                        wspace=0, hspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    # label_names {0, 1}
    if label_names is None:
        label_names = [str(l) for l in range(label.max() + 1)]

    # 256で割った値
    if colormap is None:
        colormap = label_colormap(len(label_names))

    label_viz = label2rgb(label, img, n_labels=len(label_names))
    plt.imshow(label_viz)
    plt.axis('off')

    plt_handlers = []
    plt_titles = []
    for label_value, label_name in enumerate(label_names):
        if label_value not in label:
            continue
        if label_name.startswith('_'):
            continue
        fc = colormap[label_value]
        p = plt.Rectangle((0, 0), 1, 1, fc=fc)
        plt_handlers.append(p)
        plt_titles.append(label_name)
    # plt.legend(plt_handlers, plt_titles, loc='lower right', framealpha=.5)

    f = io.BytesIO()
    plt.savefig(f, bbox_inches='tight', pad_inches=0)
    plt.cla()
    plt.close()

    plt.switch_backend(backend_org)

    out_size = (label_viz.shape[1], label_viz.shape[0])
    out = PIL.Image.open(f).resize(out_size, PIL.Image.BILINEAR).convert('RGB')
    out = np.asarray(out)
    return out


def shapes_to_label(img_shape, shapes, label_name_to_value, type='class'):
    assert type in ['class', 'instance']

    cls = np.zeros(img_shape[:2], dtype=np.int32)
    if type == 'instance':
        ins = np.zeros(img_shape[:2], dtype=np.int32)
        instance_names = ['_background_']
    for shape in shapes:
        polygons = shape['points']
        label = shape['label']
        if type == 'class':
            cls_name = label
        elif type == 'instance':
            cls_name = label.split('-')[0]
            if label not in instance_names:
                instance_names.append(label)
            ins_id = len(instance_names) - 1
        cls_id = label_name_to_value[cls_name]
        mask = polygons_to_mask(img_shape[:2], polygons)
        cls[mask] = cls_id
        if type == 'instance':
            ins[mask] = ins_id

    if type == 'instance':
        return cls, ins
    return cls


def labelme_shapes_to_label(img_shape, shapes):
    warnings.warn('labelme_shapes_to_label is deprecated, so please use '
                  'shapes_to_label.')
    # 背景→ 0 (0, 0, 0), エッジ→ 255 (255, 255, 255)，クラス→1~22
    label_name_to_value = {'_background_': 0, '_edge_': 255}
    for shape in shapes:
        label_name = shape['label']
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value

    lbl = shapes_to_label(img_shape, shapes, label_name_to_value)
    return lbl, label_name_to_value

def main():
    try:
        OUTPUT_DIR.mkdir(parents=True)
    except:
        pass

    color_value = defaultdict(lambda: 0)
    json_file_list = list(JSON_DIR.glob("*.json"))
    for i in range(len(json_file_list)):
        # jsonファイルをロード
        json_file = str(json_file_list[i])
        print(json_file)
        data = json.load(open(json_file))
        if data['imageData']:
            imageData = data['imageData']
        else:
            imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
            with open(imagePath, 'rb') as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode('utf-8')

        # 画像ファイルをRGBの配列に変換
        # <class 'numpy.ndarray'>
        # ex.(673, 576, 3)
        img = img_b64_to_arr(imageData)
        # print(img)

        # 背景→ 0 (0, 0, 0), エッジ→ 255 (255, 255, 255)，クラス→1~22
        # label_name_to_value = {'_background_': 0, '_edge_': 255}
        label_name_to_value = {'_background_': 0}
        for shape in data['shapes']:
            label_name = shape['label']
            # _background_が含まれている場合，
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                # それ以外のラベルの場合，辞書に追加する
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value
        # print(label_name_to_value)

        # label_values must be dense
        label_values, label_names = [], []
        for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
            # ln : 繰り返し回数(init : 0)
            # lv : ラベル名
            label_values.append(lv)
            label_names.append(ln)
        assert label_values == list(range(len(label_values)))

        # img.shape : (673, 576, 3), lbl.shape : (673, 576)
        lbl = shapes_to_label(img.shape, data['shapes'], label_name_to_value)

        # 21をpepperクラスに割り当てる
        lbl = np.where((lbl != 0) & (lbl != 255), 21, lbl).astype('uint8')
        print(lbl.shape)
        print(lbl.dtype)

        # ファイルの保存
        PIL.Image.fromarray(lbl).save(osp.join(OUT_DIR, '{}.png'.format(osp.basename(json_file).split(".")[0])))
        for p in range(len(lbl)):
            for q in range(len(lbl[p])):
                    color_value[int(lbl[p][q])] += 1
        print(color_value)
    print('Saved to: %s' % OUT_DIR)

if __name__ == '__main__':
    main()
