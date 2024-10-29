import math
import struct
import random
import cv2

import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mask_util
from torch.distributions.multivariate_normal import MultivariateNormal
import torchvision.transforms as transforms


def write_log(p: str, content: str, mode='a', is_print=True):
    assert mode in ['a', 'w']
    if is_print:
        print(content, end='')

    with open(p, mode) as f:
        f.write(content)


# def load_img(p, padding=True, factor=64):
#     x = Image.open(p)
#     x = np.array(x)
#     if len(x.shape) == 2:
#         x = np.expand_dims(x, axis=2)
#         x = np.repeat(x, 3, axis=2)
#     x = transforms.ToTensor()(x)
#     x = x.unsqueeze(0).float()
#     h, w = x.shape[2:4]
#     if padding:
#         dh = factor * math.ceil(h / factor) - h
#         dw = factor * math.ceil(w / factor) - w
#         x = F.pad(x, (0, dw, 0, dh))
#     return x, h, w

def load_img(p, padding=True, factor=64):
    x = Image.open(p)
    x = torch.from_numpy(np.asarray(x))
    if len(x.shape) == 2:
        x = x.unsqueeze(-1).repeat(1,1,3)    # h,w -> h,w,3
    x = x.permute(2, 0, 1).unsqueeze(0).float().div(255)
    h, w = x.shape[2:4]

    if padding:
        dh = factor * math.ceil(h / factor) - h
        dw = factor * math.ceil(w / factor) - w
        x = F.pad(x, (0, dw, 0, dh))
    return x, h, w


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def quality2lambda(qmap, low_lmbda, high_lmbda):
    return 2 ** (low_lmbda + (high_lmbda - low_lmbda) * qmap)

def generate_random_qmap(patch_size):
    sample = random.random()
    if sample < 0.01:
        qmap = torch.zeros(1, patch_size, patch_size)
    elif sample < 0.02:
        qmap = torch.ones(1, patch_size, patch_size)
    elif sample < 0.25:
        qmap = torch.ones(1, patch_size, patch_size) * random.random()
    elif sample < 0.5:
        v1 = random.random()
        v2 = random.random()
        qmap = torch.linspace(v1, v2, patch_size).repeat(patch_size, 1)
        if random.random() < 0.5:
            qmap = qmap.T
        qmap = qmap.view(1, patch_size, patch_size)
    elif sample < 0.75:
        scale = random.randint(1, patch_size // 8)
        qmap = torch.rand(patch_size // scale, patch_size // scale)
        qmap = qmap.unsqueeze(0).unsqueeze(0).repeat(1, scale**2, 1, 1)
        qmap = F.pixel_shuffle(qmap, scale).squeeze()
        qmap = F.pad(qmap, (patch_size % scale, 0, patch_size % scale, 0), 'constant', random.random())
        if random.random() < 0.5:
            qmap = torch.flip(qmap, dims=[0])
        if random.random() < 0.5:
            qmap = torch.flip(qmap, dims=[1])
        qmap = qmap.view(1, patch_size, patch_size)
    else:
        qmap = torch.zeros(1, patch_size, patch_size)
        gaussian_num = int(1 + random.random() * 20)
        grid = qmap_get_grid((patch_size, patch_size))
        for i in range(gaussian_num):
            mu_x = patch_size * random.random()
            mu_y = patch_size * random.random()
            var_x = 2000 * random.random() + 1000
            var_y = 2000 * random.random() + 1000

            m = MultivariateNormal(torch.tensor([mu_x, mu_y]), torch.tensor([[var_x, 0], [0, var_y]]))
            p = m.log_prob(grid)
            kernel = torch.exp(p)
            qmap += kernel
        qmap *= 1 / qmap.max() * (0.5 * random.random() + 0.5)
    return qmap

def qmap_get_grid(size):
    x1 = torch.tensor(range(size[0]))
    x2 = torch.tensor(range(size[1]))
    grid_x1, grid_x2 = torch.meshgrid(x1, x2)

    grid1 = grid_x1.view(size[0], size[1], 1)
    grid2 = grid_x2.view(size[0], size[1], 1)
    grid = torch.cat([grid1, grid2], dim=-1)
    return grid

def generate_local_region_msk(b, h, w, size_local_region, patch_size):
    h_s, w_s = int(h / size_local_region), int(w /size_local_region)
    msk = torch.tensor(range(h_s*w_s)).reshape(h_s, w_s).unsqueeze(0).unsqueeze(0).float()
    h_fea, w_fea = int(h / patch_size), int(w / patch_size)
    msk = F.interpolate(msk, size=(h_fea, w_fea), mode='nearest')
    return msk.repeat(b, 1, 1, 1).int()

# def generate_random_group_msk(b, h, w, patch_size, grid_size=16, factor=0.96, anchor_h=[2,4], anchor_w=[2,4]):
def generate_random_group_msk(b, h, w, patch_size, grid_size=32, factor=0.9, anchor_h=[2,3], anchor_w=[2,3]):

    hg, wg = int(h / grid_size), int(w /grid_size)
    msk = torch.rand((b, 1, hg, wg))
    msk[msk<factor] = 0.
    msk[msk>=factor] = 1.

    idxs = torch.argwhere(msk==1.)
    for idx in idxs:
        b_,c_,bbox_x1,bbox_y1 = idx
        bbox_h, bbox_w = random.random() * (anchor_h[1]-anchor_h[0]), random.random() * (anchor_w[1]-anchor_w[0])
        bbox_h = torch.round(torch.tensor(bbox_h)) + anchor_h[0]
        bbox_w = torch.round(torch.tensor(bbox_w)) + anchor_w[0]
        bbox_x2 = (bbox_x1+bbox_h).clamp(0, h).int()
        bbox_y2 = (bbox_y1+bbox_w).clamp(0, w).int()

        msk[b_,c_,bbox_x1:bbox_x2, bbox_y1:bbox_y2] = 1

    msk = msk.permute(0,2,3,1)
    msk = msk.numpy().astype(np.uint8)*255

    for i, msk_tmp in enumerate(msk):
        num_labels, msk_tmp, stats, centroids = cv2.connectedComponentsWithStats(
            msk_tmp, connectivity=4) 
        msk[i, :, :, 0] = msk_tmp
    msk = torch.from_numpy(msk).permute(0,3,1,2)

    hg, wg = int(h / patch_size), int(w /patch_size)
    msk = F.interpolate(msk, size=(hg, wg))

    return msk.int()

def generate_edgemsk_fgmsk(msk):
    # edge_msk generation
    edge_msk = msk.clone()
    edge_msk = F.interpolate(edge_msk.float(), scale_factor=8).int()
    edge_msk = edge_msk.permute(0,2,3,1)
    edge_msk = edge_msk.cpu().numpy().astype(np.uint8)*255
    for i, edge_msk_tmp in enumerate(edge_msk):
        edges_tmp = cv2.Canny(edge_msk_tmp, 50, 100)
        edge_msk[i, :, :, 0] = edges_tmp
    edge_msk = torch.from_numpy(edge_msk).permute(0,3,1,2).float() / 255

    # fg_msk generation
    fg_msk = msk.clone().float()
    fg_msk[fg_msk!=0] = 1

    # import torchvision
    # torchvision.utils.save_image(edge_msk.cpu(), 'edges.png')
    # torchvision.utils.save_image(fg_msk.cpu(), 'fgs.png')
    return edge_msk, fg_msk


# --- actual encoding and decoding utils ---

def pack_string(string):
    byte_stream = struct.pack(f'>I', len(string))
    byte_stream += struct.pack(f'>{len(string)}s', string)
    return byte_stream

def unpack_string(byte_stream):
    s1 = struct.calcsize('I')
    s2 = struct.calcsize('s')
    length = struct.unpack(f'>I', byte_stream[:s1])[0]
    string = struct.unpack(f'>{length}s', byte_stream[s1:s1+s2*length])[0]
    return string, byte_stream[s1+s2*length:]

def pack_strings(strings):
    byte_stream = b''
    for string in strings:
        byte_stream += pack_string(string)
    return byte_stream

def unpack_strings(byte_stream, n):
    strings = []
    for i in range(n):
        string, byte_stream = unpack_string(byte_stream)
        strings.append(string)
    return strings, byte_stream

def pack_uints(uints):
    byte_stream = struct.pack(f'>{len(uints)}I', *uints)
    return byte_stream

def unpack_uints(byte_stream, n):
    s1 = struct.calcsize('I')
    uints = struct.unpack(f'>{n}I', byte_stream[:n*s1])
    return uints, byte_stream[n*s1:]

def pack_ushorts(ushorts):
    byte_stream = struct.pack(f'>{len(ushorts)}H', *ushorts)
    return byte_stream

def unpack_ushorts(byte_stream, n):
    s1 = struct.calcsize('H')
    ushorts = struct.unpack(f'>{n}H', byte_stream[:n*s1])
    return ushorts, byte_stream[n*s1:]

def pack_bool(bool):
    byte_stream = struct.pack(f'>?', int(bool))
    return byte_stream

def unpack_bool(byte_stream):
    s1 = struct.calcsize('?')
    bool = struct.unpack(f'>?', byte_stream[:s1])
    return bool, byte_stream[s1:]

# --- coco tools --- 

def load_coco_labels():
    coco_id_name_map={1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
                    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
                    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
                    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
                    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                    40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                    44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                    51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                    56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                    61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
                    70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                    77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
                    82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
                    88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

    category_ids = [i for i in range(0, 80)]
    real_category_ids = list(coco_id_name_map.keys())
    tmp_dict = {}
    for x, y in zip(category_ids, real_category_ids):
        tmp_dict[x] = y

    return tmp_dict

def parse_instance(instance):
    keys = instance._fields.keys()
    info_dict = {}
    # x1, y1, x2, y2 (x means in W, y means in H)
    info_dict['pred_boxes'] = instance._fields['pred_boxes'][0].tensor.cpu().numpy()[0].tolist()
    info_dict['scores'] = instance._fields['scores'].cpu().item()
    info_dict['pred_class'] = instance._fields['pred_classes'].cpu().item()

    has_mask = instance.has("pred_masks")
    has_keypoints = instance.has("pred_keypoints")

    if has_mask:
        masks = instance._fields['pred_masks'].cpu()
        masks = masks.squeeze()
        mask = np.asfortranarray(masks).astype(np.uint8)
        segmentation = mask_util.encode(mask)
        mask_rle = {
            'counts': segmentation["counts"].decode('utf-8'),
            'size': segmentation["size"]
        }
        info_dict['segmentation'] = mask_rle

    if has_keypoints:
        # keypoints = predictions["instance"].to(torch.device("cpu")).pred_keypoints
        # keypoints[i][:, :2] -= 0.5
        # info_dict['keypoints'] = keypoints[i].flatten().tolist()
        keypoints = instance.pred_keypoints
        keypoints[:, :2] -= 0.5 # ?
        info_dict['keypoints'] = keypoints.flatten().tolist()

    return info_dict


# --- visualization tools ---
def save_fea(p_dir, fea, is_thres, thres_bottom=0, thres_up=0.2):
    if is_thres:
        fea[fea<thres_bottom]=thres_bottom
        fea[fea>thres_up]=thres_up

    def normalization(x):
        _range = x.max() - x.min()
        return (x - x.min()) / _range
    import os
    import torchvision

    upsample_2 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    upsample_4 = torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    upsample_8 = torch.nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    fea = fea.squeeze(dim=0).unsqueeze(dim=1)

    p_save_1 = os.path.join(p_dir, 'fea_1.jpg')
    p_save_2 = os.path.join(p_dir, 'fea_2.jpg')
    p_save_4 = os.path.join(p_dir, 'fea_4.jpg')
    p_save_8 = os.path.join(p_dir, 'fea_8.jpg')
    fea_1 = fea
    fea_2 = upsample_2(fea)
    fea_4 = upsample_4(fea)
    fea_8 = upsample_8(fea)
    
    # torchvision.utils.save_image(normalization(fea_8), p_save_8, nrow=16)
    # torchvision.utils.save_image(normalization(fea_4), p_save_4, nrow=16)
    # torchvision.utils.save_image(normalization(fea_2), p_save_2, nrow=16)
    torchvision.utils.save_image(normalization(fea_1), p_save_1, nrow=16)

def denormalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.tensor(mean).unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3)
    std = torch.tensor(std).unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3)
    img = img * std + mean     # unnormalize
    return img

def visualize_bitmap(x, y_log_probs, p_save):
    import os
    import matplotlib.pyplot as plt

    ## bpp
    # b,c,hx,wx = img.shape
    b,c,hx,wx = x.shape
    # fea_y_prob = nn.Upsample(scale_factor=2, mode='bilinear')(fea_y_prob)
    # bpp_y = -torch.sum(torch.log2(y_prob)).item()
    # bpp_y_spatial = -torch.sum(torch.log2(y_prob), dim = 1)
    bpp_y = -torch.sum(y_log_probs).item()
    bpp_y_spatial = -torch.sum(y_log_probs, dim = 1)
    bpp_y_spatial = bpp_y_spatial.cpu().numpy().reshape((hx // 16), (wx // 16))
    ratio = bpp_y_spatial / bpp_y
    ratio = torch.from_numpy(ratio).unsqueeze(0).unsqueeze(0)
    # ratio = nn.Upsample(scale_factor=2, mode='bilinear')(ratio)
    ratio = (ratio / 4.0).squeeze(0).squeeze(0).cpu().numpy()

    ## visualize
    plt.cla()
    fig, ax = plt.subplots()
    ax.set_axis_off()
    # im = ax.imshow(ratio * 100, cmap='RdBu')         # new. 
    # im = ax.imshow(ratio * 100, cmap='Reds')         # new. 
    # im = ax.imshow(ratio * 100)         # new. 
    # cbar = fig.colorbar(im)
    im = ax.imshow(bpp_y_spatial)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    from matplotlib import ticker, cm
    cax = divider.append_axes("right", size=0.15, pad=0.15)      ## new. setting of color bar.
    cbar = fig.colorbar(im, shrink=2, cax=cax) 

    cbar.ax.set_xlabel('%')
    # cbar.set_clim(0, 0.1)

    # ax.set_title("Spatial Bit Allocation" )  
    fig.tight_layout()
    fig.savefig(p_save)