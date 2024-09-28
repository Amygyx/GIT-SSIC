import torch, lmdb
import io, random, numbers
import numpy as np
from PIL import Image

import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms
from . import video_transforms

YCBCR_WEIGHTS = {
    # Spec: (K_r, K_g, K_b) with K_g = 1 - K_r - K_b
    "ITU-R_BT.709": (0.2126, 0.7152, 0.0722)
}

def rgb2ycbcr(rgb):
    r, g, b = rgb.chunk(3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5
    ycbcr = torch.cat((y, cb, cr), dim=-3)
    return ycbcr

def ycbcr2rgb(ycbcr):
    y, cb, cr = ycbcr.chunk(3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    rgb = torch.cat((r, g, b), dim=-3)
    return rgb


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        with torch.cuda.stream(self.stream):
            if isinstance(self.next_input, list):
                self.next_input = [next_input.cuda(non_blocking=True) for next_input in self.next_input]
            else:
                self.next_input = [self.next_input.cuda(non_blocking=True)]

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        self.preload()
        return input


class TensorDataset(data.Dataset):
    def __init__(self, path):
        self.path = path
        self.data = torch.load(path)
        self.nSamples = len(self.data)

    def __getitem__(self, index):
        return self.data[index].to(torch.float32)

    def __len__(self):
        return self.nSamples


class ImageLmdbDataset(data.Dataset):
    def __init__(self, root_dir, crop_size):
        self.root_dir = root_dir
        self.crop_size = crop_size

        # Delay loading LMDB data until after initialization to avoid "can't
        # pickle Environment Object error"
        self.env, self.txn = None, None
        with lmdb.open(root_dir, readonly=True, lock=False,
                             readahead=False, meminit=False) as env:
            with env.begin(write=False) as txn:
                self.n_samples = int(txn.get("num-samples".encode()))

        self.transform = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.ToTensor()
        ])

    def _init_db(self):
        self.env = lmdb.open(self.root_dir, readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)

    def __getitem__(self, index):
        # Delay loading LMDB data until after initialization:
        # https://github.com/chainer/chainermn/issues/129
        if self.env is None:
            self._init_db()
        img_key = 'img-{:0>9}'.format(index + 1)
        img_bin = self.txn.get(img_key.encode())
        img = Image.open(io.BytesIO(img_bin))
        img = img.convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return self.n_samples

class ImgLMDBDataset(data.Dataset):
    def __init__(self, db_path, is_training, patch_size=256, qmap_gen=False):
        self.db_path = db_path
        self.is_training = is_training
        self.patch_size = patch_size
        self.qmap_gen = qmap_gen
        env = lmdb.open(self.db_path,
                             max_readers=1,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        txn = env.begin(write=False)
        self.nSamples = int(txn.get('num-samples'.encode()))
        env.close()

        if is_training:
            self.transform = transforms.Compose([
                transforms.RandomCrop(patch_size),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

    def open_lmdb(self):
        self.env = lmdb.open(self.db_path,
                             max_readers=1,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        self.txn = self.env.begin(write=False)
    
    def __getitem__(self, index):
        if not hasattr(self, 'txn'):
            self.open_lmdb()
        img_key = 'img-{:0>9}'.format(index + 1)
        img_bin = self.txn.get(img_key.encode())
        img = Image.open(io.BytesIO(img_bin))
        img = img.convert('RGB')
        img = self.transform(img)
        
        if self.qmap_gen:
            qmap = torch.ones(1, self.patch_size, self.patch_size) * random.random()
            return img, qmap
        else:
            return img

    def __len__(self):
        return self.nSamples


class VimeoDataset(data.Dataset):
    def __init__(self, lmdb_dir, frame_index, crop_size, color_space='rgb'):
        self.lmdb_dir = lmdb_dir
        self.env = lmdb.open(lmdb_dir,
                             max_readers=1,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        self.txn = self.env.begin(write=False)
        self.num_frames = 7
        self.frame_index = frame_index
        self.index_max= max(frame_index)
        self.nSamples = int(self.txn.get(b'num-samples'))
        self.transform = transforms.Compose([
            video_transforms.RandomCrop(crop_size),
            video_transforms.ToTensor()
        ])
        self.color_space = color_space

    def __getitem__(self, index):
        assert index < len(self), 'index range error'
        frames = []
        start = 1 + torch.randint(0, self.num_frames - self.index_max, [1]).item()
        for i in self.frame_index:
            fid = start + i
            frameKey = '{:0>9}-{:0>3}'.format(index + 1, fid)
            img_bin = self.txn.get(frameKey.encode())
            frame = Image.open(io.BytesIO(img_bin))
            frame = frame.convert('RGB')
            frames.append(frame)
        frames = self.transform(frames)
        if self.color_space == 'ycbcr':
            frames = [rgb2ycbcr(frame) for frame in frames]
        return frames

    def __len__(self):
        return self.nSamples


class YUVDataset(data.Dataset):
    def __init__(self, lmdb_dir, frame_index, crop_size, to_yuv444=True):
        self.lmdb_dir = lmdb_dir
        # print(lmdb_dir)
        self.env = lmdb.open(lmdb_dir,
                             max_readers=1,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        self.txn = self.env.begin(write=False)
        self.num_frames = 9
        self.frame_index = frame_index
        self.index_max = max(frame_index)
        self.nSamples = int(self.txn.get('num-samples'.encode()))
        self.to_yuv444 = to_yuv444

        if isinstance(crop_size, numbers.Number):
            crop_size //= 2
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = tuple(csize //2 for csize in crop_size)

    def __getitem__(self, index):
        assert index < len(self), 'index range error'
        frames = []
        start = random.randint(0, self.num_frames - self.index_max - 1)
        for i in self.frame_index:
            name = '{:0>9}-{:0>3}'.format(index + 1, start + i + 1)
            yuv = self.string_to_arraydict(self.txn.get(name.encode()))
            y, u, v = yuv['y'], yuv['u'], yuv['v']
            # print(y.shape, u.shape)

            if i == self.frame_index[0]:
                params_u = self.get_crop_params(u.shape, self.crop_size)  ## crop parameters
                params_y = tuple(s * 2 for s in params_u)

            ## crop
            y = y[params_y[0]:params_y[1], params_y[2]:params_y[3]]
            u = u[params_u[0]:params_u[1], params_u[2]:params_u[3]]
            v = v[params_u[0]:params_u[1], params_u[2]:params_u[3]]

            y, u, v = (torch.from_numpy(x).unsqueeze(0).float().div(255) for x in (y, u, v))
            if self.to_yuv444:
                yuv = self.yuv420_to_yuv444(y, u, v)
            else:
                yuv = (y, u, v)
            frames.append(yuv)
        return frames

    def string_to_arraydict(self, string):
        array_dict = np.load(io.BytesIO(string))
        return array_dict

    def get_crop_params(self, input_size, output_size):
        w, h = input_size
        tw, th = output_size
        if w <= tw or h <= th:
            print(w, h)
            return 0, w, 0, h
        w1 = random.randint(0, w - tw)
        h1 = random.randint(0, h - th)
        w2 = w1 + tw
        h2 = h1 + th
        return w1, w2, h1, h2

    def yuv420_to_yuv444(self, y, u, v):
        u, v = map(self._upsample_nearest_neighbor, (u, v))
        return torch.cat((y, u, v), dim=0)

    def _upsample_nearest_neighbor(self, t, factor=2):
        return F.interpolate(t.unsqueeze(0), scale_factor=factor, mode='nearest').squeeze(0)

    def __len__(self):
        return self.nSamples


if __name__ == '__main__':
    # lmdb_dir = '/data/gaoyx/dataset/CDVL_lmdb'
    lmdb_dir = '/gdata2/fengrs/dataset/CDVL_lmdb'

    frame_index = [0, 1, 2, 3, 4, 5]
    # The "frame_index" can be any combinations of the indexes
    # which are less than clip length of the dataset.
    # example: [0, 2, 5], [1, 2, 3], [7, 6, 2], ... .
    # Then the YUVDataset will output a tuple of the yuv frames
    # sampled from a video clip, by using the frame_index.
    train_set = YUVDataset(
        lmdb_dir=lmdb_dir,
        frame_index=frame_index,
        crop_size=256,  ## crop size for y
        to_yuv444=True
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )

    data = next(iter(train_loader)) ## list of yuv444 if to_yuv444=True
    print(len(data))
    print(data[0].shape)