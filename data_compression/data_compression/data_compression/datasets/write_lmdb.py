import os, io, glob, time, pickle
import lmdb, random, shutil
import numpy as np
from PIL import Image


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def checkImageIsValid(input_size, minsize):
    w, h = input_size
    if w * h == 0 or w < minsize[0] or h < minsize[1]:
        return False
    return True

def get_crop_params(input_size, output_size):
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

def img_to_string(img):
    f = io.BytesIO()
    img.save(f, 'PNG', compress_level=1)
    return f.getvalue()

def string_to_img(string):
    img = Image.open(io.BytesIO(string))
    img = img.convert('RGB')
    return img


def create_dataset_img(lmdb_dir, img_ps, epoch=1, crop_size=None, minsize=(256, 256)):
    shutil.rmtree(lmdb_dir, ignore_errors=True)
    os.makedirs(lmdb_dir, exist_ok=True)

    nSamples = len(img_ps)
    env = lmdb.open(lmdb_dir, map_size=1099511627776)
    cache = {}
    cnt = 1
    for ie in range(epoch):
        for i in range(nSamples):
            img_p = img_ps[i]
            if not os.path.exists(img_p):
                print('%s does not exist' % img_p)
                continue
            with open(img_p, 'rb') as f:
                img_bin = f.read()
            if img_bin is None:
                continue

            img = string_to_img(img_bin)
            assert checkImageIsValid(img.size, minsize)

            if crop_size:
                w1, w2, h1, h2 = get_crop_params(img.size, crop_size)
                img = img.crop((w1, h1, w2, h2))

            img_bin = img_to_string(img)
            cropiKey = 'img-{:0>9}'.format(cnt)
            cache[cropiKey.encode()] = img_bin

            if cnt % 100 == 0:
                writeCache(env, cache)
                cache = {}
                print('Written %d / %d' % (cnt, nSamples))
            cnt += 1

    nSamples = cnt-1
    cache[b'num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with {} samples'.format(nSamples))


def create_dataset_vimeo(lmdb_dir, img_ps, nFrames):
    shutil.rmtree(lmdb_dir, ignore_errors=True)
    os.makedirs(lmdb_dir, exist_ok=True)

    nSamples = len(img_ps)
    env = lmdb.open(lmdb_dir, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        cropPaths = []
        crop1path = img_ps[i]
        cropPaths.append(crop1path)
        for idx in range(2, nFrames+1):
            cropiPath = crop1path.replace('im1.png', 'im{:0>1}.png'.format(idx))
            assert os.path.exists(cropiPath)
            cropPaths.append(cropiPath)

        img_bins = []
        for cropiPath in cropPaths:
            with open(cropiPath, 'rb') as f:
                img_bini = f.read()

            # assert checkImageIsValid(img.size, minsize)
            img_bins.append(img_bini)

        for i in range(nFrames):
            cropiKey = '{:0>9}-{:0>3}'.format(cnt, i+1)
            cache[cropiKey.encode()] = img_bins[i]

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache[b'num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


def arraydict_to_string(array_dict):
    f = io.BytesIO()
    np.savez(f, **array_dict)
    return f.getvalue()

def string_to_arraydict(string):
    array_dict = np.load(io.BytesIO(string))
    return array_dict


def create_dataset_frameyuv(lmdb_dir, dataset_dir, gop_size, gop_step=1., pixfmt='yuv420'):
    assert pixfmt == 'yuv420'
    shutil.rmtree(lmdb_dir, ignore_errors=True)
    os.makedirs(lmdb_dir, exist_ok=True)

    env = lmdb.open(lmdb_dir, map_size=1099511627776)

    cnt = 0
    cache = {}
    crop_size_u = (192, 320) ## crop size for u, v in yuv420
    scale_factor = 2 if pixfmt == 'yuv420' else 1

    total_nframe = len(glob.glob(os.path.join(dataset_dir, '*', '*.yuv')))
    estimated_total_ngop = total_nframe // gop_size // gop_step
    string = '{}, number of frames: {:.0f}, estimated number of clips: {:.0f}'
    print(string.format('All', total_nframe, estimated_total_ngop))
    start = time.time()
    for video_name in os.listdir(dataset_dir):
        yuv_ps = sorted(glob.glob(os.path.join(dataset_dir, video_name, '*.yuv')))
        string = '{}, number of frames: {:.0f}, estimated number of clips: {:.0f}'
        print(string.format(video_name, len(yuv_ps), len(yuv_ps) // gop_size // gop_step))
        for igop in np.arange(0, len(yuv_ps) // gop_size - 1, gop_step):
            cache_temp = {}
            for offset in range(gop_size):
                iframe = int(igop * gop_size) + offset

                #print(yuv_ps[iframe])
                ## load yuv file
                with open(yuv_ps[iframe], 'rb') as f:
                    yuv = np.fromfile(f, dtype=np.uint8, sep='')
                #print(yuv.shape, yuv.dtype)


                ## get the resolution info
                if offset == 0: ## the following info unchanges in a gop
                    if len(yuv) == 720*1280*(1 + 2/scale_factor**2):
                        size_y = (720, 1280)
                    else:
                        assert len(yuv) == 1080*1920*(1 + 2/scale_factor**2)
                        size_y = (1080, 1920)
                    size_u = tuple(s // scale_factor for s in size_y)
                    npixels_y = size_y[0]*size_y[1]
                    npixels_u = size_u[0]*size_u[1]

                    params_u = get_crop_params(size_u, crop_size_u)  ## crop parameters
                    params_y = tuple(s * scale_factor for s in params_u)

                ## split y, u, v and reshape
                y = yuv[:npixels_y].reshape(size_y)
                u = yuv[npixels_y: npixels_u + npixels_y].reshape(size_u)
                v = yuv[npixels_u + npixels_y:].reshape(size_u)

                ## crop
                y = y[params_y[0]:params_y[1], params_y[2]:params_y[3]]
                u = u[params_u[0]:params_u[1], params_u[2]:params_u[3]]
                v = v[params_u[0]:params_u[1], params_u[2]:params_u[3]]

                ## write as string
                name = '{:0>9}-{:0>3}'.format(cnt + 1, offset + 1)
                array_dict = {'y':y, 'u':u, 'v': v}
                cache_temp[name.encode()] = arraydict_to_string(array_dict)

                #recons = string_to_arraydict(cache_temp[name.encode()])
                #print(len(recons))
                #print(recons['y'].shape, recons['u'].shape)
                #assert 1==0

            cache.update(cache_temp)
            cnt += 1
            if cnt % 100 == 0:
                writeCache(env, cache)
                cache = {}
                string = 'Written {} / {}, time: {:.2f} min'
                print(string.format(cnt, estimated_total_ngop, (time.time() - start) / 60))
    nSamples = cnt
    cache[b'num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


def create_dataset_toysource(path, distrib, n_sample):
    import torch
    from scipy import stats
    if distrib == 'laplace':
         x = stats.laplace(loc=0, scale=1)
         sample = x.rvs(size=n_sample)
         sample = torch.from_numpy(sample).to(torch.float32)
         sample = sample.view(n_sample, 1)
         torch.save(sample, path)
    if distrib == 'banana':
        rho = 0.9
        cov = np.float32(np.eye(N=2) + rho * np.eye(N=2)[::-1])
        x = stats.multivariate_normal(cov=cov)
        sample = x.rvs(size=n_sample)
        x = sample[:, 0]
        y = sample[:, 1] - sample[:, 0] ** 2 + 1
        sample = torch.from_numpy(np.dstack([x, y])).to(torch.float32)
        sample = sample.view(n_sample, 2)
        print((sample - sample.mean(0, keepdim=True)).pow(2).sum(1).mean())
        torch.save(sample, path)
    if distrib ==  'uniform2d':
        x = stats.uniform(loc=-3, scale=6)
        sample = np.dstack([x.rvs(size=n_sample),
                            x.rvs(size=n_sample)])
        sample = torch.from_numpy(sample).to(torch.float32)
        sample = sample.view(n_sample, -1)
        print((sample - sample.mean(0, keepdim=True)).pow(2).sum(1).mean())
        torch.save(sample, path)
    if distrib ==  'uniform8d':
        x = stats.uniform(loc=-3, scale=6)
        sample = np.dstack([x.rvs(size=n_sample) for _ in range(8)])
        sample = torch.from_numpy(sample).to(torch.float32)
        sample = sample.view(n_sample, -1)
        print((sample - sample.mean(0, keepdim=True)).pow(2).sum(1).mean())
        torch.save(sample, path)
    if distrib == 'image48d':
        crop_size = (48 / 3) ** 0.5
        set = ImgLMDBDataset(lmdb_dir='/gdata1/wuyj/dataset/Imagenet', crop_size=crop_size)
        sample = []
        indices = np.random.randint(len(set), size=n_sample)
        from tqdm import tqdm
        for i in tqdm(range(n_sample)):
            sample.append(set[indices[i]])
            if i % 10000 == 0:
                print(i)
        sample = torch.stack(sample, dim=0).to(torch.float32)
        sample = sample.view(n_sample, -1)
        print((sample - sample.mean(0, keepdim=True)).pow(2).sum(1).mean())
        torch.save(sample, path)
    if distrib == 'image768d':
        crop_size = (768 / 3) ** 0.5
        set = ImgLMDBDataset(lmdb_dir='/gdata1/wuyj/dataset/Imagenet', crop_size=crop_size)
        sample = []
        indices = np.random.randint(len(set), size=n_sample)
        from tqdm import tqdm
        for i in tqdm(range(n_sample)):
            sample.append(set[indices[i]])
            if i % 10000 == 0:
                print(i)
        sample = torch.stack(sample, dim=0).to(torch.float32)
        sample = sample.view(n_sample, -1)
        print((sample - sample.mean(0, keepdim=True)).pow(2).sum(1).mean())
        torch.save(sample, path)

if __name__ == '__main__':
    ## iscasgc train 234
    # data_dir = '/data/xiegq/Dataset/CDVLCut'
    # lmdb_dir = '/data/gaoyx/dataset/CDVL_lmdb'
    # create_dataset_frameyuv(lmdb_dir, data_dir, gop_size=9, gop_step=0.7)

    ## 237
    img_ps = sorted(glob.glob('/data/datasets/Flickr2K_HR/*.png'))
    # img_ps += sorted(glob.glob('/data/datasets/CLIC/img_data/mobile/*/*.png'))
    # img_ps += sorted(glob.glob('/data/datasets/CLIC/img_data/professional/*/*.png'))
    lmdb_dir = '/data/datasets/Flickr2K_HR_lmdb'
    create_dataset_img(lmdb_dir, img_ps, epoch=2, crop_size=(512, 512))

    # distrib = 'banana'
    # n_sample = 1000000
    # path = '/gdata2/fengrs/datasets/banana_trainset.pt'
    # create_dataset_toysource(path, distrib, n_sample)
    # n_sample = 10000
    # path = '/gdata2/fengrs/datasets/banana_evalset.pt'
    # create_dataset_toysource(path, distrib, n_sample)

    # distrib = 'image768d'
    # n_sample = 1000000
    # path = '/gdata2/fengrs/datasets/image768d_trainset.pt'
    # create_dataset_toysource(path, distrib, n_sample)
    # n_sample = 10000
    # path = '/gdata2/fengrs/datasets/image768d_evalset.pt'
    # create_dataset_toysource(path, distrib, n_sample)




