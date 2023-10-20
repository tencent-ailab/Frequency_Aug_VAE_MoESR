from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paths_from_lmdb
from basicsr.utils import FileClient, imfrombytes, img2tensor, rgb2ycbcr, scandir
from basicsr.utils.registry import DATASET_REGISTRY

from pathlib import Path
import random
import cv2
import numpy as np
import torch

@DATASET_REGISTRY.register()
class SingleImageDataset(data.Dataset):
    """Read only lq images in the test phase.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc).

    There are two modes:
    1. 'meta_info_file': Use meta information file to generate paths.
    2. 'folder': Scan folders to generate paths.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt):
        super(SingleImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.lq_folder = opt['dataroot_lq']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder]
            self.io_backend_opt['client_keys'] = ['lq']
            self.paths = paths_from_lmdb(self.lq_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [osp.join(self.lq_folder, line.rstrip().split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.lq_folder, full_path=True)))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load lq image
        lq_path = self.paths[index]
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
        return {'lq': img_lq, 'lq_path': lq_path}

    def __len__(self):
        return len(self.paths)

@DATASET_REGISTRY.register()
class SingleImageSRDataset(data.Dataset):
    """Read only lq images in the test phase.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc).

    There are two modes:
    1. 'meta_info_file': Use meta information file to generate paths.
    2. 'folder': Scan folders to generate paths.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt):
        super(SingleImageSRDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        if 'image_type' not in opt:
            opt['image_type'] = 'png'

        # file client (lmdb io backend)
        if isinstance(opt['gt_path'], str):
            self.paths = sorted([str(x) for x in Path(opt['gt_path']).glob('*.'+opt['image_type'])])
        else:
            self.paths = sorted([str(x) for x in Path(opt['gt_path'][0]).glob('*.'+opt['image_type'])])
            if len(opt['gt_path']) > 1:
                for i in range(len(opt['gt_path'])-1):
                    self.paths.extend(sorted([str(x) for x in Path(opt['gt_path'][i+1]).glob('*.'+opt['image_type'])]))
        if 'face_gt_path' in opt:
            face_list = sorted([str(x) for x in Path(opt['face_gt_path']).glob('*.'+opt['image_type'])])
            self.paths.extend(face_list[:opt['num_face']])
        # self.paths.extend(sorted([str(x) for x in Path(opt['wed_path']).glob('*.bmp')]))
        if 'num_pic' in opt:
            if 'val' or 'test' in opt:
                random.shuffle(self.paths)
                self.paths = self.paths[:opt['num_pic']]
            else:
                self.paths = self.paths[:opt['num_pic']]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load lq image
        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]

        h, w = img_gt.shape[0:2]
        crop_pad_size = 512
        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        # crop
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            # randomly choose top and left coordinates
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            # top = (h - crop_pad_size) // 2 -1
            # left = (w - crop_pad_size) // 2 -1
            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_gt, self.mean, self.std, inplace=True)
        img_gt = img_gt * 2.0 - 1.0
        return {'gt': img_gt, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)

@DATASET_REGISTRY.register()
class SingleImageNPDataset(data.Dataset):
    """Read only lq images in the test phase.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc).

    There are two modes:
    1. 'meta_info_file': Use meta information file to generate paths.
    2. 'folder': Scan folders to generate paths.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt):
        super(SingleImageNPDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        if 'image_type' not in opt:
            opt['image_type'] = 'png'

        if isinstance(opt['gt_path'], str):
            self.gt_paths = sorted([str(x) for x in Path(opt['gt_path']+'/gts').glob('*.'+opt['image_type'])])
            self.lq_paths = sorted([str(x) for x in Path(opt['gt_path']+'/inputs').glob('*.'+opt['image_type'])])
            self.np_paths = sorted([str(x) for x in Path(opt['gt_path']+'/latents').glob('*.npy')])
            self.sample_paths = sorted([str(x) for x in Path(opt['gt_path']+'/samples').glob('*.'+opt['image_type'])])
        else:
            self.gt_paths = sorted([str(x) for x in Path(opt['gt_path'][0]+'/gts').glob('*.'+opt['image_type'])])
            self.lq_paths = sorted([str(x) for x in Path(opt['gt_path'][0]+'/inputs').glob('*.'+opt['image_type'])])
            self.np_paths = sorted([str(x) for x in Path(opt['gt_path'][0]+'/latents').glob('*.npy')])
            self.sample_paths = sorted([str(x) for x in Path(opt['gt_path'][0]+'/samples').glob('*.'+opt['image_type'])])
            if len(opt['gt_path']) > 1:
                for i in range(len(opt['gt_path'])-1):
                    self.gt_paths.extend(sorted([str(x) for x in Path(opt['gt_path'][i+1]+'/gts').glob('*.'+opt['image_type'])]))
                    self.lq_paths.extend(sorted([str(x) for x in Path(opt['gt_path'][i+1]+'/inputs').glob('*.'+opt['image_type'])]))
                    self.np_paths.extend(sorted([str(x) for x in Path(opt['gt_path'][i+1]+'/latents').glob('*.npy')]))
                    self.sample_paths.extend(sorted([str(x) for x in Path(opt['gt_path'][i+1]+'/samples').glob('*.'+opt['image_type'])]))

        assert len(self.gt_paths) == len(self.lq_paths)
        assert len(self.gt_paths) == len(self.np_paths)
        assert len(self.gt_paths) == len(self.sample_paths)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load lq image
        lq_path = self.lq_paths[index]
        gt_path = self.gt_paths[index]
        sample_path = self.sample_paths[index]
        np_path = self.np_paths[index]

        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        img_bytes_gt = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes_gt, float32=True)

        img_bytes_sample = self.file_client.get(sample_path, 'sample')
        img_sample = imfrombytes(img_bytes_sample, float32=True)

        latent_np = np.load(np_path)

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_sample = rgb2ycbcr(img_sample, y_only=True)[..., None]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
        img_sample = img2tensor(img_sample, bgr2rgb=True, float32=True)
        latent_np = torch.from_numpy(latent_np).float()
        latent_np = latent_np.to(img_gt.device)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_sample, self.mean, self.std, inplace=True)
        return {'lq': img_lq, 'lq_path': lq_path, 'gt': img_gt, 'gt_path': gt_path, 'latent': latent_np[0], 'latent_path': np_path, 'sample': img_sample, 'sample_path': sample_path}

    def __len__(self):
        return len(self.gt_paths)
