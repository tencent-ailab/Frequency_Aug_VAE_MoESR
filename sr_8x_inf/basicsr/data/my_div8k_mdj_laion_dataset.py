import random
import time
import os
from os import path as osp
import json
import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import augment, random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.imresize import imresize
from basicsr.utils.registry import DATASET_REGISTRY

import cv2
import glob
from omegaconf import OmegaConf


# @DATASET_REGISTRY.register()
class PAIREDDIV8KMDJLAIONDataset(data.Dataset):
    """FFHQ dataset.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            mean (list | tuple): Image mean.
            std (list | tuple): Image std.
            use_hflip (bool): Whether to horizontally flip.

    """

    def __init__(self, opt):
        super(PAIREDDIV8KMDJLAIONDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None

        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']


        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = self.gt_folder
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError("'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            self.paths = []
            for idx, _ in enumerate(self.gt_folder):
                if os.path.isdir(self.gt_folder[idx]): 
                    this_gt_folder = self.gt_folder[idx]
                    gt_fns = sorted(glob.glob(os.path.join(this_gt_folder, "*.png")))
                    if len(gt_fns) == 0:
                        gt_fns = sorted(glob.glob(os.path.join(this_gt_folder, "*.jpg")))
                    
                    self.paths.extend([os.path.join(this_gt_folder, os.path.basename(fn)) for fn in gt_fns])
                
                elif os.path.isfile(self.gt_folder[idx]) and self.gt_folder[idx].endswith(".txt"): 
                    filename = self.gt_folder[idx]
                    f = open(filename, 'r')
                    openimage_names = [line.strip() for line in f.readlines()]
                    self.paths.extend(openimage_names)
                
                elif os.path.isfile(self.gt_folder[idx]) and self.gt_folder[idx].endswith(".json"): 
                    filename = self.gt_folder[idx]
                    json_file = json.load(open(filename, 'r'))
                    image_names = [item['image_file'] for item in json_file]
                    self.paths.extend(image_names)
                    
            print("Length of dataset:", len(self.paths))

    def __getitem__(self, index):
        '''
        Returns:
            img_lr: lr image with siz of img_gt
            lr_sr: lr image the size of 1/4 img_gt 
        '''
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load gt image
        gt_path = self.paths[index]
        # avoid errors caused by high latency in reading files
        
        retry = 3
        while retry > 0:
            try:
                gt_img_bytes = self.file_client.get(gt_path)
                img_gt = imfrombytes(gt_img_bytes, float32=False) # bgr, [0,255]
                gt_size = self.opt['gt_size']           
                img_gt = random_crop(img_gt, gt_size)
                break
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion

            
        # random horizontal flip
        img_gt = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False)

        # get lr image
        new_size = (int(img_gt.shape[0]/8), int(img_gt.shape[1]/8)) 
        img_lr = imresize(img_gt, output_shape=new_size) # (H/8, W/8)

        # new_size = (int(img_gt.shape[0]), int(img_gt.shape[1])) 
        # img_lr_8x = imresize(img_lr, output_shape=new_size) # (H, W)

        new_size = (int(img_gt.shape[0]/4), int(img_gt.shape[1]/4)) 
        img_lr_2x = imresize(img_lr, output_shape=new_size) # (H/4, W/4)

                
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True) / 255.0
        # img_lr_8x = img2tensor(img_lr_8x, bgr2rgb=True, float32=True) / 255.0
        img_lr_2x = img2tensor(img_lr_2x, bgr2rgb=True, float32=True) / 255.0
        
        # assert (img_lr_8x >=0).all() and (img_lr_8x <=1).all()
        assert (img_lr_2x >=0).all() and (img_lr_2x <=1).all()

        return {'gt': img_gt, 'gt_path': gt_path, 'lr': 0, 'lr_sr': img_lr_2x}

    def __len__(self):
        return len(self.paths)


def collate_fn(data):
    img_gt = torch.stack([example["gt"] for example in data])
    img_gt = img_gt.to(memory_format=torch.contiguous_format)

    gt_path = [example["gt_path"] for example in data]

    img_lr_8x = torch.stack([example["lr"] for example in data])
    img_lr_8x = img_lr_8x.to(memory_format=torch.contiguous_format)

    img_lr_2x = torch.stack([example["lr_sr"] for example in data])
    img_lr_2x = img_lr_2x.to(memory_format=torch.contiguous_format)

    return {
        'gt': img_gt, 
        'gt_path': gt_path, 
        'lr': img_lr_8x, 
        'lr_sr': img_lr_2x
    }


if __name__ == "__main__":
    from ldm.util import instantiate_from_config
    import numpy as np
    from PIL import Image
    config_path = "configs/extreme_sr/config_sr_timeMoe_DIV2K.yaml"
    config = OmegaConf.load(config_path)
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    
    for item in data.train_dataloader():
        lr = item['lr']
        gt = item['gt']
        lr_sr = item['lr_sr']

        # lr = np.transpose((lr[0] * 255).cpu().numpy(), (1,2,0)).astype(np.uint8)
        # gt = np.transpose((gt[0] * 255).cpu().numpy(), (1,2,0)).astype(np.uint8)
        # lr_sr = np.transpose((lr_sr[0] * 255).cpu().numpy(), (1,2,0)).astype(np.uint8)
        
        # Image.fromarray(lr).save('./debug/lr.png')
        # Image.fromarray(gt).save('./debug/gt.png')
        # Image.fromarray(lr_sr).save('./debug/lr_sr.png')

        # Image.fromarray(np.clip(np.transpose(((1+c_concat[0][0]) * 0.5 * 255).cpu().numpy(), (1,2,0)).astype(np.uint8), 0, 255)).save('./debug/1.png')