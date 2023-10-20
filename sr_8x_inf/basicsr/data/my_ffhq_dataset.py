import random
import time
import os
from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

import cv2
from omegaconf import OmegaConf


# @DATASET_REGISTRY.register()
class PAIREDFFHQDataset(data.Dataset):
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
        super(PAIREDFFHQDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
        self.lr_folder = opt['dataroot_lr']
        self.lr_8x_folder = opt['dataroot_lr_8x']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = self.gt_folder
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError("'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            self.paths = sorted(os.listdir(self.gt_folder))
            self.paths = [(os.path.join(self.gt_folder, gt_path), os.path.join(self.lr_folder, os.path.basename(gt_path)), os.path.join(self.lr_8x_folder, os.path.basename(gt_path))) for gt_path in self.paths]
            # FFHQ has 70000 images in total
            # self.paths = [osp.join(self.gt_folder, f'{v:08d}.png') for v in range(70000)]
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
        (gt_path, lr_path, lr_8x_path) = self.paths[index]
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                gt_img_bytes = self.file_client.get(gt_path)
                lr_img_bytes = self.file_client.get(lr_path)
                lr_8x_img_bytes = self.file_client.get(lr_8x_path)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                (gt_path, lr_path, lr_8x_path) = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        img_gt = imfrombytes(gt_img_bytes, float32=True)
        img_lr = imfrombytes(lr_img_bytes, float32=True)
        img_lr_8x = imfrombytes(lr_8x_img_bytes, float32=True)
        img_lr_2x = cv2.resize(img_lr, (img_lr.shape[0]*2, img_lr.shape[1]*2), interpolation=cv2.INTER_CUBIC)
    
        # random horizontal flip
        img_gt, img_lr_8x, img_lr_2x = augment([img_gt, img_lr_8x, img_lr_2x], hflip=self.opt['use_hflip'], rotation=False)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
        img_lr_8x = img2tensor(img_lr_8x, bgr2rgb=True, float32=True)
        img_lr_2x = img2tensor(img_lr_2x, bgr2rgb=True, float32=True)

        return {'gt': img_gt, 'gt_path': gt_path, 'lr': img_lr_8x, 'lr_path': lr_path, 'lr_sr': img_lr_2x}

    def __len__(self):
        return len(self.paths)


if __name__ == "__main__":
    from ldm.util import instantiate_from_config
    import numpy as np
    from PIL import Image
    config_path = "configs/bsr_sr/config_sr_finetune_FFHQ.yaml"
    config = OmegaConf.load(config_path)
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    
    for item in data.train_dataloader():
        lr = item['lr']
        gt = item['gt']
        lr_sr = item['lr_sr']
        lr = np.transpose((lr[0] * 255).cpu().numpy(), (1,2,0)).astype(np.uint8)
        gt = np.transpose((gt[0] * 255).cpu().numpy(), (1,2,0)).astype(np.uint8)
        lr_sr = np.transpose((lr_sr[0] * 255).cpu().numpy(), (1,2,0)).astype(np.uint8)
        Image.fromarray(lr).save('./debug/lr.png')
        Image.fromarray(gt).save('./debug/gt.png')
        Image.fromarray(lr_sr).save('./debug/lr_sr.png')
        import pdb;pdb.set_trace()

        # Image.fromarray(np.clip(np.transpose(((1+c_concat[0][0]) * 0.5 * 255).cpu().numpy(), (1,2,0)).astype(np.uint8), 0, 255)).save('./debug/1.png')
