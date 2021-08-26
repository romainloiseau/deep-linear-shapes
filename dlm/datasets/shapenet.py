import sys

from PIL import Image

from os.path import join, dirname, exists

from easydict import EasyDict
import numpy as np

import matplotlib.pyplot as plt

from torchvision import transforms as T

from .AtlasNetData import dataset_shapenet
from .AtlasNetData import argument_parser as argument_parser
from .AtlasNetData import my_utils as my_utils

from torch_geometric.data import Data

from copy import deepcopy

from tqdm import tqdm 

import torch
from .AtlasNetData import augmenter as augmenter

from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.utils.colors import COLORS

from ..viz import print_pc
from ..utils import ShapeSampler

from omegaconf import OmegaConf

class ShapeNet(dataset_shapenet.ShapeNet):
    
    def __init__(self, *args, **kwargs):
        super(ShapeNet, self).__init__(*args, **kwargs)
        
        self.classes2int = {v: i for i, v in enumerate(self.classes)}
        
        self.data = EasyDict({"y": torch.tensor([self.classes2int[m["category"]] for m in self.data_metadata])})
        
        self.idx_image_val = 0
        
    def init_singleview(self):
        super(ShapeNet, self).init_singleview()
        
        self.train_preprocess = T.Compose([
            T.RandomCrop(127),
            T.RandomHorizontalFlip(),
            T.Resize(size=224, interpolation=2),
            T.ToTensor(),           
        ])

        self.val_preprocess = T.Compose([
            T.CenterCrop(127),
            T.Resize(size=224, interpolation=2),
            T.ToTensor(),          
        ])
        
        #self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __getitem__(self, index):
        return_dict = deepcopy(self.data_metadata[index])
        # Point processing
        points = self.data_points[index]
        points = points.clone()
        if self.opt.sample:
            choice = np.random.choice(points.size(0), self.num_sample, replace=True)
            points = points[choice, :]

        if self.opt.SVR:
            # Image processing
            N = np.random.randint(1, self.num_image_per_object) if self.num_image_per_object != 1 else 1
            if self.train: 
                im = Image.open(join(return_dict['image_path'], ShapeNet.int2str(N) + ".png"))
                im = self.train_preprocess(im)  # random crop
            else:
                im = Image.open(join(return_dict['image_path'],
                                     ShapeNet.int2str(N if self.idx_image_val is None else self.idx_image_val) + ".png"))
                im = self.val_preprocess(im)  # center crop
            
            im = im[:3, :, :]
            #im = self.normalize(im)
            
            return Data(pos=points[:, :3].contiguous(),
                        im=im,
                        y=torch.tensor([self.classes2int[return_dict["category"]]]))
            
        else:
            return Data(pos=points[:, :3].contiguous(),
                        y=torch.tensor([self.classes2int[return_dict["category"]]]))

class ShapeNetDataset(BaseDataset):
    def __init__(self, options):
        super(ShapeNetDataset, self).__init__(OmegaConf.create({"dataroot": ""}))
        
        self.opt = argument_parser.parser()

        self.opt.SVR = "svr" in options.dataset.lower()
        self.opt.number_points = options.n_points

        if self.opt.SVR or options.categories == ["svr"]:
            self.opt.shapenet13 = True
        else:
            self.opt.class_choice = options.categories

        self.train_dataset = ShapeNet(self.opt, train=True)
        self.test_dataset = [ShapeNet(self.opt, train=False)]
        
        self._categories = [self.train_dataset.id2names[v] for v in self.train_dataset.classes]
        
    def show(self, K = 10, dataset = "train"):
        plt.figure(figsize = (20, 5), dpi = 100)        
        plt.hist([self.train_dataset.classes2int[pc["category"]] for pc in self.train_dataset.data_metadata],
                 bins = np.arange(1+len(self._categories)) - .5 - .33,
                 rwidth = .33, label = "train")
        if hasattr(self, "test_dataset") and self.test_dataset is not None:
            plt.hist([self.test_dataset[0].classes2int[pc["category"]] for pc in self.test_dataset[0].data_metadata],
                     bins = np.arange(1+len(self._categories)) - .5,
                     rwidth = .33, label = "test")
        if hasattr(self, "val_dataset") and self.val_dataset is not None:
            plt.hist([self.val_dataset[0].classes2int[pc["category"]] for pc in self.val_dataset[0].data_metadata],
                     bins = np.arange(1+len(self._categories)) - .5 + .33,
                     rwidth = .33, label = "val")

        plt.legend(loc="best")
        plt.xticks(ticks = np.arange(len(self._categories)),
                   labels = self._categories,
                   rotation = 45)
        plt.xlim(-1, len(self._categories))
        plt.show()
        
        to_print = ShapeSampler("template" if K>=len(self._categories) else "random", K)(self.test_dataset[0] if dataset == "test" else self.train_dataset)
        print_pc(to_print, titles = np.array(self._categories)[[tp.y.item() for tp in to_print]])