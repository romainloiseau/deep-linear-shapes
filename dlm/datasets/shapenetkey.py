import os.path as osp
import os
import json
import torch

import numpy as np

from glob import glob

from tqdm.auto import tqdm

from torch_geometric.datasets import ModelNet
from torch_geometric.data import DataLoader, InMemoryDataset, Data
import torch_geometric.transforms as T

from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.utils.download import download_url

from .fulldataset import FullDataset
from ..utils import sample_obj

from ..global_variables import NAMES2ID, ID2NAMES

def naive_read_pcd(path):
    lines = open(path, 'r').readlines()
    idx = -1
    for i, line in enumerate(lines):
        if line.startswith('DATA ascii'):
            idx = i + 1
            break
    lines = lines[idx:]
    lines = [line.rstrip().split(' ') for line in lines]
    data = np.asarray(lines)
    pc = np.array(data[:, :3], dtype=np.float)
    colors = np.array(data[:, -1], dtype=np.int)
    colors = np.stack([(colors >> 16) & 255, (colors >> 8) & 255, colors & 255], -1)
    return pc, colors

def normalize_pc(pc):
    pc = pc - pc.mean(0)
    #Don't do the following to measure the same kp accuracy as competitors
    #print(np.max(np.linalg.norm(pc, axis=-1)))
    #pc /= np.max(np.linalg.norm(pc, axis=-1))
    return pc

class SampledShapeNetKey(InMemoryDataset):

    def __init__(self, categories, root, dataset="train", transform=None, pre_transform=None, pre_filter=None):
        self.categories = categories
        super(SampledShapeNetKey, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[{"train": 0, "val": 1, "test": 2}[dataset]]
        self.data, self.slices = torch.load(path)
        self.nclasses = self[0].iskeypoint.size(0)
        
    @property
    def raw_file_names(self):
        return ["airplane", "bathtub", "bed", "bottle",
                "cap", "car", "chair", "guitar", "helmet",
                "knife", "laptop", "motorcycle", "mug",
                "skateboard", "table", "vessel"]

    @property
    def processed_file_names(self):
        return ["training{}.pt".format("".join(self.categories)),
                "val{}.pt".format("".join(self.categories)),
                "test{}.pt".format("".join(self.categories))]
                    
    def process(self):
        torch.save(self.process_set("train"), self.processed_paths[0])
        torch.save(self.process_set("val"), self.processed_paths[1])
        torch.save(self.process_set("test"), self.processed_paths[2])

    def process_set(self, dataset):
        annots = json.load(open(os.path.join(self.root, "annotations/all.json")))
        if self.categories != []:
            annots = [annot for annot in annots if ID2NAMES[annot['class_id']] in self.categories]
            
        keypoints = dict([(annot['model_id'], [(kp_info['pcd_info']['point_index'], kp_info['semantic_id']) for kp_info in annot['keypoints']]) for annot in annots])
        rotation_groups = dict([(annot['model_id'], annot['symmetries']['rotation']) for annot in annots])
        
        self.nclasses = max([max([kp_info['semantic_id'] for kp_info in annot['keypoints']]) for annot in annots]) + 1
        
        split_models = open(os.path.join(self.root, f"splits/{dataset}.txt")).readlines()
        split_models = [m.split('-')[-1].rstrip('\n') for m in split_models]
        
        NAMES2IDX = {x: i for i, x in enumerate(self.raw_file_names)}
        
        data_list = []
        for cat in tqdm(self.categories if self.categories != [] else self.raw_file_names, leave = False, desc = f"Processing {dataset} set"):
            for fn in tqdm(glob(os.path.join(self.root, "pcds", NAMES2ID[cat], '*.pcd')), leave = False, desc = f"Processing {cat}s"):
                model_id = os.path.basename(fn).split('.')[0]
                if model_id not in split_models:
                    continue
                    
                idx2semid = dict()
                curr_keypoints = -np.ones((self.nclasses,), dtype=np.int)
                for i, kp in enumerate(keypoints[model_id]):
                    curr_keypoints[kp[1]] = kp[0]
                    idx2semid[i] = kp[1]
                    
                pos = torch.tensor(normalize_pc(naive_read_pcd(fn)[0])).float()
                kp = []
                for idx in curr_keypoints:
                    if idx >= 0:
                        kp.append(pos[idx].unsqueeze(0))
                    else:
                        kp.append(torch.tensor([np.nan, np.nan, np.nan]).unsqueeze(0))
                        
                kp = torch.cat(kp, 0)
                
                data_list.append(Data(
                    pos = pos,
                    y=torch.tensor([NAMES2IDX[cat]]),
                    keypoints=kp,
                    iskeypoint=torch.tensor(curr_keypoints>=0),
                    idxkeypoints=torch.tensor(curr_keypoints),
                    #name = model_id,
                    #rotation = rotation_groups[model_id],
                    #idx2semid = idx2semid,                    
                ))
                
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)
    
    def __getitem__(self, idx):
        data = super(SampledShapeNetKey, self).__getitem__(idx)
        return data

    def __repr__(self):
        return "{}{}({})".format(self.__class__.__name__, self.name, len(self))
        
class ShapeNetKeyDataset(FullDataset, BaseDataset):
    def __init__(self, dataset_opt, train_data, test_data, categories = [], *args, **kwargs):
        
        BaseDataset.__init__(self, dataset_opt)
        
        self.train_dataset = SampledShapeNetKey(
            categories,
            self._data_path,
            dataset="train",
            transform=self.train_transform,
            pre_transform=self.pre_transform,
        )
        self.test_dataset = SampledShapeNetKey(
            categories,
            self._data_path,
            dataset="test",
            transform=self.test_transform,
            pre_transform=self.pre_transform,
        )
        self.val_dataset = SampledShapeNetKey(
            categories,
            self._data_path,
            dataset="val",
            transform=self.test_transform,
            pre_transform=self.pre_transform,
        )
        
        FullDataset.__init__(self, train_data, test_data)
        
        self._categories = self.train_dataset.raw_file_names