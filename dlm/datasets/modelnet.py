from itertools import repeat
import os.path as osp
from torch_geometric.io import read_txt_array
from torch_geometric.data import Data

import torch
from torch_points3d.datasets.classification.modelnet import ModelNetDataset as tp3dModelNetDataset
from torch_points3d.datasets.classification.modelnet import SampledModelNet as tp3dSampledModelNet

from .fulldataset import FullDataset

class SampledModelNet(tp3dSampledModelNet):
    def __init__(self, categories, *args, **kwargs):
        self.categories = categories
        super(SampledModelNet, self).__init__(*args, **kwargs)
        
    @property
    def processed_file_names(self):
        return ["training_{}{}.pt".format(self.name, "".join(self.categories)),
                "test_{}{}.pt".format(self.name, "".join(self.categories))]
    
    def process(self):
        torch.save(self.process_set("train"), self.processed_paths[0])
        torch.save(self.process_set("test"), self.processed_paths[1])
        
    def process_set(self, dataset):
        if not osp.exists(osp.join(self.raw_dir, "modelnet{}_shape_names.txt".format(self.name))):
            self.download()
        with open(osp.join(self.raw_dir, "modelnet{}_shape_names.txt".format(self.name)), "r") as f:
            categories = f.read().splitlines()
            categories = sorted(categories)
        with open(osp.join(self.raw_dir, "modelnet{}_{}.txt".format(self.name, dataset)), "r") as f:
            split_objects = f.read().splitlines()
        
        target_categories = [t for t in range(len(categories))]
        if self.categories != []:
            target_categories = [t for t, c in enumerate(categories) if c in self.categories]
            categories = [c for c in categories if c in self.categories]
            
        data_list = []
        for target, category in zip(target_categories, categories):
            folder = osp.join(self.raw_dir, category)
            category_ojects = filter(lambda o: category in o, split_objects)
            paths = ["{}/{}.txt".format(folder, o.strip()) for o in category_ojects]
            for path in paths:
                raw = read_txt_array(path, sep=",")
                data = Data(pos=raw[:, :3], norm=raw[:, 3:], y=torch.tensor([target]))
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)
    
    def get(self, idx):
        data = self.data.__class__()
        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            start, end = slices[idx].item(), slices[idx + 1].item()
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(start, end)
            elif start + 1 == end:
                s = slices[start]
            else:
                s = slice(start, end)
            data[key] = item[s]

        return data
        
class ModelNetDataset(FullDataset, tp3dModelNetDataset):
    def __init__(self, dataset_opt, train_data, test_data, categories = []):
        super(tp3dModelNetDataset, self).__init__(dataset_opt)

        number = dataset_opt.number
        if str(number) not in self.AVAILABLE_NUMBERS:
            raise Exception("Only ModelNet10 and ModelNet40 are available")
        self.train_dataset = SampledModelNet(
            categories,
            self._data_path,
            name=str(number),
            train=True,
            transform=self.train_transform,
            pre_transform=self.pre_transform,
        )
        self.test_dataset = SampledModelNet(
            categories,
            self._data_path,
            name=str(number),
            train=False,
            transform=self.test_transform,
            pre_transform=self.pre_transform,
        )

        FullDataset.__init__(self, train_data, test_data)
        
        if str(dataset_opt.number) == "10":
            self._categories = [
                "bathtub", "bed", "chair", "desk", "dresser",
                "monitor", "night_stand", "sofa", "table", "toilet"
            ]
        elif str(dataset_opt.number) == "40":
            self._categories = [
                'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl',
                'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser',
                'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop',
                'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio',
                'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent',
                'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
            ]
        else:
            raise ValueError