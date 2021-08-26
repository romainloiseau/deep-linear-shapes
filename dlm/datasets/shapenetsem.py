import torch
from torch_points3d.datasets.segmentation.shapenet import ShapeNetDataset as tp3dShapeNetDataset

from .fulldataset import FullDataset

from functools import partial
import torch
import numpy as np
        
class ShapeNetSemDataset(FullDataset, tp3dShapeNetDataset):
    def __init__(self, dataset_opt, train_data, test_data, categories):
        dataset_opt.category = categories if categories != [] else None
        tp3dShapeNetDataset.__init__(self, dataset_opt)
        FullDataset.__init__(self, train_data, test_data)