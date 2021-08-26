import os
from omegaconf import OmegaConf

from ..global_variables import AVAILABLE_DATASETS

from .modelnet import ModelNetDataset
from .shapenetsem import ShapeNetSemDataset
from .shapenet import ShapeNetDataset
from .abc import ABCDataset
from .shapenetkey import ShapeNetKeyDataset

def load(options = None, config = None):
    """
    Loads dataset with given options and config
    """
    
    dataset_name = options.dataset.lower()
    if dataset_name not in AVAILABLE_DATASETS:
        raise NotImplementedError
        
    if dataset_name in ["modelnet10", "modelnet10_zrotated", "modelnet40", "abc", "shapenetsem", "shapenetkey"]:
        if config is None:
            config = f"configs/{dataset_name}.yaml"
            config = os.path.join(os.path.dirname(__file__), config)
        assert os.path.exists(config), "Config doesn't exists"
    
        if dataset_name == "shapenetsem":
            config = open(config).read() % (options.directory,
                                            options.n_points,
                                            options.n_points,
                                            options.n_points)            
        else:
            config = open(config).read() % (options.directory,
                                            options.n_points,
                                            options.n_points)
            
        dataset = {
            "modelnet10": ModelNetDataset,
            "modelnet10_zrotated": ModelNetDataset,
            "modelnet40": ModelNetDataset,
            "shapenetsem": ShapeNetSemDataset,
            "shapenetkey": ShapeNetKeyDataset,
            "abc": ABCDataset
        }[
            dataset_name
        ](
            OmegaConf.create(config),
            options.train_data,
            options.test_data,
            options.categories
        )
        
    elif dataset_name in ["shapenet", "shapenetsvr"]:
        dataset = ShapeNetDataset(options)
        
    print(dataset)
    return dataset