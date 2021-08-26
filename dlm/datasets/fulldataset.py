import copy
import numpy as np
import matplotlib.pyplot as plt

from functools import partial

import torch
from torch_geometric.data import Data

from ..utils import ShapeSampler
from ..viz import print_pc

class FullDataset:
    """
    A class implemented to unlock the loading of a proportion of the data or
    the training on train and test data
    """
    
    def __init__(self, train_data = 1, test_data = 1):
        
        if hasattr(self.train_dataset, "__data_list__"): 
            del self.train_dataset.__data_list__
        if hasattr(self, "test_dataset") and self.test_dataset is not None and hasattr(self.test_dataset[0], "__data_list__"):
            del self.test_dataset[0].__data_list__
        if hasattr(self, "val_dataset") and self.test_dataset is not None and hasattr(self.val_dataset, "__data_list__"):
            del self.val_dataset.__data_list__
        
        self.hparams = {"n_points": self.train_dataset[0].pos.size(0)}
        
        if hasattr(self.train_dataset.data, "category"):
            self.train_dataset.data["point_y"] = self.train_dataset.data["y"]
            self.train_dataset.slices["point_y"] = self.train_dataset.slices["y"]

            self.train_dataset.data["y"] = self.train_dataset.data["category"][self.train_dataset.slices["category"][:-1]]
            self.train_dataset.slices["y"] = self.train_dataset.slices["id_scan"]

            del self.train_dataset.data.category
            del self.train_dataset.slices["category"]
        
        if hasattr(self, "test_dataset") and self.test_dataset is not None and hasattr(self.test_dataset[0].data, "category"):
            self.test_dataset[0].data["point_y"] = self.test_dataset[0].data["y"]
            self.test_dataset[0].slices["point_y"] = self.test_dataset[0].slices["y"]

            self.test_dataset[0].data["y"] = self.test_dataset[0].data["category"][self.test_dataset[0].slices["category"][:-1]]
            self.test_dataset[0].slices["y"] = self.test_dataset[0].slices["id_scan"]

            del self.test_dataset[0].data.category
            del self.test_dataset[0].slices["category"]
            
        if self.val_dataset is not None and hasattr(self.val_dataset.data, "category"):
            self.val_dataset.data["point_y"] = self.val_dataset.data["y"]
            self.val_dataset.slices["point_y"] = self.val_dataset.slices["y"]

            self.val_dataset.data["y"] = self.val_dataset.data["category"][self.val_dataset.slices["category"][:-1]]
            self.val_dataset.slices["y"] = self.val_dataset.slices["id_scan"]

            del self.val_dataset.data.category
            del self.val_dataset.slices["category"]
                
        if train_data >= 2:
            if hasattr(self, "test_dataset") and self.test_dataset is not None:
                for k in self.train_dataset.data.keys:
                    self.train_dataset.data[k] = torch.cat([self.train_dataset.data[k],
                                                            self.test_dataset[0].data[k]])
                for k in self.train_dataset.slices.keys():
                    self.train_dataset.slices[k] = torch.cat([self.train_dataset.slices[k],
                                                              self.train_dataset.slices[k][-1] + self.test_dataset[0].slices[k][1:]])
                
            if train_data == 3 and hasattr(self, "val_dataset") and self.val_dataset is not None:
                for k in self.train_dataset.data.keys:
                    self.train_dataset.data[k] = torch.cat([self.train_dataset.data[k],
                                                            self.val_dataset.data[k]])
                for k in self.train_dataset.slices.keys():
                    self.train_dataset.slices[k] = torch.cat([self.train_dataset.slices[k],
                                                              self.train_dataset.slices[k][-1] + self.val_dataset.slices[k][1:]])
        elif train_data == 1:
            pass
        elif train_data > 0 and train_data < 1:
            choice = ShapeSampler(
                "template",
                np.ceil(train_data * len(self.train_dataset))
            )(
                self.train_dataset,
                output = "indices"
            )

            reduced_data = Data()
            reduced_slides = {}

            for k in self.train_dataset.data.keys:
                reduced_data[k] = torch.cat([self.train_dataset.get(c)[k] for c in choice], 0)
                reduced_slides[k] = torch.tensor(np.cumsum([0] + [len(self.train_dataset.get(c)[k]) for c in choice]))

            del self.train_dataset.data, self.train_dataset.slices
            self.train_dataset.data, self.train_dataset.slices = reduced_data, reduced_slides
        else:
            raise ValueError(f"Invalid value {train_data} for train_data")
        
        if test_data >= 2:
            assert train_data == test_data, "Should train with all data if testing with all data"
            self.test_dataset = copy.deepcopy(self.train_dataset)
        elif test_data == 1:
            pass
        elif test_data > 0 and test_data < 1:
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid value {test_data} for test_data")
            
    def show(self, K = 10, dataset = "train"):
        plt.figure(figsize = (10, 3), dpi = 100)
        plt.hist(self.train_dataset.data.y.numpy().flatten(),
                 bins = np.arange(1+len(self._categories)) - .5 - .33,
                 rwidth = .33, label = "train")
        if hasattr(self, "test_dataset") and self.test_dataset is not None:
            plt.hist(self.test_dataset[0].data.y.numpy().flatten(),
                     bins = np.arange(1+len(self._categories)) - .5,
                     rwidth = .33, label = "test")
        if hasattr(self, "val_dataset") and self.val_dataset is not None:
            plt.hist(self.val_dataset.data.y.numpy().flatten(),
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
        
    def _dataloader(self, dataset, pre_batch_collate_transform, conv_type, precompute_multi_scale, **kwargs):
        batch_collate_function = self.__class__._get_collate_function(
            conv_type, precompute_multi_scale, pre_batch_collate_transform
        )
        
        num_workers = kwargs.get("num_workers", 0)
        persistent_workers = (num_workers > 0)
        
        dataloader = partial(
            torch.utils.data.DataLoader, collate_fn=batch_collate_function, worker_init_fn=np.random.seed,
            persistent_workers=persistent_workers
        )
        return dataloader(dataset, **kwargs)