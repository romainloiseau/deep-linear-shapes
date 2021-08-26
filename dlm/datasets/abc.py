import os.path as osp
import os
import torch

from tqdm.auto import tqdm

import py7zr

from torch_geometric.data import InMemoryDataset, Data

from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.utils.download import download_url

from .fulldataset import FullDataset
from ..utils import sample_obj


class SampledABC(InMemoryDataset):

    def __init__(self, root, chuncks = 1, train=True, transform=None, pre_transform=None, pre_filter=None):
        self.chuncks = chuncks
        super(SampledABC, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)
        
    @property
    def raw_file_names(self):
        return ["object"]

    @property
    def processed_file_names(self):
        return ["training.pt", "test.pt"]

    def download(self):
        if not osp.exists(osp.join(self.root, f"processed/training.pt")):
            for i in range(min(self.chuncks, 100)):
                print(f"Chunk {i}")
                try:
                    url = f"https://archive.nyu.edu/rest/bitstreams/{89085 + 3*i}/retrieve"
                    destination = f"abc_{str(i).zfill(4)}_obj_v00.7z"

                    path = osp.join(self.root, destination)
                    
                    if str(10000*i).zfill(8) not in os.listdir(self.raw_dir):
                        if not osp.exists(path):
                            retrieved = download_url(url, self.root)
                            os.rename(retrieved, retrieved.replace("retrieve", destination))
                            assert path == retrieved.replace("retrieve", destination)
                        else:
                            print(f"\tUsing already downloaded file {path}")

                        with py7zr.SevenZipFile(path, mode='r') as z:
                            if z.getnames()[0] not in os.listdir(self.raw_dir):
                                print(f"\tExtracting 7z file to {self.raw_dir}")
                                z.extractall(path=self.raw_dir)
                            else:
                                print(f"\tUsing already extracted file {path}")
                    else:
                        print(f"\tUsing already extracted objs")
                except:
                    print(f"\tFailed to download chunk {i}")
                    print(f"\tRemoving retrived file")
                    try:
                        if osp.exists(retrieved):
                            os.remove(retrieved)
                    except:
                        pass
                    
    def process(self):
        torch.save(self.process_set("train"), self.processed_paths[0])
        torch.save(self.process_set("test"), self.processed_paths[1])

    def process_set(self, dataset):
        split_objects = os.listdir(self.raw_dir)
        split_objects.sort()
        
        if dataset != "train":
            split_objects = split_objects[:10]
        
        data_list = []
        
        for obj in tqdm(split_objects, desc = "Preprocessing"):
            obj = osp.join(self.raw_dir, obj)
            path_obj = [p for p in os.listdir(obj) if p[-3:] == "obj"]
            if len(path_obj) == 1:
                path_obj = path_obj[0]

                raws = sample_obj(os.path.join(obj, path_obj))

                for raw in raws:
                    data_list.append(Data(pos=torch.tensor(raw[:, :3]).float(),
                                          y=torch.tensor([0])))

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)

    def __repr__(self):
        return "{}{}({})".format(self.__class__.__name__, self.name, len(self))
        
class ABCDataset(FullDataset, BaseDataset):
    def __init__(self, dataset_opt, train_data, test_data, *args, **kwargs):
        
        BaseDataset.__init__(self, dataset_opt)
        
        self.train_dataset = SampledABC(
            self._data_path,
            train=True,
            chuncks = 20,
            transform=self.train_transform,
            pre_transform=self.pre_transform,
        )
        
        FullDataset.__init__(self, train_data, test_data)
        
        self._categories = self.train_dataset.raw_file_names