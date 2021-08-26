import torch

from tqdm.auto import tqdm

import numpy as np

from .random import sample_best_distributed, sample_best_distributed_pointwise
from ..losses import Chamfer

class ShapeSampler:
    
    def __init__(self, method = "random", N = 5):
        
        if (method not in ["random", "kmeans++", "template", "firsttemplate"]) and (method.replace("template", "") == ""):
            raise NotImplementedError
        
        self.method = method
        self.N = N
        
    def __call__(self, dataset, aligner = None, output = "samples"):
        if "template" in self.method:
            if self.method in ["template", "firsttemplate"]:
                if hasattr(dataset.data, "point_y"):
                    indices = sample_best_distributed_pointwise(
                        dataset, self.N, get_firsts = "first" in self.method
                    )
                else:
                    indices = sample_best_distributed(
                        dataset.data.y, self.N, get_firsts = "first" in self.method
                    )
            else:
                i = int(self.method.replace("template", ""))
                unique = np.unique(dataset.data.y)
                assert i*len(unique) == self.N, "Invalid number of desired templates"
                
                indices = []
                for u in unique:
                    indices.append(np.random.choice(np.arange(len(dataset.data.y))[dataset.data.y == u], i, replace = False))      
                indices = np.hstack(indices)
                
        elif self.method == "random":
            indices = np.random.choice(
                len(dataset), self.N, replace = len(dataset) < self.N
            )
        elif self.method == "kmeans++":
            indices = self.sample_kmeanspp(dataset, aligner)
        else:
            raise NotImplementedError
            
        if output == "samples":
            return [dataset[int(i)] for i in indices]
        elif output == "indices":
            return indices
        else:
            raise ValueError
    
    def sample_kmeanspp(self, dataset, aligner = None):
                
        if aligner is not None:
            if type(aligner) is str:
                aligner = torch.load(aligner)
                aligner = {"encoder": aligner.encoder, "anet": aligner.LSMs[0].ANet, "aligner": aligner.LSMs[0].Aligner}
            assert "encoder" in aligner.keys()
            assert "anet" in aligner.keys()
            assert "aligner" in aligner.keys()
            aligner["encoder"].eval()
            aligner["anet"].eval()
            aligner["aligner"].eval()
        
        def do_batch(b, aligner = None):
            with torch.no_grad():
                b0 = torch.cat(b[0]).cuda().permute(0, 2, 1)
                b1 = torch.cat(b[1]).cuda().permute(0, 2, 1)
                
                if aligner is not None:
                    b0_params = aligner["anet"](aligner["encoder"](b0)[0])
                    b1 = aligner["aligner"](b1, b0_params)[0]
                
                d = criterion(b0, b1).detach().cpu().numpy()
            return d**2
        
        criterion = Chamfer()
        
        init = [np.random.randint(len(dataset))]
        distances = []
        
        for k in tqdm(range(self.N - 1), leave=False, desc="KMeans++ init"):
            distances.append([])
            
            batch = ([], [])
            for i in tqdm(range(len(dataset)), desc=f"Number {k}", leave=False):
                batch[0].append(dataset[i].pos.unsqueeze(0))
                batch[1].append(dataset[init[-1]].pos.unsqueeze(0))
                if len(batch[0]) >= 512:
                    d = do_batch(batch, aligner)
                    distances[-1] = np.concatenate([distances[-1], d])
                    batch = ([], [])
            if len(batch[0]) >= 1:
                d = do_batch(batch)
                distances[-1] = np.concatenate([distances[-1], d])
                batch = ([], [])
                
            closer_init = np.min(distances, axis = 0)
            tqdm.write(f"Mean minimum Chamfer distance from {1+k} selected shapes: {1000*np.mean(closer_init**.5):.2f}")
            closer_init = closer_init / closer_init.sum()
            init.append(int(np.random.choice(len(dataset), 1, p = closer_init)[0]))
            
        del criterion
        if aligner is not None:
            del aligner
        torch.cuda.empty_cache()
        return init
            
           