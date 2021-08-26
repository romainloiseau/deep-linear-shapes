import numpy as np

from tqdm.auto import tqdm
from collections import OrderedDict

class DiscreetMetric:
    
    def __init__(self, mode = "train"):
        
        self.mode = mode
        
        if not hasattr(self, "names"):
            self.names = []
        self.results = OrderedDict(zip(self.names, [0] * len(self.names)))
        
        self.reset()
        
    def __getitem__(self, k):
        return self.results[k]
    
    def __str__(self):
        s = ""
        maxk = max([len(k) for k in self.results.keys()])
        for k, v in self.results.items():
            s += "{}{}\t{:.2f}\n".format(k, " "*(1 + maxk - len(k)), v)
        return s

    def reset(self):
        self.true = np.array([], dtype=np.int64)
        self.pred = np.array([], dtype=np.int64)

    def update(self, true, pred):
        self.true = np.hstack([self.true, true.flatten()])
        self.pred = np.hstack([self.pred, pred.flatten()])

    def compute(self):
        raise NotImplementedError
        
    def can_compute(self):
        return len(self.true) > 0
        
    def log(self, epoch, writer = None, *args, **kwargs):
        self.compute(*args, **kwargs)
        
        metrics = dict()
        for k, v in self.results.items():
            name = f'{type(self).__name__}/{k}/{self.mode}'
            if writer:
                writer.add_scalar(name, v, epoch)
            else:
                tqdm.write(name + " "*(25 - len(name)) + f'{v:.5f}')
            metrics[name] = v
                
        if hasattr(self, "detailed_results"):
            for k, v in self.detailed_results.items():
                name = f'detailed{type(self).__name__}/{k}/{self.mode}'
                if writer:
                    writer.add_scalar(name, v, epoch)
                else:
                    tqdm.write(name + " "*(25 - len(name)) + f'{v:.5f}')
            
                metrics[name] = v
                    
        return metrics