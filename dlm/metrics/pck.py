import numpy as np
from collections import OrderedDict

import torch

from .basemetric import DiscreetMetric

class PCK(DiscreetMetric):
    
    def __init__(self, classes, *args, **kwargs):
        self.names = []
        self.perclass_names = []
        self.precs = [.01, .02]
        
        for prec in self.precs:
            self.names.append(f'global_pck_{prec}')
            self.names.append(f'avg_pck_{prec}')
        
        if type(classes) is int:
            for prec in self.precs:
                for i in range(classes):
                    self.perclass_names.append(f'pck_cls_{i}_{prec}')
            self.n_classes = classes
        elif type(classes) is list:
            for prec in self.precs:
                for c in classes:
                    self.perclass_names.append(f'pck_{c}_{prec}')
            self.n_classes = len(classes)
            
        super(PCK, self).__init__(*args, **kwargs)
        
    def update(self, true, pred, cat):
        self.true += true
        self.pred += pred
        self.cat = np.hstack([self.cat, cat.flatten()])
        
    def reset(self):
        self.true = []
        self.pred = []
        self.cat = np.array([], dtype=np.int64)
        
    def compute(self):
        
        true = torch.stack(self.true)
        pred = torch.stack(self.pred)
        
        pck_per_class_per_prec = []
        results = []
        
        self.pck = torch.norm(pred-true, dim = 1)#((pred - true)**2).sum(1)**.5##
        valid = (~true[:, 0].isnan()).sum(-1).float()
        
        for prec in self.precs:
            pck = (self.pck <= prec).sum(-1) / valid
            
            pck_per_class = []
            for i in range(self.n_classes):
                if (self.cat == i).sum() > 0:
                    pck_per_class.append(pck[self.cat == i].mean().item())
                else:
                    pck_per_class.append(np.nan)
            pck_per_class_per_prec.append(pck_per_class)
            results.append(pck.mean().item())
            results.append(np.mean(np.nan_to_num(pck_per_class)))
            
        self.results = OrderedDict(zip(self.names,
                                       results))
        self.detailed_results = OrderedDict(zip(self.perclass_names, np.array(pck_per_class_per_prec).flatten()))
        
        self.reset()
        
    def log(self, epoch, writer = None, *args, **kwargs):
        metrics = super(PCK, self).log(epoch, writer, *args, **kwargs)
        if writer and epoch%10 == 0:
            #self.pck[torch.isnan(self.pck)] = -.01
            writer.add_histogram(f"pcks/{self.mode}", torch.where(self.pck < .1, self.pck, torch.tensor(.11).to(self.pck.device)), global_step = epoch)
        return metrics
        
if __name__=="__main__":
    
    pck = PCK()