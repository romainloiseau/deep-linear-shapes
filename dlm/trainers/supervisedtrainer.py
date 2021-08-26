import numpy as np

import torch
from torch import nn

from .clusteringtrainer import ClusteringTrainer

class SupervisedTrainer(ClusteringTrainer):
    
    def __init__(self, *args, **kwargs):
        super(SupervisedTrainer, self).__init__(*args, **kwargs)
        
        self.supervise_category = {"train": True, "test": False, "val": False}
    
    def initialize_materials(self, *args, **kwargs):
        super(SupervisedTrainer, self).initialize_materials(*args, **kwargs)
        
        self.model.keep_cat = True
        for i in range(self.model.n_clusters):
            if not hasattr(self.model.LSMs[i], "y"):
                self.model.LSMs[i].y = nn.Parameter(
                    torch.tensor(np.random.randint(len(self.dataset._categories))),
                    requires_grad = False
                )
                
    def compute_indexes(self, batch, reconstructions):
        dist = self.criterion(batch.pos, reconstructions, reduction = None)
        idx = dist.min(axis = 1).indices

        cats = self.model.LSMs.ys
        cats_take = (cats == batch.y.to(cats.device)).detach().to(reconstructions.device)
        assert cats_take.any(axis = -1).all(), "Should have at least one lsm per categories"
            
        dist = torch.where(cats_take, dist, torch.tensor(10.).to(dist.device))

        mindist = dist.min(axis = 1)
        rec_loss = mindist.values
        if hasattr(self, "supervise_point_category"):
            idx = mindist.indices

        assert (batch.y.squeeze() == cats[mindist.indices].to(batch.y.device)).all()
        
        return rec_loss, idx
    
    def forward(self, batch, mode = "train"):
        if self.supervise_category[mode]:
            reconstructions, x_latent, result = self.model(batch.pos)
            rec_loss, idx = self.compute_indexes(batch, reconstructions)
            return reconstructions, x_latent, result, rec_loss, idx
        else:
            return super(SupervisedTrainer, self).forward(batch, mode = mode)