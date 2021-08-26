import os

import numpy as np
from tqdm.auto import tqdm

import torch
from torch import nn

from ..utils import ShapeSampler

from .basemodel import BaseModel
from dlm.models import models3D
from .decoders import MLP
from .linearshapemodel import LinearShapeModel, LinearShapeModels
import dlm.models.aligners as aligners

from dlm.viz import print_pc

from ..global_variables import DECODER_INIT_MEAN, DECODER_INIT_STD

class DTI3D(BaseModel):
    
    def __init__(self, init, args):
        super(DTI3D, self).__init__()
        
        self.d = args.d
        self.D_max = args.D_parametric + args.D_pointwise
        self.alignment = args.alignment
        self.n_points = args.n_points
        self.empty_cluster_threshold = args.empty_cluster_threshold
        
        self.encoder = models3D.PointNetfeat(
                d = self.d,
                global_feat=True,
                stn_transform = False,
                feature_transform=False
            )
        
        self.encoder_out = 2**(self.d+4)
        self.decoders_layers = [self.encoder_out,  2**(self.d+2), 2**(self.d+1)]
        
        self.LSMs = LinearShapeModels([
            LinearShapeModel(
                self.n_points if init is None else init[k],
                args.D_parametric, args.D_pointwise,
                self.alignment,
                self.decoders_layers,
                args.archi_parametric
            ) for k in range(args.K)
        ])
            
        self.keep_cat = False
        
        if args.load is not None:
            if os.path.exists(args.load):
                self.load(args.load)
            else:
                self.load(args.load.format("last_epoch"))
       
    @property
    def n_clusters(self):
        return self.K()
        
    def forward_latent(self, x_latent):
        
        preds = [lsm(x_latent) for lsm in self.LSMs]
        reconstructions = torch.cat([pred[0].unsqueeze(1) for pred in preds], 1)
        
        params = dict()
        for k in preds[0][1].keys():
            params[k] = torch.cat([pred[1][k].unsqueeze(1) for pred in preds], 1)
            
        return reconstructions, x_latent, params
    
    def forward(self, x):
        return self.forward_latent(self.encoder(x)[0])
        
    def do_reassign(self, proportions, threshold = None, optimizer = None, epoch = None, mean_rec_loss = None):
        
        threshold = threshold if threshold is not None else self.empty_cluster_threshold
        
        proportions = np.array(list(proportions) + [1.] * (self.n_clusters - len(proportions)))
        if mean_rec_loss is not None:
            mean_rec_loss = mean_rec_loss.detach().cpu().numpy()**4
            mean_rec_loss[proportions < threshold / self.n_clusters] = 0
            mean_rec_loss = mean_rec_loss / mean_rec_loss.sum()
        
        reassigned = []
        for i, p in enumerate(proportions):
            if p < threshold / self.n_clusters:
                
                if self.keep_cat and hasattr(self.LSMs[i], "y") and (self.LSMs[i].y == self.LSMs.ys).sum().item() <= 1:
                    tqdm.write("Epoch {} - Empty lsm {} is the last from category {}".format(epoch, i, self.LSMs[i].y))
                else:
                    if mean_rec_loss is not None:
                        newi = np.random.choice(
                            len(proportions), 1,
                            p = mean_rec_loss if not self.keep_cat else mean_rec_loss * (self.LSMs[i].y == self.LSMs.ys).cpu().detach().numpy() / (mean_rec_loss * (self.LSMs[i].y == self.LSMs.ys).cpu().detach().numpy()).sum())[0]
                    else:
                        newi = np.random.choice(
                            len(proportions), 1,
                            p = proportions if not self.keep_cat else proportions * (self.LSMs[i].y == self.LSMs.ys).cpu().detach().numpy() / (proportions * (self.LSMs[i].y == self.LSMs.ys).cpu().detach().numpy()).sum())[0]
                    
                    self.LSMs[i].copy(self.LSMs[newi], optimizer = optimizer)
                    reassigned.append(newi)
                    
                    tqdm.write("Epoch {} - Reassigning lsm {} from lsm {}".format(epoch, i, newi))
            
        if len(reassigned) > 0:
            used = np.unique(reassigned)
            for u in used:
                self.LSMs[u].copy(self.LSMs[u])
            
        return reassigned
    
    def reinitialize(self, dataset, optimizer):
        
        initialization = ShapeSampler("random", len(self.LSMs))(dataset)
        
        for i, init in enumerate(initialization):
            self.LSMs[i].copy(init)
        
    def train_pos(self, b: bool = None, optimizer = None):
        if b is None:
            return self.LSMs[0].pos.requires_grad
        for lsm in self.LSMs:
            lsm.pos.requires_grad = b
            if not b:
                if hasattr(lsm.pos, "grad"):
                    del lsm.pos.grad
                if optimizer is not None:
                    if lsm.pos in optimizer.state:
                        del optimizer.state[lsm.pos]
    
    def align(self, b: bool = None, optimizer = None):
        if b is None:
            return self.LSMs[0].align
        if b and hasattr(self, "Aencoder") and self.train_global_align:
            self.freeze_global_align(optimizer)
        for i in range(self.n_clusters):
            self.LSMs[i].align = b
            
    def D(self, D: int = None):
        if D is None:
            return self.LSMs[0].D
                    
        for i in range(self.n_clusters):
            self.LSMs[i].D = D
            
    def K(self, K: int = None, dataset = None, criterion = None):
        if K is None:
            return len(self.LSMs)
        
        distances = []
        self.eval()
        with torch.no_grad():
            batch = []
            for i in tqdm(range(len(dataset)), desc=f"Adding sample", leave=False):
                batch.append(dataset[i].pos.unsqueeze(0))
                if len(batch) >= 32:
                    batch = torch.cat(batch).cuda().permute(0, 2, 1)                    
                    reconstructions, _, result = self(batch)
                    rec_loss, _ = criterion(
                        batch,
                        reconstructions
                    )
                    distances.append(rec_loss**2)
                    batch = []
            if len(batch[0]) >= 1:
                batch = torch.cat(batch).cuda().permute(0, 2, 1)
                reconstructions, _, result = self(batch)
                rec_loss, _ = criterion(
                    batch,
                    reconstructions
                )
                distances.append(rec_loss**2)
                batch = []
         
            distances = torch.cat(distances).detach().cpu().numpy()
            
        for _ in range(K):
            choice = int(np.random.choice(len(dataset), 1, p = distances / distances.sum())[0])
            self.LSMs.append(LinearShapeModel(
                dataset[choice],
                self.D_max,
                self.alignment,
                self.decoders_layers,
                bayesian = self.bayesian
            ).cuda())
            
    def update_running_parameters(self, k, params, idx):
        for i in range(len(self.LSMs)):
            self.LSMs[i].update_running_parameters(k, params[idx == i])
            
    def show(self):
        print_pc(self.LSMs, titles = [f"Linear shape model {i}" for i in range(len(self.LSMs))])