import time

import numpy as np

import copy

import torch
from torch import nn

from tqdm.auto import tqdm

from dlm.losses import SplittedChamfer
from dlm.metrics import IoU

from .supervisedtrainer import SupervisedTrainer

class SegmentationTrainer(SupervisedTrainer):
    
    def __init__(self, *args, **kwargs):
        super(SegmentationTrainer, self).__init__(*args, **kwargs)
        
        self.supervise_category = {"train": True, "test": True, "val": True}
        self.supervise_point_category = {"train": False, "test": False, "val": False}
        
    def log_greedy_data(self):
        if not self.supervise_point_category["train"]:
            self.annotate_model()
        super(SegmentationTrainer, self).log_greedy_data()
        
    def initialize_criterion(self):
        self.criterion = SplittedChamfer()
        
    def initialize_metrics(self):
        super(SegmentationTrainer, self).initialize_metrics()
        self.iou = {"train": IoU(self.dataset.train_dataset.seg_classes, mode = "train")}
        if self.test:
            self.iou["test"] = IoU(self.dataset.test_dataset[0].seg_classes, mode = "test")
        if self.val:
            self.iou["val"] = IoU(self.dataset.val_dataset.seg_classes, mode = "val")
            
    def annotate_model(self, voting_strategy = "random"):
        
        assert voting_strategy in ["all", "best", "random"], 'Voting strategy should be in [all", "best", "random"]'
        
        self.model.eval()
        with torch.no_grad():
            
            segmentations = {f"{i}": [] for i in range(self.model.n_clusters)}
            distances = {f"{i}": [] for i in range(self.model.n_clusters)}
        
            for batch in tqdm(self.dataset.train_dataloader, desc='Annotating prototypes', leave=False):
        
                batch = self.prepare_batch(batch)
                reconstructions, _, result = self.model(
                    batch.pos if hasattr(batch, "pos") else batch
                )
                
                rec_loss, idx = self.compute_indexes(batch, reconstructions)
                
                reconstructions = torch.cat([rec[i].unsqueeze(0) for rec, i in zip(reconstructions, idx)], 0)
                
                if torch.cuda.is_available():
                    batch.point_y = batch.point_y.cuda()
                    
                indices = self.criterion.assignements(batch.pos, reconstructions, ref="y")
                for cluster, distance, assignments, labels in zip(idx, rec_loss, indices, batch.point_y):
                    segmentation = torch.gather(labels,
                                                0,
                                                assignments.long())
                    segmentations[f"{cluster.item()}"].append(segmentation.detach().cpu().unsqueeze(0))
                    distances[f"{cluster.item()}"].append(distance.item())
                    
        segmentations = {f"{i}": torch.cat(segmentations[f"{i}"]) if len(segmentations[f"{i}"]) !=0 else None for i in range(self.model.n_clusters)}
        
        if voting_strategy == "random":
            few = max(1, int(10 / self.model.n_clusters))
            segmentations = {f"{i}": segmentations[f"{i}"][np.random.choice(segmentations[f"{i}"].size(0), few, replace = (segmentations[f"{i}"].size(0) <= few))] if segmentations[f"{i}"] is not None else None for i in range(self.model.n_clusters)}
            
        elif voting_strategy in "best":
            distances = {f"{i}": torch.cat(distances[f"{i}"]) for i in range(self.model.n_clusters)}
            segmentations = {f"{i}": segmentations[f"{i}"][torch.argmin(distances[f"{i}"]).item()].unsqueeze(0) if segmentations[f"{i}"] is not None else None for i in range(self.model.n_clusters)}      
            
        classes = torch.unique(torch.cat([torch.unique(segmentations[f"{i}"]) if segmentations[f"{i}"] is not None else torch.tensor([]) for i in range(self.model.n_clusters)])).int()
        for i, classe in enumerate(classes):
            for cluster in range(self.model.n_clusters):
                if segmentations[f"{cluster}"] is not None:
                    segmentations[f"{cluster}"][segmentations[f"{cluster}"] == classe] = i
                    
        for cluster in tqdm(range(self.model.n_clusters), desc='Aggregating annotations', leave=False):
            
            if segmentations[f"{cluster}"] is not None:
                one_hot_segmentations = torch.nn.functional.one_hot(segmentations[f"{cluster}"], num_classes=classes.size(0))
                one_hot_segmentations = one_hot_segmentations.sum(axis = 0)
                segmentations[f"aggregated_one_hot_{cluster}"] = torch.argmax(one_hot_segmentations, dim = 1)

                segmentations[f"aggregated_{cluster}"] = segmentations[f"aggregated_one_hot_{cluster}"].clone()

                for i, classe in enumerate(classes):
                    segmentations[f"aggregated_{cluster}"][segmentations[f"aggregated_one_hot_{cluster}"] == i] = classe.item()

                self.model.LSMs[cluster].point_y.data = segmentations[f"aggregated_{cluster}"].to(self.model.LSMs[cluster].point_y.data.device)
                
    def forward(self, batch, mode = "train"):
        
        if self.supervise_point_category[mode]:
            
            recs, x_latent, result = self.model(batch.pos)
                
            splited_recs = recs.clone()
            splited_pos = batch.pos.clone()
            
            idx_recs = torch.cat([(self.model.LSMs.point_ys == l).unsqueeze(0) for l in self.iou[mode].seg_label_to_cat.keys()], 0).to(batch.pos.device)
            idx_pos = torch.cat([(batch.point_y == l).unsqueeze(0) for l in self.iou[mode].seg_label_to_cat.keys()], 0).to(batch.pos.device)
            
            for l, (ip, ir) in enumerate(zip(idx_pos, idx_recs)):
                splited_recs = torch.where(ir.unsqueeze(0).unsqueeze(2),
                                           l * 10. + splited_recs,
                                           splited_recs)
                splited_pos = torch.where(ip.unsqueeze(1),
                                          l * 10. + splited_pos,
                                          splited_pos)
                
            cats = self.model.LSMs.ys
            cats_take = (cats == batch.y.to(cats.device)).detach().to(recs.device)
            assert cats_take.any(axis = -1).all(), "Should have at least one lsm per categories"
                
            splited_dist = self.criterion(splited_pos, splited_recs, reduction = None)
            splited_dist = torch.where(cats_take,
                                       splited_dist,
                                       torch.tensor(10.).to(splited_dist.device))
            min_splited_dist = splited_dist.min(axis = -1)
            
            dist = self.criterion(batch.pos, recs, reduction = None)
            dist = torch.where(cats_take, dist, torch.tensor(10.).to(dist.device))
            mindist = dist.min(axis = 1)
            idx = mindist.indices
            
            splited_take = min_splited_dist.values < 5
            rec_loss = torch.where(splited_take,
                                   min_splited_dist.values,
                                   mindist.values)
            
            assert rec_loss.max().item() < 5
            assert (batch.y.squeeze() == cats[mindist.indices].to(batch.y.device)).all()
            
            return recs, x_latent, result, rec_loss, idx
        
        else:
            return super(SegmentationTrainer, self).forward(batch, mode = mode)
            
    def do_prediction(self, batch, mode = "train"):
        result = super(SegmentationTrainer, self).do_prediction(batch, mode)
        with torch.no_grad():
            indices = self.criterion.assignements(batch.pos, result["reconstructions"])
            
        segmentations = []
        for i, best_lsm in enumerate(result["assignments"]):
            segmentation = torch.gather(self.model.LSMs[best_lsm].point_y,
                                        0,
                                        indices[i].long())
            segmentations.append(segmentation.detach().cpu())
            
        self.iou[mode].update(batch.point_y, segmentations)
        
        return result        
    
    def log_scalars(self, mode = "train"):
        super(SegmentationTrainer, self).log_scalars(mode)
        
        for k, v in self.iou[mode].log(
            self.curr_epoch,
            self.tbw if self.use_tb else None
        ).items():
            self.hparams_metrics[k] = copy.deepcopy(v)
            
class SupervisedSegmentationTrainer(SegmentationTrainer):
    
    def __init__(self, *args, **kwargs):
        super(SupervisedSegmentationTrainer, self).__init__(*args, **kwargs)
        
        self.supervise_category = {"train": True, "test": True, "val": True}
        self.supervise_point_category = {"train": True, "test": False, "val": False}