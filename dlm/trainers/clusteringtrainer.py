from tqdm.auto import tqdm

import copy

import math

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from torch_geometric.data import Data

from dlm.metrics import Clustering
from dlm.losses import Chamfer
from dlm.viz import print_pc
from dlm.utils import ShapeSampler

import dlm.models.aligners as aligners

from sklearn.decomposition import PCA

import copy

from .basetrainer import BaseTrainer

class ClusteringTrainer(BaseTrainer):
    
    def __init__(self, options, *args, **kwargs):
        super(ClusteringTrainer, self).__init__(options, *args, **kwargs)
            
        self.A_reg = options.A_reg
        self.a_reg = options.a_reg
    
    def initialize_curr_preds(self):
        self.curr_preds = {"assignments": torch.tensor([]),
                           "x_latent": torch.tensor([]),
                           "a": torch.tensor([]),
                           "A": torch.tensor([])}
        
        self.curr_losses = {"train": {"total_loss": torch.tensor([]),
                                      "rec_loss": torch.tensor([]),
                                      "A_reg_loss": torch.tensor([]),
                                      "a_reg_loss": torch.tensor([])}}
        if self.test:
            self.curr_losses["test"] = {"total_loss": torch.tensor([]),
                                        "rec_loss": torch.tensor([]),
                                        "A_reg_loss": torch.tensor([]),
                                        "a_reg_loss": torch.tensor([])}
        if self.val:
            self.curr_losses["val"] = {"total_loss": torch.tensor([]),
                                        "rec_loss": torch.tensor([]),
                                        "A_reg_loss": torch.tensor([]),
                                        "a_reg_loss": torch.tensor([])}
            
    def initialize_materials(self, *args, **kwargs):
        super(ClusteringTrainer, self).initialize_materials(*args, **kwargs)                
        self.define_printable_samples()
            
    def log_curr_preds(self):
        if self.use_tb:
            for k, v in self.curr_preds.items():
                if k == "a":
                    for vv in range(v.size(-1)):
                        self.tbw.add_histogram(f"a/{vv}", v[:, vv],
                                               global_step = self.curr_epoch)
                elif k in ["A"] and self.model.alignment == "Affine":
                    self.tbw.add_histogram(f"{k}/matrix", v[:, :-3],
                                           global_step = self.curr_epoch)
                    self.tbw.add_histogram(f"{k}/bias", v[:, -3:],
                                           global_step = self.curr_epoch)
                elif k in ["A"] and ("Q" in self.model.alignment):
                    self.tbw.add_histogram(f"{k}/quaternion", v[:, :4],
                                           global_step = self.curr_epoch)
                    self.tbw.add_histogram(f"{k}/bias", v[:, 4:7],
                                           global_step = self.curr_epoch)
                    if self.model.alignment in ["Qd", "dQ"]:
                        self.tbw.add_histogram(f"{k}/d", v[:, -1],
                                               global_step = self.curr_epoch)
                    elif self.model.alignment in ["QD", "DQ"]:
                        self.tbw.add_histogram(f"{k}/D", v[:, -3:],
                                               global_step = self.curr_epoch)
                    elif self.model.alignment == "DQD":
                        self.tbw.add_histogram(f"{k}/Dr", v[:, -3:],
                                               global_step = self.curr_epoch)
                        self.tbw.add_histogram(f"{k}/Dl", v[:, -6:-3],
                                               global_step = self.curr_epoch)
                elif k in ["A"] and (self.model.alignment in ["D", "d"]):
                    self.tbw.add_histogram(f"{k}/bias", v[:, -3:],
                                           global_step = self.curr_epoch)
                    if self.model.alignment == "d":
                        self.tbw.add_histogram(f"{k}/d", v[:, 0],
                                               global_step = self.curr_epoch)
                    elif self.model.alignment == "D":
                        self.tbw.add_histogram(f"{k}/D", v[:, :3],
                                               global_step = self.curr_epoch)
                    
                elif k == "assignments":
                    self.tbw.add_histogram(f"clusters/{k}", v,
                                           bins = self.model.n_clusters,
                                           global_step = self.curr_epoch)        
                else:
                    try:
                        self.tbw.add_histogram(k, v,
                                               global_step = self.curr_epoch)
                    except:
                        pass
        
    def initialize_metrics(self):
        self.metrics = {"train": Clustering(
            self.dataset._categories,
            self.model.n_clusters,
            mode = "train"
        )}
        if self.test:
            self.metrics["test"] = Clustering(
                self.dataset._categories,
                self.model.n_clusters,
                mode = "test"
            )
        if self.val:
            self.metrics["val"] = Clustering(
                self.dataset._categories,
                self.model.n_clusters,
                mode = "val"
            )
            
    def initialize_criterion(self):
        self.criterion = Chamfer()
        
    def compute_loss(self, batch, reconstructions, result):
        return self.criterion(
            batch.pos if hasattr(batch, "pos") else batch,
            reconstructions
        )
        
    def forward(self, batch, mode = "train"):
        
        reconstructions, x_latent, result = self.model(
            batch.pos if hasattr(batch, "pos") else batch
        ) 
        
        rec_loss, idx = self.compute_loss(batch, reconstructions, result)
        
        return reconstructions, x_latent, result, rec_loss, idx
        
    def do_prediction(self, batch, mode = "train"):
        reconstructions, x_latent, result, rec_loss, idx = self.forward(batch, mode = mode)
        
        idx_one_hot = F.one_hot(idx, num_classes=reconstructions.size(1))
        
        for k, v in result.items():
            if k in ["a", "A", "a_reg_loss", "A_reg_loss"]:
                result[k] = (v * (idx_one_hot if v.size()==idx_one_hot.size() else idx_one_hot.unsqueeze(-1))).sum(1)
            if k in ["a", "A"]:
                self.model.update_running_parameters(k, result[k], idx)
                    
                    
        result["x_latent"] = x_latent
        result["reconstructions"] = torch.cat([rec[i].unsqueeze(0) for rec, i in zip(reconstructions, idx)], 0)
        
        total_loss = rec_loss
        if self.A_reg:
            total_loss = total_loss + self.A_reg * result["A_reg_loss"]
        if self.a_reg:
            total_loss = total_loss + self.a_reg * result["a_reg_loss"]
        
        result["total_loss"] = total_loss
        result["rec_loss"] = rec_loss
        result["assignments"] = idx
        
        self.metrics[mode].update(batch.y.squeeze(), result["assignments"].cpu())
        return result
    
    def do_epoch(self):
        super(ClusteringTrainer, self).do_epoch()
        self.do_reassign()
        
    def do_activator(self):
        activated = super(ClusteringTrainer, self).do_activator()
        if activated is not None:
            for k in self.metrics.keys():
                self.metrics[k].update_n_preds(self.model.n_clusters)
        
    def do_reassign(self):
        idx_one_hot = F.one_hot(self.curr_preds["assignments"].long(),
                                num_classes=self.model.n_clusters)
        mean_rec_loss = (idx_one_hot * self.curr_losses["train"]["rec_loss"].unsqueeze(1)).mean(0)
        reassigned = self.model.do_reassign(self.metrics["train"].proportions,
                                            optimizer = self.optimizer,
                                            epoch = self.curr_epoch,
                                            mean_rec_loss = mean_rec_loss)
        
        if self.use_tb:
            self.tbw.add_scalar("Train/reassignments", len(reassigned), self.curr_epoch)
                
    def define_printable_samples(self):
        
        samples = ShapeSampler("firsttemplate", max(10, len(self.dataset._categories)))(
            self.dataset.test_dataset[0] if self.test else self.dataset.train_dataset
        )        
        
        printable_samples = {k: torch.cat([s[k].unsqueeze(0) for s in samples], 0) for k in samples[0].keys}
        self.printable_samples = self.prepare_batch(Data(**printable_samples))
        
    def forward_sequential(self, batch):
        self.model.eval()
        with torch.no_grad():
            _, _, _, _, idx = self.forward(batch, mode = "test")
            lsms = [Data(**{k: getattr(self.model.LSMs[i], k).data.detach().clone().cpu() for k in batch.keys if hasattr(self.model.LSMs[i], k)}) for i in idx]
            
            curr_align = self.model.align()
            curr_D = self.model.D()
            aligner = (type(self.model.LSMs[0].Aligner).__name__).replace("Transformer", "")
            sequence = dict()

            self.model.align(False)
            self.model.D(0)
            
            recs, _, _, _, _ = self.forward(batch, mode = "test")
            for sample, (rec, i) in enumerate(zip(recs, idx)):
                lsms[sample].pos = rec[i]
            sequence["Chamfer"] = copy.deepcopy(lsms)
                
            if curr_align:
                self.model.align(True)
                recs, _, _, _, _ = self.forward(batch, mode = "test")
                for sample, (rec, i) in enumerate(zip(recs, idx)):
                    lsms[sample].pos = rec[i]
                sequence[f"{aligner}"] = copy.deepcopy(lsms)
                
            for D in range(1, curr_D + 1):
                self.model.D(D)
                recs, _, _, _, _ = self.forward(batch, mode = "test")
                for sample, (rec, i) in enumerate(zip(recs, idx)):
                    lsms[sample].pos = rec[i]
                sequence[f"{aligner}, D={D}"] = copy.deepcopy(lsms)

            sequence["Input"] = []
            for sample in range(batch.pos.size(0)):
                sequence["Input"].append(Data(**{k: batch[k][sample] for k in batch.keys}))
                
                
            del lsms
        return sequence
                
    def log_greedy_data(self):
        super(ClusteringTrainer, self).log_greedy_data()
        
        if False and self.use_tb:
            self.tbw.add_image("Prototypes",
                               print_pc(self.model.LSMs,
                                        return_as_numpy = True).transpose(2, 0, 1),
                               self.curr_epoch)
            
            
            to_print = self.forward_sequential(self.printable_samples)
            
            titles = list(to_print.keys())
            
            to_print = [to_print[k][i] for i in range(len(to_print[titles[0]])) for k in titles]
            
            self.tbw.add_image("Inference",
                               print_pc(to_print,
                                        max_cols = len(titles),
                                        titles = titles,
                                        return_as_numpy = True).transpose(2, 0, 1),
                               self.curr_epoch)
            
    def compute_features(self, dataloader):
        
        self.softmin = torch.nn.Softmin(dim = -1)
        labels = []
        features = {"assignements": [],
                    "distances": [], 
                    "softd100": [],
                    "latent": [],
                    "fields": [],
                    "alignment": []
                   }
        
        for batch in tqdm(dataloader, desc='Extracting features', leave=False):
            batch = self.prepare_batch(batch)
            
            self.model.eval()
            with torch.no_grad():
                reconstructions, x_latent, result = self.model(batch.pos)
                distances = self.criterion(
                    batch.pos,
                    reconstructions,
                    reduction = None
                ).detach().cpu()
                
                features["assignements"].append(F.one_hot(
                    torch.argmin(distances, axis = -1),
                    num_classes=distances.size(-1)
                ))
                features["distances"].append(distances)
                features["softd100"].append(self.softmin(
                    100 * torch.clamp(distances, 0, 1)
                ))
                features["latent"].append(x_latent.detach().cpu())
                features["fields"].append(result["a"].flatten(start_dim=1).detach().cpu())
                features["alignment"].append(result["A"].flatten(start_dim=1).detach().cpu())
                
                labels.append(batch.y.detach().cpu())
                
        for k in features.keys():
            features[k] = torch.cat(features[k], 0).numpy()
                
        return features, torch.cat(labels, 0).squeeze().detach().cpu().numpy()
        
    def perform_classification(
        self,
        epoch,
        features = ["distances", "softd100", "fields", "softd100+fields", "latent", "latent+random"]
    ):
        
        results = {}
        
        train_all_features, train_labels = self.compute_features(self.dataset.train_dataloader)
        test_all_features, test_labels = self.compute_features(self.dataset.test_dataloaders[0])
        available_labels = np.unique(train_labels)
        
        for use_features in features:
            
            used_features = use_features.split("+")
                
            train_features = np.concatenate([v for k, v in train_all_features.items() if k in used_features], -1)
            test_features = np.concatenate([v for k, v in test_all_features.items() if k in used_features], -1)
            
            metrics = {"n_features": train_features.shape[-1]}
            
            for percentage in tqdm(
                10.**np.array([-2., -1.66, -1.33, -1., -.66, -.33, .0]),
                desc=f'SVM {use_features} ({train_features.shape[-1]} feats.)',
                leave=False):
                
                train_acc = []
                test_acc = []
                for i in tqdm(
                    range(math.ceil((5. if "random" in used_features else 1.) / percentage)),
                    leave = False):
                    
                    choice_available_labels = [np.random.choice(np.where(train_labels == lbl)[0]) for lbl in available_labels]
                    choice = np.random.choice(len(train_features), int(percentage * len(train_features)), replace = False)
                    
                    for chidx, ch in enumerate(choice_available_labels):
                        if ch not in choice[len(choice_available_labels):]:
                            choice[chidx] = ch

                    p_train_features = train_features[choice]
                    p_train_labels = train_labels[choice]
                    assert (available_labels == np.unique(p_train_labels)).all()

                    mini, maxi = np.min(p_train_features, axis = 0), np.max(p_train_features, axis = 0)
                    mini = np.where((maxi - mini) == 0, 0, mini)
                    maxi = np.where((maxi - mini) == 0, 1, maxi)
                    p_train_features = (p_train_features - mini) / (maxi - mini)
                    p_test_features = (test_features - mini) / (maxi - mini)

                    if "random" in used_features:
                        random_features = np.random.choice(p_train_features.shape[-1], 512, replace = False)
                        p_train_features = p_train_features[:, random_features]
                        p_test_features = p_test_features[:, random_features]
                        metrics["n_features"] = p_train_features.shape[-1]

                    clf = svm.LinearSVC(max_iter=10000).fit(p_train_features, p_train_labels)
                    train_acc.append(accuracy_score(p_train_labels, clf.predict(p_train_features)))
                    test_acc.append(accuracy_score(test_labels, clf.predict(p_test_features)))
                    
                metrics[f"train_mean_p{100*percentage:.2f}"] = np.mean(train_acc)
                metrics[f"test_mean_p{100*percentage:.2f}"] = np.mean(test_acc)
                metrics[f"train_std_p{100*percentage:.2f}"] = np.std(train_acc)
                metrics[f"test_std_p{100*percentage:.2f}"] = np.std(test_acc)
                    
            results[use_features] = metrics
        return results
    
    def log_classification(self, results, title = None):
        
        plt.figure(figsize = (6, 4))
        for i, (used_features, result) in enumerate(results.items()):
            
            ptrain = np.array([])
            acc_train = np.array([])
            acc_test = np.array([])
            acc_std_train = np.array([])
            acc_std_test = np.array([])
            
            n_features = result["n_features"]
            
            for k, v in result.items():
                if "train_mean" in k:
                    ptrain = np.append(ptrain, float(k.split("p")[-1]))
                    acc_train = np.append(acc_train, v)
                    acc_test = np.append(acc_test, result[k.replace("train_mean", "test_mean")])
                    acc_std_train = np.append(acc_std_train, result[k.replace("train_mean", "train_std")])
                    acc_std_test = np.append(acc_std_test, result[k.replace("train_mean", "test_std")])
                    
            plt.plot(ptrain, 100*acc_test,
                     marker='o', alpha = .75,
                     color=plt.cm.hsv(i / len(results)),
                     label = "{} ({} feat.)   {:.3f}".format(
                         used_features,
                         n_features,
                         100*acc_test[ptrain == 100][0]))
            plt.errorbar(ptrain,
                         100*acc_test,
                         100*acc_std_test,
                         alpha = .75,
                         color=plt.cm.hsv(i / len(results)),
                         linestyle='None', marker='')
        plt.grid()
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.xscale("log")
        plt.xlabel("Data percentage usage (%)")
        plt.ylabel("Test accuracy")
        if title is not None:
            plt.title(title)
        plt.show()