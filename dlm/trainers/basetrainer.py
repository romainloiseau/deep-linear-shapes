import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm

from dlm.viz import HTMLGenerator, print_gif, print_deformed_gif

class BaseTrainer:
    
    def __init__(
        self,
        args,
        name = "training"
    ):
        
        self.task = None
        
        self.name = os.path.join("runs", name + "_{}")
        i = 0
        while os.path.exists(self.name.format(i)):
            i += 1
        self.name = self.name.format(i)   
        
        self.use_tb = args.use_tensorboard
        self.log_every_n_epochs = args.log_every_n_epochs
        
        self.curr_epoch = 0
        
        hparams = ["opt", "lr", "batch_size", "num_workers", "data_sampling", "a_reg", "A_reg", "warming_steps", "warming_intensity"]
        self.hparams = {k: getattr(args, k) for k in hparams}
        self.hparams_metrics = {}
            
    def initialize_materials(self, model, dataset, activator = None):
        
        self.model = model
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        self.dataset = dataset
        self.initialize_dataloaders()
        self.test = hasattr(self.dataset, "test_dataset") and self.dataset.test_dataset is not None
        self.val = hasattr(self.dataset, "val_dataset") and self.dataset.val_dataset is not None
        
        if activator is not None:
            self.activator = activator
            
            self.activator.warm(self.hparams["warming_intensity"], self.hparams["warming_steps"])
            
            self.activator.deactivate(self.model)
            self.hparams["activated"] = self.activator.remember_activations
        
        self.initialize_optimizer()
        self.initialize_criterion()
        self.initialize_metrics()
        
    def initialize_dataloaders(self):
        if not hasattr(self.dataset, "train_dataloader"):
            
            if self.hparams["data_sampling"] == "weighted":
                y = self.dataset.train_dataset.data.y
                p = 1. / torch.unique(y, return_counts = True)[1][y].float()
                self.dataset.train_sampler = torch.utils.data.WeightedRandomSampler(
                    weights = p / p.sum(),
                    num_samples = len(p),
                    replacement = True
                )
                shuffle = False
            elif self.hparams["data_sampling"] == "random":
                shuffle = True
            elif self.hparams["data_sampling"] == "sequential": 
                shuffle = False
            else:
                raise NotImplementedError(
                    "Parameter 'data_sampling' should be in ['weighted', 'random', 'sequential'].\
                    Actually:{}".format(self.hparams["data_sampling"])
                )
                
            self.dataset.create_dataloaders(
                self.model, 
                batch_size=self.hparams["batch_size"],
                shuffle=shuffle,
                num_workers=self.hparams["num_workers"],
                precompute_multi_scale=False
            )
        
    def initialize_optimizer(self):
        
        parameters = [
                {"params": [], "name": "base"},
            ] + [
                {"params": lsm.pos, "name": f"pos_{i}"} for i, lsm in enumerate(self.model.LSMs)
            ] + [
                {"params": self.model.encoder.parameters(),
                 #"lr": self.hparams["lr"]*self.hparams["warming_intensity"],
                 "name": "encoder"}
            ] + [
                {"params": lsm.ANet.parameters(),
                 #"lr": self.hparams["lr"]*self.hparams["warming_intensity"],
                 "name": f"ANet_{i}"} for i, lsm in enumerate(self.model.LSMs)
            ] + [
                {"params": lsm.fields.parameters(),
                 #"lr": self.hparams["lr"]*self.hparams["warming_intensity"],
                 "name": f"fields_{i}"} for i, lsm in enumerate(self.model.LSMs)
            ] + [
                {"params": lsm.PNet.parameters(),
                 #"lr": self.hparams["lr"]*self.hparams["warming_intensity"],
                 "name": f"PNet_{i}"} for i, lsm in enumerate(self.model.LSMs)
            ]
        
        self.optimizer = getattr(optim, self.hparams["opt"])(
            parameters,
            lr = self.hparams["lr"]
        )
        
    def initialize_writers(self):
        if self.use_tb:            
            if hasattr(self.dataset, "hparams"):
                self.hparams.update(self.dataset.hparams)
            if hasattr(self.model, "hparams"):
                self.hparams.update(self.model.hparams)
            
            if not hasattr(self, "tbw"):
                self.tbw = SummaryWriter(self.name)
                self.model.describe(self.tbw)
            
    def initialize_metrics(self):
        raise NotImplementedError
            
    def initialize_criterion(self):
        raise NotImplementedError
            
    def prepare_batch(self, batch):        
        if hasattr(batch, "pos"):
            batch.pos = batch.pos.permute(0, 2, 1)
            if torch.cuda.is_available():
                batch.pos = batch.pos.cuda()
        if hasattr(batch, "keypoints"):
            batch.keypoints = batch.keypoints.permute(0, 2, 1)
            if torch.cuda.is_available():
                batch.keypoints = batch.keypoints.cuda()
        if hasattr(batch, "im"):
            if torch.cuda.is_available():
                batch.im = batch.im.cuda()
        return batch
    
    def save_checkpoint(self, name = None):
        name = "{}/{}.pt".format(
            self.name,
            name if name is not None else "last_epoch")
        
        torch.save({'epoch': self.curr_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                   name)        
        
    def log_scalars(self, mode = "train"):
        if mode == "train":
            curr_empty_cluster_thresh = self.model.empty_cluster_threshold
            name = "Train/empty_cluster_threshold"
            if self.use_tb:
                self.tbw.add_scalar(name, curr_empty_cluster_thresh, self.curr_epoch)
            else:
                tqdm.write(name + " "*(25 - len(name)) + f'{curr_empty_cluster_thresh:.5f}')
            self.hparams["empty_cluster_thresh"] = curr_empty_cluster_thresh
            
        for k, v in self.metrics[mode].log(
            self.curr_epoch,
            self.tbw if self.use_tb else None,
            best_assign_idx = self.metrics["train"].best_assign_idx_or_indices if (hasattr(self.metrics["train"], "best_assign_idx_or_indices") and mode != "train") else None
        ).items():
            self.hparams_metrics[k] = v
        
        for k, v in self.curr_losses[mode].items():
            name = f'Loss/{k}/{mode}'
            value = v.mean().item()
            if self.use_tb:
                self.tbw.add_scalar(name, value, self.curr_epoch)
            else:
                tqdm.write(name + " "*(25 - len(name)) + f'{value:.5f}')
                
            self.hparams_metrics[name] = value
            
        if mode == "train":
            curr_lr = self.optimizer.param_groups[0]["lr"]
            name = "Train/lr"
            if self.use_tb:
                self.tbw.add_scalar(name, curr_lr, self.curr_epoch)
            else:
                tqdm.write(name + " "*(25 - len(name)) + f'{curr_lr:.5f}')
            self.hparams["lr"] = curr_lr
        
    def log_curr_preds(self):
        if self.use_tb:
            for k, v in self.curr_preds.items():
                self.tbw.add_histogram(k, v, global_step = self.curr_epoch)
        
    def log_greedy_data(self):
        self.save_checkpoint()
        self.log_curr_preds()     
                
    def log_epoch(self, mode = "train"):
        self.log_scalars(mode)
        
        if mode == "train":
            if (self.curr_epoch%(10*self.log_every_n_epochs) == 0):
                self.log_greedy_data()
        
    def log_checkpoint(self, name = None):
        self.save_checkpoint(name = name)
        
        if self.use_tb:
            self.tbw.add_hparams(self.hparams, self.hparams_metrics)
            
    def do_prediction(self, clouds):
        raise NotImplementedError
            
    def do_batch(self, batch, batch_idx = 0, mode = "train"):
        batch = self.prepare_batch(batch)
        
        if mode == "train":
            self.optimizer.zero_grad()            
        prediction = self.do_prediction(batch, mode = mode)
        if mode == "train":
            prediction["total_loss"].mean().backward()       
            self.optimizer.step()

            for k in self.curr_preds.keys():
                if k in prediction.keys() and (len(prediction[k].size()) == 1 or batch_idx * self.hparams["batch_size"] < 512):
                    self.curr_preds[k] = torch.cat([self.curr_preds[k], prediction[k].detach().cpu()], 0)
                    
        for k in self.curr_losses[mode].keys():
            self.curr_losses[mode][k] = torch.cat([self.curr_losses[mode][k], prediction[k].detach().cpu()], 0)
                                    
    def do_epoch_train(self):
        self.model.train()
        for batch_idx, batch in enumerate(tqdm(self.dataset.train_dataloader, desc='Train', leave=False)):
            self.do_batch(batch, batch_idx = batch_idx)
        self.log_epoch()
        
    def do_epoch_test(self):
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataset.test_dataloaders[0], desc='Test', leave=False)):
                self.do_batch(batch, mode = "test")            
            self.log_epoch("test")
            
    def do_epoch_val(self):
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataset.val_dataloader, desc='Val', leave=False)):
                self.do_batch(batch, mode = "val")            
            self.log_epoch("val")
            
    def initialize_curr_preds(self):
        self.curr_preds = {}
        self.curr_losses = {"train": {"total_loss": torch.tensor([])}}
        if self.test:
            self.curr_losses["test"] = {"total_loss": torch.tensor([])}
        if self.val:
            self.curr_losses["val"] = {"total_loss": torch.tensor([])}
        
    def do_epoch(self):        
        self.initialize_curr_preds()
        self.do_epoch_train()
        if self.test and (self.curr_epoch%self.log_every_n_epochs == 0):
            self.do_epoch_test()
        if self.val and (self.curr_epoch%self.log_every_n_epochs == 0):
            self.do_epoch_val()
        self.curr_epoch = self.curr_epoch + 1
        self.do_scheduler()
        self.do_activator()
       
    def do_scheduler(self):
        if hasattr(self, "scheduler"):
            self.scheduler.step()
            
    def do_activator(self):
        if hasattr(self, "activator"):
            activated, loss_decrease = self.activator(
                self.model, self.dataset.train_dataset, self.optimizer, self.criterion, self.curr_epoch, self.curr_losses["train"]["total_loss"].mean().item()
            )
            if activated is not None:
                self.log_checkpoint(name = f"best_{activated}")
                self.hparams["activated"] = self.activator.remember_activations
            
            name = "Train/100*(1-loss_decrease)"
            if self.use_tb:
                self.tbw.add_scalar(name, 100*(1-loss_decrease), self.curr_epoch-1)
            else:
                tqdm.write(name + " "*(25 - len(name)) + f'{100*(1-loss_decrease):.5f}')
            
            return activated, loss_decrease
        return

    def train(self, n_epochs = 100):
        self.n_epochs = n_epochs       
        self.initialize_writers()
        for _ in tqdm(range(self.n_epochs), desc="Epoch"):
            self.do_epoch()
            
        self.log_checkpoint(name = "last")
        
    def log_examples(self, path = None):
        
        NINFERENCE = 8
        EXAMPLES_NAMES = ["Random", "Best", "Worst"]
        EXAMPLES_FUNCT = [lambda x, xx: False, lambda x, xx: x < max(xx), lambda x, xx: x > min(xx)]
        EXAMPLES_CHOICE = [lambda xx: 0, lambda xx: np.argmax(xx), lambda xx: np.argmin(xx)]
        
        if path is None:
            path = self.name.replace("runs/", "")
        path = os.path.join("web", path)
        if not os.path.exists("web"):
            os.mkdir("web")
        if not os.path.exists(path):
            os.mkdir(path)
        
        html = HTMLGenerator(path + ".html", self.name, datetime.now().strftime("%d/%m/%Y - %H:%M:%S"))
        cmap = plt.viridis() if not self.task == "segmentation" else plt.get_cmap("hsv")
        
        results = {
            "assignments": [],
            "rec_loss": [],
            "a": [],
            "A": [],
            "reconstructions": []
        }
        
        examples = {}
        for k in EXAMPLES_NAMES:
            examples[k] = {n: [] for n in range(self.model.n_clusters)}
        
        dataloader = self.dataset.test_dataloaders[0] if (hasattr(self.dataset, "test_dataset") and (self.dataset.test_dataset is not None) and len(self.dataset.test_dataset[0]) > 32) else self.dataset.train_dataloader
        for i, batch in enumerate(tqdm(dataloader)):
            batch = self.prepare_batch(batch)
            self.model.eval()
            with torch.no_grad():
                result = self.do_prediction(batch)
                
            for k in results.keys():
                results[k].append(result[k].detach().cpu())
            
            for sample, assigned in enumerate(result["assignments"]):
                cat = self.dataset._categories[batch.y[sample]]
                
                for k, f, g in zip(EXAMPLES_NAMES, EXAMPLES_FUNCT, EXAMPLES_CHOICE):
                    if len(examples[k][assigned.item()]) < NINFERENCE:
                        examples[k][assigned.item()].append({"input": batch.pos[sample].detach().cpu(),
                                                             "reconstruction": result["reconstructions"][sample].detach().cpu(),#, assigned.item()
                                                             "category": cat,
                                                             "chamfer": result["rec_loss"][sample],
                                                             "image": batch.im[sample].detach().cpu() if hasattr(batch, "im") else None})
                        
                    else:
                        _ = [eg["chamfer"].detach().cpu().numpy() for eg in examples[k][assigned.item()]]
                        
                        if f(result["rec_loss"][sample].detach().cpu().numpy(), _):
                            
                            examples[k][assigned.item()][g(_)] = {"input": batch.pos[sample].detach().cpu(),
                                                                  "reconstruction": result["reconstructions"][sample].detach().cpu(),#, assigned.item()
                                                                  "category": cat,
                                                                  "chamfer": result["rec_loss"][sample],
                                                                  "image": batch.im[sample].detach().cpu() if hasattr(batch, "im") else None}
                            
                            
        
        
        
        for k in results.keys():
            results[k] = torch.cat(results[k], 0)
            
        return results
                            
        notempty, assignements = torch.unique(results["assignments"], return_counts = True)
        notempty = notempty.numpy()
        assignements = assignements.numpy()
        
        paramhtml = []
        corrected_clusters = []
        corrected_fields = []
        for k, v in results.items():
            v = v.numpy()
            
            if k == "A":
                
                mean_transfo = []
                Aligner = self.model.LSMs[0].Aligner
                
                for p in range(self.model.n_clusters):
                    mean_transfo.append(torch.tensor(v[results["assignments"] == p, :].mean(0)).unsqueeze(0))
                    
                    corrected_clusters.append(Aligner(self.model.LSMs[p].pos.unsqueeze(0),
                                                      mean_transfo[p].to(self.model.LSMs[p].pos.device))[0][0].detach().cpu().numpy())
                    
                    if "a" in results.keys():
                        corrected_fields.append(Aligner(self.model.LSMs[p].get_fields,
                                                        mean_transfo[p].expand(self.model.LSMs[p].fields.D_parametric + self.model.LSMs[p].fields.D_pointwise, -1).to(self.model.LSMs[p].pos.device))[0].detach().cpu().numpy())
                        
                mean_transfo = torch.cat(mean_transfo, 0)
        
                v = {"matrix": v[:-3],
                     "bias": v[-3:]}
            
            elif k == "a":
                a_percentiles = dict()
                
                for f in range(v.shape[1]):
                    for p in range(self.model.n_clusters):
                            try:
                                a_percentiles[f"p{p}_f{f}"] = (
                                    np.percentile(v[results["assignments"] == p, f].flatten(), 5) - v[results["assignments"] == p, f].flatten().mean(),
                                    np.percentile(v[results["assignments"] == p, f].flatten(), 50) - v[results["assignments"] == p, f].flatten().mean(),
                                    np.percentile(v[results["assignments"] == p, f].flatten(), 95) - v[results["assignments"] == p, f].flatten().mean()
                                )
                            except:
                                a_percentiles[f"p{p}_f{f}"] = (-1, 0, 1)
                                
                v = {f'{v.shape[1]} Vector basis': v}
            elif k in ["assignments", "rec_loss"]:
                v = {k: v}
            else:
                tqdm.write(f"IMPLEMENT PARAMETER VIZ FOR {k}")
                    
                    
            if len(v) > 4:
                n_rows = int(np.ceil(len(v)/4))
                n_cols = 4
            else:
                n_rows = 1
                n_cols = len(v)
                    
            parampath = os.path.join(path, f"predicted_parameter_{k}")
            paramhtmlpath = parampath.replace("web/", "")
                
            if not os.path.exists(f"{parampath}.png"):
                fig = plt.figure(figsize = (4 * n_cols, 3 * n_rows))
                for i, (kk, vv) in enumerate(v.items()):
                    ax = fig.add_subplot(n_rows, n_cols, i + 1)
                    ax.hist(vv.flatten(), density=True, bins = 50, label = f"Mean: {vv.mean():.8f}\nStd:  {vv.std():.8f}")
                    ax.legend()
                    ax.set_title(kk)
                fig.tight_layout()
                plt.savefig(parampath)
                plt.close(fig)
                
                
            paramhtml.append(f'\t<h{2}>\n')
            paramhtml.append(f'\t\t{k}\n')
            paramhtml.append(f'\t</h{2}>\n')
            paramhtml.append(f"\t<img src='{paramhtmlpath}.png' alt='predicted parameter {k}'>\n")
                
        if len(corrected_clusters) == 0:
            corrected_clusters = [self.model.LSMs[p].pos.detach().cpu().numpy() for p in range(self.model.n_clusters)]
            
        if paramhtml != []:
            html.add_to_body("Predicted parameters", "".join(paramhtml))
            
            
        phtml = []
        count = 0
        phtml.append(f'\t<h{2}>\n')
        phtml.append(f'\t\tPrototypes from 0 to {self.model.n_clusters - 1}\n')
        phtml.append(f'\t</h{2}>\n')                
        phtml.append("<div class=hgif>\n")
        for ii, (n_reconstructed, i) in enumerate(zip(
            tqdm(np.sort(assignements)[::-1], leave = False),
            np.take_along_axis(notempty, np.argsort(assignements)[::-1], axis = 0)
        )):
            
            p = self.model.LSMs[i]
            
            ppath = os.path.join(path, f"prototype_{i}")
            phtmlpath = ppath.replace("web/", "")
            
            if not os.path.exists(f"{ppath}.gif"):
                
                if hasattr(p, "segmentation"):
                    colorscale = (p.segmentation.numpy() - np.min(p.segmentation.numpy())) / 5.
                else:
                    colorscale = None
                    
                print_gif(corrected_clusters[i] + (np.array([a_percentiles[f"p{i}_f{f}"][1] * corrected_fields[i][f] for f in range(p.D)])).sum(0),
                          name = ppath, frames = 8, colorscale = colorscale, cmap = cmap)
                
            phtml.append(f"\t<figure><img src='{phtmlpath}.gif' alt='prototype {i}'><figcaption>prototype {i}</figcaption></figure>\n")
                
            count += 1
            if count >= 5:
                count = 0
                phtml.append("</div>\n")
                phtml.append("<div class=hgif>\n")
                    
        if phtml[-1] != "</div>\n":
            phtml.append("</div>\n")
            
        html.add_to_body("Prototypes", "".join(phtml))                    
            
        phtml = []
        count = 0
        num_proto_to_print = 32
        for ii, (n_reconstructed, i) in enumerate(zip(
            tqdm(np.sort(assignements)[::-1], leave = False),
            np.take_along_axis(notempty, np.argsort(assignements)[::-1], axis = 0)
        )):
            
            p = self.model.LSMs[i]
            
            if hasattr(p, "segmentation"):
                colorscale = (p.segmentation.cpu().detach().numpy() - np.min(p.segmentation.cpu().detach().numpy())) / 5.
            else:
                colorscale = None
            
            if ii < num_proto_to_print:
                phtml.append(f'\t<h{2}>\n')
                n_reconstructed_proportion = 100*n_reconstructed/len(results["assignments"])
                phtml.append(f'\t\tPrototype {i}\t\t\treconstructs {n_reconstructed_proportion:.2f}% shapes\n')
                phtml.append(f'\t</h{2}>\n')

                phtml.append("<div class=hgif>\n")
            elif ii == num_proto_to_print:
                phtml.append(f'\t<h{2}>\n')
                phtml.append(f'\t\tPrototypes from {ii} to {self.model.n_clusters - 1}\n')
                phtml.append(f'\t</h{2}>\n')
                
                phtml.append("<div class=hgif>\n")
            
            
            ppath = os.path.join(path, f"prototype_{i}")
            phtmlpath = ppath.replace("web/", "")
            
            assert os.path.exists(f"{ppath}.gif")
            if ii < num_proto_to_print:
                phtml.append(f"\t<img src='{phtmlpath}.gif' alt='prototype {i}'>\n")
                phtml.append("</div>\n")
            else:
                phtml.append(f"\t<figure><img src='{phtmlpath}.gif' alt='prototype {i}'><figcaption>prototype {i}</figcaption></figure>\n")
                
                count += 1
                if count >= 5:
                    count = 0
                    phtml.append("</div>\n")
                    phtml.append("<div class=hgif>\n")
            
            if ii < num_proto_to_print and p.D > 0:
                count = 0
                phtml.append(f'\t<h{3}>\n')
                phtml.append(f'\tVector basis\n')
                phtml.append(f'\t</h{3}>\n')
                phtml.append("<div class=hgif>\n")

                for f in range(p.D):
                    if not os.path.exists(f"{ppath}_field_{f}.gif"):
                    
                        print_deformed_gif(corrected_clusters[i],
                                           corrected_fields[i][f],
                                           name =  f"{ppath}_field_{f}",
                                           scales = np.linspace(a_percentiles[f"p{i}_f{f}"][0],
                                                                a_percentiles[f"p{i}_f{f}"][2],
                                                                8),
                                           cmap = cmap)
                    phtml.append(f"\t<img src='{phtmlpath}_field_{f}.gif' alt='prototype {i} morpho {f}'>\n")
                    count += 1
                    if count >= 5:
                        count = 0
                        phtml.append("</div>\n")
                        phtml.append("<div class=hgif>\n")

                phtml.append("</div>\n")
                
                phtml.append("<div class=hgif>\n")

                for f in range(p.D):
                    if not os.path.exists(f"{ppath}_field_{f}_intensity.png"):
                        fig = plt.figure(figsize = (2, 2))
                        ax = fig.add_subplot(1, 1, 1)
                        ax.hist(results["a"][results["assignments"] == i, f].detach().cpu().numpy().flatten(), density=True, bins = 25)
                        ax.set_title(f"field {f} intensities")
                        fig.tight_layout()
                        plt.savefig(f"{ppath}_field_{f}_intensity.png")
                        plt.close(fig)
                        
                    phtml.append(f"\t<img src='{phtmlpath}_field_{f}_intensity.png' alt='prototype {i} field {f} intensity'>\n")
                    count += 1
                    if count >= 5:
                        count = 0
                        phtml.append("</div>\n")
                        phtml.append("<div class=hgif>\n")
                        
                phtml.append("</div>\n")
                
            if ii < num_proto_to_print:
                
                for name in EXAMPLES_NAMES:
                
                    phtml.append(f'\t<h{3}>\n')
                    phtml.append(f'\t{name} reconstructions\n')
                    phtml.append(f'\t</h{3}>\n')

                    for sample, pair in enumerate(examples[name][i]):
                        phtml.append("<div class=hgif>\n")

                        if hasattr(pair, "im"):
                            bpath = os.path.join(path, f"{name}_sample_image_{sample}_p{i}.png")
                            bhtmlpath = bpath.replace("web/", "")
                            if not os.path.exists(bpath):
                                im = Image.fromarray(pair["image"].transpose(1, 2, 0), 'RGB')
                                im.save(bpath)
                            phtml.append(f"\t<figure><img src='{bhtmlpath}' alt='{name} sample image {sample}'><figcaption>{name} sample image {sample}</figcaption></figure>\n")


                        bpath = os.path.join(path, f"prototype_{i}")
                        bhtmlpath = bpath.replace("web/", "")
                        phtml.append(f"\t<figure><img src='{bhtmlpath}.gif' alt='prototype {i}'><figcaption>prototype {i}</figcaption></figure>\n")

                        bpath = os.path.join(path, f"{name}_{sample}_p{i}")
                        bhtmlpath = bpath.replace("web/", "")
                        if not os.path.exists(f"{bpath}.gif"):
                            print_gif(pair["input"].cpu().detach().numpy(),
                                         name = bpath,
                                         frames = 8)
                        cat = pair["category"]
                        phtml.append(f"\t<figure><img src='{bhtmlpath}.gif' alt='{name} {sample}'><figcaption>{name} {sample} - {cat}</figcaption></figure>\n")

                        bpath = os.path.join(path, f"deformed_{name}_{sample}_p{i}")
                        bhtmlpath = bpath.replace("web/", "")
                        if not os.path.exists(f"{bpath}.gif"):
                            print_gif(pair["reconstruction"].cpu().detach().numpy(),
                                         name = bpath,
                                         frames = 8,
                                         colorscale = colorscale, cmap = cmap)
                        chamfer = pair["chamfer"]
                        phtml.append(f"\t<figure><img src='{bhtmlpath}.gif' alt='deformed {sample} from prototype {i}'><figcaption>deformed: chamfer = {int(100000*chamfer)/100.}</figcaption></figure>\n")
                        phtml.append("</div>\n")
                
        if phtml[-1] != "</div>\n":
            phtml.append("</div>\n")
            
        html.add_to_body("Prototypes and reconstructions", "".join(phtml))                
                
        html.return_html()