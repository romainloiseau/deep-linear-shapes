from tqdm.auto import tqdm

class ActivatorDTI3D:
    
    def __init__(self, args = None):
        
        if args is None:
            activations = {"align": 100}
            for i in range(10):
                activations[f"field{i}"] = 100
        else:
            activations = args.activator
                
        if type(activations) == list:
            activations = {activations[2 * i]: int(activations[2 * i + 1]) for i in range(int(len(activations) / 2))}
            
        print("Activator ", activations)
                        
        self.activations = activations
        self.remember_activations = "id"
        
        self.lowloss = 10**10
        
        self.auto = False
        if "auto_time" in self.activations.keys():
            assert len(self.activations.keys()) == 2 and "auto_tol" in self.activations.keys()
            self.auto = True
            self.counter = 0
            self.order = ["align"] + [f"field{i}" for i in range(1, 1 + args.D_pointwise + args.D_parametric)]
        
    def warm(self, intensity, steps):
        self.smooth = {"n_step": steps, "intensity": intensity,
                       "align": steps, "encoder": steps, "field": steps}
        
    def __call__(self, model, dataset, optimizer, criterion, epoch = 0, loss = 0):
        old_remember_activation = None
        
        self.do_smooth(optimizer, epoch)
        loss_decrease = loss/self.lowloss
        
        activatek = False
        if self.auto:
            if loss < (1. - self.activations["auto_tol"])*self.lowloss:
                self.lowloss = loss
                self.counter = 0
            else:
                self.counter += 1
                
            if len(self.order) == 0:
                return old_remember_activation, loss_decrease
            
            if self.counter == self.activations["auto_time"]:
                self.counter = 0
                k = self.order[0]
                del self.order[0]
                activatek = True
                
            
        else:
            if loss < self.lowloss:
                self.lowloss = loss
                
            if len(self.activations) == 0:
                return old_remember_activation, loss_decrease
        
            k = list(self.activations.keys())[0]
            self.activations[k] -= 1           

            if self.activations[k] == 0:
                del self.activations[k]
                activatek = True
                
        if activatek:
            self.activate(model, dataset, optimizer, criterion, k, epoch)
            
            model.empty_cluster_threshold /= 10.
            old_remember_activation = self.remember_activations
                
            for spec in ["field"]:
                if (spec in k) and (spec in old_remember_activation):
                    self.remember_activations = "+".join([ora for ora in old_remember_activation.split("+") if spec not in ora])
            self.remember_activations = "+".join([self.remember_activations, k]) if self.remember_activations!="" else k
        
        return old_remember_activation, loss_decrease
            
    def do_smooth(self, optimizer, epoch = 0):
                    
        if "align" in self.remember_activations and self.smooth["align"] > 0:
            written = False
            for param_group in optimizer.param_groups:
                
                if "ANet_" in param_group["name"]:
                    if self.smooth["align"] == self.smooth["n_step"]:
                        param_group['lr'] = optimizer.param_groups[0]["lr"]*self.smooth["intensity"]
                    param_group['lr'] += (1 - self.smooth["intensity"]) * optimizer.param_groups[0]["lr"] / self.smooth["n_step"]
                    if not written:
                        tqdm.write(f"Epoch {epoch}: warming up {param_group['name']} to lr={param_group['lr']:.6f}")
                        written = True
                elif "encoder" in param_group["name"] and self.smooth["encoder"] > 0:
                    if self.smooth["encoder"] == self.smooth["n_step"]:
                        param_group['lr'] = optimizer.param_groups[0]["lr"]*self.smooth["intensity"]
                    self.smooth["encoder"] -= 1
                    param_group['lr'] += (1 - self.smooth["intensity"]) * optimizer.param_groups[0]["lr"] / self.smooth["n_step"]
                    tqdm.write(f"Epoch {epoch}: warming up {param_group['name']} to lr={param_group['lr']:.6f}")
            self.smooth["align"] -= 1
                    
        elif "field" in self.remember_activations and self.smooth["field"] > 0:
            written = False
            for param_group in optimizer.param_groups:
                
                if "fields_" in param_group["name"] or "PNet_" in param_group["name"]:
                    if self.smooth["field"] == self.smooth["n_step"]:
                        param_group['lr'] = optimizer.param_groups[0]["lr"]*self.smooth["intensity"]
                    param_group['lr'] += (1 - self.smooth["intensity"]) * optimizer.param_groups[0]["lr"] / self.smooth["n_step"]
                    if not written:
                        tqdm.write(f"Epoch {epoch}: warming up {param_group['name']} to lr={param_group['lr']:.6f}")
                        written = True
                elif "encoder" in param_group["name"] and self.smooth["encoder"] > 0:
                    if self.smooth["encoder"] == self.smooth["n_step"]:
                        param_group['lr'] = optimizer.param_groups[0]["lr"]*self.smooth["intensity"]
                    self.smooth["encoder"] -= 1
                    param_group['lr'] += (1 - self.smooth["intensity"]) * optimizer.param_groups[0]["lr"] / self.smooth["n_step"]
                    tqdm.write(f"Epoch {epoch}: warming up {param_group['name']} to lr={param_group['lr']:.6f}")
            self.smooth["field"] -= 1
        
    def activate(self, model, dataset, optimizer, criterion, k, epoch = 0):
        tqdm.write(f"Activating {k} at epoch {epoch + 1}")
        
        if k == "align":
            model.align(True, optimizer)
        elif "field" in k:
            model.D(int(k.replace("field", "")))
            
    def deactivate(self, model):
        if len(self.activations) != 0:
            if sum(["field" in act for act in self.activations]) != 0 or self.auto:
                model.D(0)
            if sum(["align" == act for act in self.activations]) != 0 or self.auto:
                model.align(False)