from collections import OrderedDict
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from scipy.optimize import linear_sum_assignment

import torch

from ..viz import plot_matrix
from .basemetric import DiscreetMetric

class Clustering(DiscreetMetric):
    """
    Compute the following scores:
        - nmi
        - global accuracy
        - mean accuracy
        - accuracy by class
        
    Inspired from https://github.com/monniert/dti-clustering/blob/master/src/utils/metrics.py
    """
    def __init__(self, classes, n_preds = None, *args, **kwargs):
        self.names = ['nmi', 'global_acc', 'avg_acc']
        
        if type(classes) is int:
            self.perclass_names = [f'acc_cls{i}' for i in range(classes)]
            self.n_classes = classes
        else:
            self.perclass_names = [f'acc_{c}' for c in classes]
            self.n_classes = len(classes)
            
        self.n_preds = self.n_classes if n_preds is None else n_preds
        
        self.n_max_labels = max(self.n_classes, self.n_preds)
        
        super(Clustering, self).__init__(*args, **kwargs)
        
    def update_n_preds(self, n_preds = None):
        self.n_preds = self.n_classes if n_preds is None else n_preds
        self.n_max_labels = max(self.n_classes, self.n_preds)

    def compute(self, best_assign_idx = None):
        if self.can_compute():
            nmi = nmi_score(self.true,
                            self.pred,
                            average_method='arithmetic')

            self.proportions = np.bincount(self.pred, minlength = self.n_preds)
            self.proportions = self.proportions / self.proportions.sum()

            self.compute_confusion_matrix(best_assign_idx)
            acc = np.diag(self.matrix).sum() / self.matrix.sum()
            with np.errstate(divide='ignore', invalid='ignore'):
                acc_by_class = np.diag(self.matrix) / self.matrix.sum(axis=1)
            avg_acc = np.mean(np.nan_to_num(acc_by_class))
            self.results = OrderedDict(zip(self.names, [nmi, acc, avg_acc]))
            self.detailed_results = OrderedDict(zip(self.perclass_names, acc_by_class.tolist()))
            self.reset()
    
    def compute_confusion_matrix(self, best_assign_idx_or_indices = None):        
        matrix = np.bincount(self.n_max_labels * self.true + self.pred,
                             minlength=self.n_max_labels**2)
        matrix = matrix.reshape(self.n_max_labels, self.n_max_labels)
        matrix = matrix[:self.n_classes, :self.n_preds]
        
        if self.n_preds == self.n_classes:
            if best_assign_idx_or_indices is not None:
                self.best_assign_idx_or_indices = best_assign_idx_or_indices
            else:
                self.best_assign_idx_or_indices = linear_sum_assignment(-matrix)[1]
            self.matrix = matrix[:, self.best_assign_idx_or_indices]
        else:
            if best_assign_idx_or_indices is not None:
                self.best_assign_idx_or_indices = best_assign_idx_or_indices
            else:
                self.best_assign_idx_or_indices = np.argmax(matrix, axis=0)
            self.matrix = np.vstack([matrix[:, self.best_assign_idx_or_indices == k].sum(axis=1) for k in range(self.n_classes)]).transpose()
            
    def plot(self):
        plot_matrix(self.matrix)
        
if __name__=="__main__":
    
    acc = Clustering(10, 10)
    acc.update(np.random.randint(0, 10, 200),
               np.random.randint(0, 10, 200))
    acc.compute()
    acc.plot()

    print(acc)