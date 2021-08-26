from collections import OrderedDict
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from scipy.optimize import linear_sum_assignment

import torch

from ..viz import plot_matrix
from .basemetric import DiscreetMetric

class Accuracy(DiscreetMetric):
    """
    Compute the following scores:
        - global accuracy
        - mean accuracy
        - accuracy by class
    """
    def __init__(self, classes, *args, **kwargs):
        self.names = ['global_acc', 'avg_acc']
        
        if type(classes) is int:
            self.perclass_names = [f'acc_cls{i}' for i in range(classes)]
            self.n_classes = classes
        elif type(classes) is list:
            self.perclass_names = [f'acc_{c}' for c in classes]
            self.n_classes = len(classes)
            
        super(Accuracy, self).__init__(*args, **kwargs)

    def compute(self, best_assign_idx = None):
        if self.can_compute():
            self.compute_confusion_matrix()
            acc = np.diag(self.matrix).sum() / self.matrix.sum()

            with np.errstate(divide='ignore', invalid='ignore'):
                acc_by_class = np.diag(self.matrix) / self.matrix.sum(axis=1)
            avg_acc = np.mean(np.nan_to_num(acc_by_class))
            self.results = OrderedDict(zip(self.names, [acc, avg_acc]))
            self.detailed_results = OrderedDict(zip(self.perclass_names, acc_by_class.tolist()))
            self.reset()
    
    def compute_confusion_matrix(self):
        matrix = np.bincount(self.n_classes * self.true + self.pred,
                             minlength=self.n_classes**2)
        self.matrix = matrix.reshape(self.n_classes, self.n_classes)
            
    def plot(self):
        plot_matrix(self.matrix)
        
if __name__=="__main__":
    
    acc = Accuracy(10, 10)
    acc.update(np.random.randint(0, 10, 200),
               np.random.randint(0, 10, 200))
    acc.compute()
    acc.plot()

    print(acc)