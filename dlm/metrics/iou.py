import numpy as np
from collections import OrderedDict

import torch

from .basemetric import DiscreetMetric

class IoU(DiscreetMetric):
    
    def __init__(self, seg_classes, *args, **kwargs):
        
        self.seg_labels = []
        self.seg_classes = {}        
        self.seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
        for cat in seg_classes.keys():
            self.seg_classes[cat] = seg_classes[cat]
            for label in self.seg_classes[cat]:
                self.seg_label_to_cat[label] = cat
                self.seg_labels.append(label)
                
        self.min_label = min(self.seg_labels)
        self.num_labels = max(self.seg_labels) - self.min_label + 1
        
        self.names = ['global_iou', 'avg_iou']
        self.perclass_names = [f'iou_{cat}' for cat in self.seg_classes.keys()]
        
        super(IoU, self).__init__(*args, **kwargs)
        
    def update(self, true, pred):
        self.true += list(true)
        self.pred += list(pred)
        
    def reset(self):
        self.true = []
        self.pred = []
        
    def compute(self):
        
        shape_ious = {cat:[] for cat in self.seg_classes.keys()}
        pred_iou = []
        
        
        for i in range(len(self.true)):            
            segp = self.pred[i]
            segl = self.true[i] 
            cat = self.seg_label_to_cat[segl[0].item()]
            part_ious = [0.0 for _ in range(len(self.seg_classes[cat]))]
        
            for l in self.seg_classes[cat]:               
                if (segl==l).sum() == 0 and (segp==l).sum() == 0:
                    part_ious[l-self.seg_classes[cat][0]] = torch.tensor(1.0)
                else:
                    part_ious[l-self.seg_classes[cat][0]] = ((segl==l) & (segp==l)).sum() / float(((segl==l) | (segp==l)).sum())
                    
                    
            part_ious = np.mean(part_ious)
            shape_ious[cat].append(part_ious)
            pred_iou.append(part_ious)
            
        with np.errstate(divide='ignore', invalid='ignore'):
            results = [np.mean(cat_ious) if cat_ious != [] else 0. for _, cat_ious in shape_ious.items()]
        self.results = OrderedDict(zip(self.names,
                                       [np.mean(pred_iou), np.mean(results)]))
        self.detailed_results = OrderedDict(zip(self.perclass_names, results))
        self.reset()
        
if __name__=="__main__":
    
    iou = IoU({"Airplane": [0, 1, 2, 3], "Chair": [4, 5, 6]})
    iou.update(np.random.randint(0, 4, (16, 32)),
               np.random.randint(0, 4, (16, 32)))
    iou.update(np.random.randint(4, 7, (8, 32)),
               np.random.randint(4, 7, (8, 32)))
    iou.compute()