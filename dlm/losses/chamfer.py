import torch
import numpy as np

from ChamferDistancePytorch.chamfer3D import dist_chamfer_3D as dc3D

from .base import Base3DLoss

class Chamfer(Base3DLoss):
    
    def __init__(self):
        super(Chamfer, self).__init__()
        self.chamfer_func = dc3D.chamfer_3DDist()
        
    def chamfer(self, x, y):
        assert len(x.size()) == 3
        
        if x.size(-1) != 3:
            x = x.permute(0, 2, 1)
            y = y.permute(0, 2, 1)
            
        assert x.size(-1) == 3
            
        return self.chamfer_func(x, y)
    
    def compute_loss(self, x, y):
        dist1, dist2, idx1, idx2 = self.chamfer(x, y)
        return dist1.mean(-1) + dist2.mean(-1)
    
    def assignements(self, x, y, ref = "x"):
        _, _, idx1, idx2 = self.chamfer(x, y)
        
        if ref in ["x", 0]:
            return idx1
        elif ref in ["y", 1]:
            return idx2
        else:
            raise ValueError(f"ref ({ref}) argument sould be in {'x', 'y', 0, 1}")
            
class ZAxedChamfer(Chamfer):
    
    def __init__(self, prec = 90):
        super(ZAxedChamfer, self).__init__()
        
        rotations = [torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).float()]

        matrix = torch.tensor([
            [np.cos(prec*np.pi/180.), 0, -np.sin(prec*np.pi/180.)],
            [0, 1, 0], 
            [np.sin(prec*np.pi/180.), 0, np.cos(prec*np.pi/180.)]
        ]).float()

        for _ in range(int(360/prec)-1):
            rotations.append(
                torch.matmul(
                    rotations[-1],
                    matrix
                )
            )
            
        self.rotations = rotations
    
    def compute_loss(self, x, y):
        if x.size(-1) != 3:
            x = x.permute(0, 2, 1)
            y = y.permute(0, 2, 1)
            
        return torch.stack([
            super(ZAxedChamfer, self).compute_loss(x, torch.matmul(y, r)) for r in torch.stack(self.rotations).float().to(y.device)
        ]).min(0)[0]
    
class AxedChamfer(ZAxedChamfer):
    
    def __init__(self, prec = 90):
        super(AxedChamfer, self).__init__(prec = prec)
        
        q = [torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).float()]
        
        matrix = torch.tensor([
            [1, 0, 0],
            [0, np.cos(prec*np.pi/180.), -np.sin(prec*np.pi/180.)],
            [0, np.sin(prec*np.pi/180.), np.cos(prec*np.pi/180.)]
        ]).float()
    
        for _ in range(int(90/prec)):
            q.append(
                torch.matmul(
                    q[-1],
                    matrix
                )
            )
            
        rotations = []
        
        for r in self.rotations:
            for qq in range(len(q)):
                if qq == 0:
                    rotations.append(r)
                else:
                    rotations.append(torch.matmul(q[qq], r))
                    rotations.append(torch.matmul(-q[qq], r))
                
        self.rotations = rotations        
            
class SplittedChamfer(Chamfer):
    
    def compute_loss(self, x, y):
        dist1, dist2, _, _ = self.chamfer(x, y)
        dist = dist1 + dist2
        return torch.where((dist.max(-1)[0] > 5),
                           torch.tensor(10.).to(dist.device),
                           dist.mean(-1))
        