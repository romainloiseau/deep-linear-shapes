import torch

class Base3DLoss:
    
    def __call__(self, x, y, reduction = "min"):
        
        reshaped = False
        if x.size()[:-2] != y.size()[:-2]:
            batch_size, n_protos, _, _ = y.size()
            x, y = self.reshape_data(x, y)
            reshaped = True
        
        dist = self.compute_loss(x, y)
        
        if reshaped:
            dist = torch.reshape(dist, (batch_size, n_protos))
            if reduction == "min":
                mindist = dist.min(axis = 1)
                dist = mindist.values
                indices = mindist.indices
                return dist, indices
            
        return dist
    
    def reshape_data(self, x, y):
        assert len(y.size()) == 4
        assert x.size(0) == y.size(0)
            
        batch_size, n_protos, dim, _ = y.size()
            
        x = x.unsqueeze(1).expand(-1, n_protos, -1, -1)
            
        x = torch.reshape(x, (batch_size * n_protos, dim, -1))
        y = torch.reshape(y, (batch_size * n_protos, dim, -1))
            
        return x, y

    def compute_loss(self, x, y):
        raise NotImplementedError
        
    def assignements(self, x, y):
        raise NotImplementedError