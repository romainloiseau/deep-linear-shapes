import torch
from torch import nn

class BaseAligner(nn.Module):

    def __init__(self, add_identity = True, add_bias = True):
        super(BaseAligner, self).__init__()
        self.n_parameters = None
        self.add_bias = add_bias
        self.add_identity = add_identity
        
        if self.add_bias:
            self.bias_slice = None
        
    def get_inverse_matrix(self):
        raise NotImplementedError
        
    def get_transformation(self, parameters):
        
        transformation = {"matrix": self.get_matrix(parameters)}
        if self.add_bias:
            transformation["bias"] = parameters[:, self.bias_slice].contiguous()
            
        transformation["reg"] = torch.norm(
            torch.bmm(
                transformation["matrix"],
                torch.transpose(transformation["matrix"], 1, 2)
            ) - torch.eye(transformation["matrix"].size(-1)).unsqueeze(0).to(parameters.device),
            dim = (1, 2)
        )
        return transformation
    
    def forward_dim(self, x, parameters, dim, verbose = False):
        transformation = self.get_transformation(parameters)
        
        x = torch.matmul(transformation["matrix"].unsqueeze(dim), x)
        if self.add_bias:
            x = x + transformation["bias"].unsqueeze(2).unsqueeze(dim)
        return  x, transformation["reg"]
    
    def forward(self, x, parameters, verbose = False):
        transformation = self.get_transformation(parameters)
        x = torch.bmm(transformation["matrix"], x)
        
        if self.add_bias:
            x = x + transformation["bias"].unsqueeze(2)
        return  x, transformation["reg"]
        
class AffineAligner(BaseAligner):
    """Affine transformation"""
    
    def __init__(self, add_identity = True, add_bias = True, dim = 3):
        super(AffineAligner, self).__init__(add_identity, add_bias)
        
        self.dim = dim
        self.n_parameters = dim**2
        if self.add_bias:
            self.bias_slice = slice(-self.dim, None)
            self.n_parameters += dim
            
    def get_inverse_matrix(self, parameters):
        return torch.inverse(self.get_matrix(parameters))
            
    def get_matrix(self, parameters):
        batch_size, n_parameters = parameters.size()
        
        assert n_parameters == self.n_parameters
        
        matrix = parameters[:, :self.dim*self.dim].contiguous()
        matrix = matrix.view(batch_size, self.dim, self.dim).contiguous()
        if self.add_identity:
            matrix = matrix + torch.eye(self.dim).unsqueeze(0).to(matrix.device)
        return matrix
    
class dAligner(BaseAligner):
    def __init__(self, add_identity = True, add_bias = True):
        super(dAligner, self).__init__(add_identity, add_bias)
        self.n_parameters = 1
        
        if self.add_bias:
            self.bias_slice = slice(1, 4)
            self.n_parameters += 3
            
    def get_matrix(self, parameters):
        d = parameters[:, 0].contiguous()
        if self.add_identity:
            d = 1. + d
        return torch.eye(3).unsqueeze(0).to(parameters.device) * d.unsqueeze(-1).unsqueeze(-1)
    
class DAligner(BaseAligner):
    def __init__(self, add_identity = True, add_bias = True):
        super(DAligner, self).__init__(add_identity, add_bias)
        self.n_parameters = 3
        
        if self.add_bias:
            self.bias_slice = slice(3, 6)
            self.n_parameters += 3
            
    def get_matrix(self, parameters):
        D = parameters[:, :3].contiguous()
        if self.add_identity:
            D = 1. + D
        return torch.eye(3).unsqueeze(0).to(parameters.device) * D.unsqueeze(-1)
    
class D6Aligner(BaseAligner):
    def __init__(self, add_identity = True, add_bias = True):
        super(D6Aligner, self).__init__(add_identity, add_bias)
        self.n_parameters = 6
        self.epsilon = 10**(-10)
        
        if self.add_bias:
            self.bias_slice = slice(6, 9)
            self.n_parameters += 3
            
    def get_inverse_matrix(self, parameters):
        return self.get_rotation(parameters).permute(0, 2, 1)
    
    def get_matrix(self, parameters):
        return self.get_rotation(parameters)
    
    def get_rotation(self, parameters):
        
        # batch*n
        def normalize_vector( v, return_mag =False):
            batch=v.shape[0]
            v_mag = torch.sqrt(v.pow(2).sum(1))# batch
            v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
            v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
            v = v/v_mag
            if(return_mag==True):
                return v, v_mag[:,0]
            else:
                return v
            
        # u, v batch*n
        def cross_product( u, v):
            batch = u.shape[0]
            i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
            j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
            k = u[:,0]*v[:,1] - u[:,1]*v[:,0]

            out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3

            return out
        
        x_raw = parameters[:, 0:3].contiguous()#batch*3
        y_raw = parameters[:, 3:6].contiguous()#batch*3
        
        if self.add_identity:
            x_raw[:, 0] = 1. + x_raw[:, 0]
            y_raw[:, 1] = 1. + y_raw[:, 1]

        x = normalize_vector(x_raw) #batch*3
        z = cross_product(x,y_raw) #batch*3
        z = normalize_vector(z)#batch*3
        y = cross_product(z,x)#batch*3

        x = x.view(-1,3,1)
        y = y.view(-1,3,1)
        z = z.view(-1,3,1)
        return torch.cat((x,y,z), 2) #batch*3*3

class QAligner(BaseAligner):
    def __init__(self, add_identity = True, add_bias = True):
        super(QAligner, self).__init__(add_identity, add_bias)
        self.n_parameters = 4
        self.epsilon = 10**(-10)
        
        if self.add_bias:
            self.bias_slice = slice(4, 7)
            self.n_parameters += 3
            
    def get_inverse_matrix(self, parameters):
        return self.get_rotation(parameters).permute(0, 2, 1)
    
    def get_matrix(self, parameters):
        return self.get_rotation(parameters)
    
    def get_rotation(self, parameters):
        
        matrix_params = parameters[:, :4].contiguous()
        
        if self.add_identity:
            matrix_params[:, 0] = 1. + matrix_params[:, 0]
            
        norm = torch.sqrt((matrix_params * matrix_params).sum(axis = -1))
        
        matrix_params = (matrix_params / (norm.unsqueeze(-1) + self.epsilon)).permute(1, 0)
        
        t2 =   matrix_params[0]*matrix_params[1]
        t3 =   matrix_params[0]*matrix_params[2]
        t4 =   matrix_params[0]*matrix_params[3]
        t5 =  -matrix_params[1]*matrix_params[1]
        t6 =   matrix_params[1]*matrix_params[2]
        t7 =   matrix_params[1]*matrix_params[3]
        t8 =  -matrix_params[2]*matrix_params[2]
        t9 =   matrix_params[2]*matrix_params[3]
        t10 = -matrix_params[3]*matrix_params[3]        
        
        return torch.eye(3).unsqueeze(0).to(matrix_params.device) + 2 * torch.cat([
            torch.cat([
                (t8 + t10).unsqueeze(0),
                (t6 - t4).unsqueeze(0),
                (t3 + t7).unsqueeze(0)
                      ], 0).unsqueeze(0),
            torch.cat([
                (t4 + t6).unsqueeze(0),
                (t5 + t10).unsqueeze(0),
                (t9 - t2).unsqueeze(0)
                      ], 0).unsqueeze(0),
            torch.cat([
                (t7 - t3).unsqueeze(0),
                (t2 + t9).unsqueeze(0),
                (t5 + t8).unsqueeze(0)
                      ], 0).unsqueeze(0),
        ], 0).permute(2, 1, 0)
    
class QzAligner(QAligner):
    
    def get_rotation(self, parameters):
        parameters[:, 1] = 0.
        parameters[:, 3] = 0.
        return super(QzAligner, self).get_rotation(parameters)
    
class QdAligner(QAligner):
    
    def __init__(self, *args, **kwargs):
        super(QdAligner, self).__init__(*args, **kwargs)
        self.n_parameters += 1
        
    def get_d(self, parameters):
        d = parameters[:, -1].contiguous()
        if self.add_identity:
            d = 1. + d
        return d.unsqueeze(-1).unsqueeze(-1)
    
    def get_inverse_matrix(self, parameters):
        return (1. / self.get_d(parameters)) * self.get_rotation(parameters).permute(0, 2, 1)
    
    def get_matrix(self, parameters):
        return self.get_d(parameters) * self.get_rotation(parameters)
    
class dQAligner(QdAligner):
    def __init__(self, *args, **kwargs):
        super(dQAligner, self).__init__(*args, **kwargs)
        
class QDAligner(QAligner):
    
    def __init__(self, *args, **kwargs):
        super(QDAligner, self).__init__(*args, **kwargs)
        self.n_parameters += 3
        
    def get_D(self, parameters):
        D = parameters[:, -3:].contiguous()
        if self.add_identity:
            D = 1. + D
        return D
    
    def get_inverse_matrix(self, parameters):
        return (1. / self.get_D(parameters).unsqueeze(-1)) * self.get_rotation(parameters).permute(0, 2, 1)
    
    def get_matrix(self, parameters):
        return self.get_D(parameters).unsqueeze(-2) * self.get_rotation(parameters)
    
class DQAligner(QDAligner):
    def __init__(self, *args, **kwargs):
        super(DQAligner, self).__init__(*args, **kwargs)
        
    def get_inverse_matrix(self, parameters):
        return (1. / self.get_D(parameters).unsqueeze(-2)) * self.get_rotation(parameters).permute(0, 2, 1)
    
    def get_matrix(self, parameters):
        return self.get_D(parameters).unsqueeze(-1) * self.get_rotation(parameters)
    
class DQDAligner(QAligner):
    
    def __init__(self, *args, **kwargs):
        super(DQDAligner, self).__init__(*args, **kwargs)
        self.n_parameters += 6
        
    def get_Dr(self, parameters):
        D = parameters[:, -3:].contiguous()
        if self.add_identity:
            D = 1. + D
        return D
    
    def get_Dl(self, parameters):
        D = parameters[:, -6:-3].contiguous()
        if self.add_identity:
            D = 1. + D
        return D
    
    def get_inverse_matrix(self, parameters):
        return (1. / self.get_Dl(parameters).unsqueeze(-2)) * (self.get_rotation(parameters).permute(0, 2, 1) * (1. / self.get_Dr(parameters).unsqueeze(-1)))
    
    def get_matrix(self, parameters):        
        return self.get_Dl(parameters).unsqueeze(-1) * (self.get_rotation(parameters) * self.get_Dr(parameters).unsqueeze(-2))