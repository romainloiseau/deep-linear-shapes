import os

import argparse
import json

from ..global_variables import AVAILABLE_DATASETS

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def get_parser():
    parser = argparse.ArgumentParser(description='Runs the method proposed in "Representing Shape Collections with Alignment-Aware Linear Models".')
    
    parser.add_argument('--fix_seed', default=False, type=str2bool, help='wether to fix the seed or not')
    parser.add_argument('--name', default=None, type=str, help='name of the run')
                        
    #Dataset
    parser.add_argument('--directory',
                        default = os.path.join(__file__.split("dlm/")[0], "../../Datasets"),
                        type=str, help='data directory')
    parser.add_argument('--dataset', default="modelnet10", type=str,
                        choices = AVAILABLE_DATASETS)
    
    parser.add_argument('--n_points', default=1024, type=int, help='number of points for point clouds')
    parser.add_argument('--train_data', default=1, type=float,
                        help='<1 stands for a proportion of the train dataset,\
                        1 stands for the train dataset, 2 stands for train+test data,\
                        3 stands for train+test+val data')
    parser.add_argument('--test_data', default=1, type=float,
                        help='<1 stands for a proportion of the test dataset,\
                        1 stands for the test dataset, 2 stands for train+test data,\
                        3 stands for train+test+val data')
    parser.add_argument('--categories', default=[], type=str, nargs="+", help='loaded categories')
    
    #Training
    parser.add_argument('--use_tensorboard', default=True, type=str2bool)
    parser.add_argument('--log_every_n_epochs', default=1, type=int)
    
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=4, type=int, help='dataloader workers')
    parser.add_argument('--data_sampling', default="random", type=str,
                        choices=['random', 'sequential', 'weighted'])
    
    parser.add_argument('--n_epochs', default=1000, type=int)
    
    parser.add_argument('--lr', default=.001, type=float)
    parser.add_argument('--opt', default="Adam", type=str,
                        choices = ["Adam"])
    
    #DTI3D
    parser.add_argument('--initialisation', default="random", type=str,
                        choices = ["random", "template", "kmeans++"])
    parser.add_argument('--K', default=10, type=int)
    parser.add_argument('--alignment', default="Affine", type=str,
                        choices = ["Affine", "Q", "Qd", "dQ", "QD", "DQ", "DQD", "d", "D"])
    parser.add_argument('--d', default=6, type=int,
                        help="The size of the encoder's bottleneck")
    
    parser.add_argument('--D_parametric', default=5, type=int,
                        help="The parametric dimensions of the model")
    parser.add_argument('--archi_parametric', default=[128, 128, 128],
                        nargs = "+", type=int)
    parser.add_argument('--D_pointwise', default=0, type=int,
                        help="The pointwise dimensions of the model")
    
    parser.add_argument('--load', default=None, type=str, help="pretrained model loaded at model creation")
    
    #Curriculum
    parser.add_argument('--activator',
                        default=["auto_time", 10, "auto_tol", 0.0001],
                        nargs = "+", type=str)
    
    #Training
    parser.add_argument('--A_reg', default=0, type=float)
    parser.add_argument('--a_reg', default=0, type=float)
    parser.add_argument('--warming_steps', default=49, type=int)
    parser.add_argument('--warming_intensity', default=.1, type=float)
    parser.add_argument('--empty_cluster_threshold', default=.2, type=float)
    
    return parser