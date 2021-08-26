import argparse
from . import my_utils as my_utils
import os
import datetime
import json
from termcolor import colored
from easydict import EasyDict
from os.path import exists, join

"""
    Author : Thibault Groueix 01.11.2019
"""


def parser():
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument("--no_learning", action="store_true", help="Learning mode (batchnorms...)")
    parser.add_argument("--train_only_encoder", action="store_true", help="only train the encoder")
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--batch_size_test', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--nepoch', type=int, default=150, help='number of epochs to train for')
    parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to train for')
    parser.add_argument("--random_seed", action="store_true", help="Fix random seed or not")
    parser.add_argument('--lrate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_1', type=int, default=120, help='learning rate decay 1')
    parser.add_argument('--lr_decay_2', type=int, default=140, help='learning rate decay 2')
    parser.add_argument('--lr_decay_3', type=int, default=145, help='learning rate decay 2')
    parser.add_argument("--run_single_eval", action="store_true", help="evaluate a trained network")
    parser.add_argument("--demo", action="store_true", help="run demo autoencoder or single-view")

    # Data
    parser.add_argument('--normalization', type=str, default="UnitBall",
                        choices=['UnitBall', 'BoundingBox', 'Identity'])
    parser.add_argument("--shapenet13", action="store_true", help="Load 13 usual shapenet categories")
    parser.add_argument("--SVR", action="store_true", help="Single_view Reconstruction")
    parser.add_argument("--sample", action="store_false", help="Sample the input pointclouds")
    parser.add_argument('--class_choice', nargs='+', default=["airplane"], type=str)
    parser.add_argument('--number_points', type=int, default=2500, help='Number of point sampled on the object during training, and generated by atlasnet')
    parser.add_argument('--number_points_eval', type=int, default=2500,
                        help='Number of points generated by atlasnet (rounded to the nearest squared number) ')
    parser.add_argument("--random_rotation", action="store_true", help="apply data augmentation : random rotation")
    parser.add_argument("--data_augmentation_axis_rotation", action="store_true",
                        help="apply data augmentation : axial rotation ")
    parser.add_argument("--data_augmentation_random_flips", action="store_true",
                        help="apply data augmentation : random flips")
    parser.add_argument("--random_translation", action="store_true",
                        help="apply data augmentation :  random translation ")
    parser.add_argument("--anisotropic_scaling", action="store_true",
                        help="apply data augmentation : anisotropic scaling")

    # Save dirs and reload
    parser.add_argument('--id', type=str, default="0", help='training name')
    parser.add_argument('--env', type=str, default="Atlasnet", help='visdom environment')
    parser.add_argument('--visdom_port', type=int, default=8890, help="visdom port")
    parser.add_argument('--http_port', type=int, default=8891, help="http port")
    parser.add_argument('--dir_name', type=str, default="", help='name of the log folder.')
    parser.add_argument('--demo_input_path', type=str, default="./doc/pictures/plane_input_demo.png", help='dirname')
    parser.add_argument('--reload_decoder_path', type=str, default="", help='dirname')
    parser.add_argument('--reload_model_path', type=str, default='', help='optional reload model path')

    # Network
    parser.add_argument('--num_layers', type=int, default=2, help='number of hidden MLP Layer')
    parser.add_argument('--hidden_neurons', type=int, default=512, help='number of neurons in each hidden layer')
    parser.add_argument('--loop_per_epoch', type=int, default=1, help='number of data loop per epoch')
    parser.add_argument('--nb_primitives', type=int, default=1, help='number of primitives')
    parser.add_argument('--template_type', type=str, default="SPHERE", choices=["SPHERE", "SQUARE"],
                        help='dim_out_patch')
    parser.add_argument('--multi_gpu', nargs='+', type=int, default=[0], help='Use multiple gpus')
    parser.add_argument("--remove_all_batchNorms", action="store_true", help="Replace all batchnorms by identity")
    parser.add_argument('--bottleneck_size', type=int, default=1024, help='dim_out_patch')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=["relu", "sigmoid", "softplus", "logsigmoid", "softsign", "tanh"], help='dim_out_patch')

    # Loss
    parser.add_argument("--no_metro", action="store_true", help="Compute metro distance")

    opt = parser.parse_args()

    opt.date = str(datetime.datetime.now())
    now = datetime.datetime.now()
    opt = EasyDict(opt.__dict__)

    if opt.dir_name == "":
        # Create default dirname
        opt.dir_name = join('log', opt.id + now.isoformat())


    # If running a demo, check if input is an image or a pointcloud
    if opt.demo:
        ext = opt.demo_input_path.split('.')[-1]
        if ext == "ply" or ext == "npy" or ext == "obj":
            opt.SVR = False
        elif ext == "png":
            opt.SVR = True

    if opt.demo or opt.run_single_eval:
        if not exists("./training/trained_models/atlasnet_singleview_25_squares/network.pth"):
            print("Dowload Trained Models.")
            os.system("chmod +x training/download_trained_models.sh")
            os.system("./training/download_trained_models.sh")

        if opt.reload_model_path == "" and opt.SVR:
            opt.dir_name = "./training/trained_models/atlasnet_singleview_1_sphere"
        elif opt.reload_model_path == "" and not opt.SVR:
            opt.dir_name = "./training/trained_models/atlasnet_autoencoder_1_sphere"


    if exists(join(opt.dir_name, "options.json")):
        # Reload parameters from options.txt if it exists
        with open(join(opt.dir_name, "options.json"), 'r') as f:
            my_opt_dict = json.load(f)
        my_opt_dict.pop("run_single_eval")
        my_opt_dict.pop("no_metro")
        my_opt_dict.pop("train_only_encoder")
        my_opt_dict.pop("no_learning")
        my_opt_dict.pop("demo")
        my_opt_dict.pop("demo_input_path")
        my_opt_dict.pop("dir_name")
        for key in my_opt_dict.keys():
            opt[key] = my_opt_dict[key]
        if not opt.demo:
            print("Modifying input arguments to match network in dirname")
            my_utils.cyan_print("PARAMETER: ")
            for a in my_opt_dict:
                print(
                    "         "
                    + colored(a, "yellow")
                    + " : "
                    + colored(str(my_opt_dict[a]), "cyan")
                )

    # Hard code dimension of the template.
    dim_template_dict = {
        "SQUARE": 2,
        "SPHERE": 3,
    }
    opt.dim_template = dim_template_dict[opt.template_type]

    # Visdom env
    opt.env = opt.env + opt.dir_name.split('/')[-1]

    return opt