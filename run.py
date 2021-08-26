import dlm

import random
import numpy as np
import torch

args = dlm.utils.get_parser().parse_args()

if args.fix_seed:
    print("Setting seed at value 1")
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

if __name__ == "__main__":
    dataset = dlm.datasets.load(
        args
    )
    
    initialisation = dlm.utils.ShapeSampler(
        args.initialisation, args.K
    )(
        dataset.train_dataset
    )
    
    model = dlm.models.DTI3D(
        initialisation, args
    )
    
    activator = dlm.schedulers.ActivatorDTI3D(args)
    
    trainer = dlm.trainers.ClusteringTrainer(
        args,
        name = args.name if args.name is not None else f"{args.dataset}_clusteringK{args.K}_{args.alignment}"
    )
    trainer.initialize_materials(model, dataset, activator)
    
    trainer.train(args.n_epochs)