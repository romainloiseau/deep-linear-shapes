import copy

import numpy as np

from tqdm.auto import tqdm

import torch

import open3d as o3d

from ..global_variables import N_MAX_POINTS

def sample_obj(obj_path, n_points = N_MAX_POINTS):
    
    mesh = o3d.io.read_triangle_mesh(obj_path)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()

    triangle_clusters, cluster_n_triangles, _ = (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    
    clouds = []
    for i, n in enumerate(cluster_n_triangles):
        if n > 100:
            mesh1 = copy.deepcopy(mesh)
            mesh1.remove_triangles_by_mask(triangle_clusters != i)

            clouds.append(np.asarray(mesh1.sample_points_uniformly(n_points).points))
            
    return clouds
    
def sample_best_distributed(y, N, get_firsts = False):
    
    unique, counts = np.unique(y, return_counts = True)
    counts = counts / counts.sum()
    
    if len(unique) > N:
        raise ValueError(f'N ({N}) should be superior to the number of categories ({len(unique)})')
    
    numbers = np.ones(len(unique))
    if N > numbers.sum():
        numbers += ((N - len(unique)) * counts)
        numbers = numbers.astype(int)
        while(numbers.sum() != N):
            numbers[np.random.choice(len(unique), 1, p = counts)[0]] += 1
            
    numbers = numbers.astype(int)
    indices = []
    for u, n in zip(unique, numbers):
        if get_firsts:
            indices.append(np.arange(len(y))[y == u][:n])
        else:
            indices.append(np.random.choice(np.arange(len(y))[y == u], n, replace = False))
        
    return np.hstack(indices)

def sample_best_distributed_pointwise(dataset, N, get_firsts = False):
    
    unique, counts = np.unique(dataset.data.y, return_counts = True)
    counts = np.log(counts)
    counts = counts / counts.sum()
    
    if len(unique) > N:
        raise ValueError(f'N ({N}) should be superior to the number of categories ({len(unique)})')
    
    numbers = np.ones(len(unique))
    if N > len(unique):
        numbers += ((N - len(unique)) * counts)
        numbers = numbers.astype(int)
        while(numbers.sum() != N):
            numbers[np.random.choice(len(unique), 1, p = counts)[0]] += 1
            
    indices = []       
    #Iterate over classes
    for u, n in zip(tqdm(unique, leave = False), numbers):
        
        #Discover available combinations in dataset
        combination = []
        n_combination = []
        idx_combination = []
        for sample in dataset[dataset.data.y == u]:
            assert sample.y == u
            comb = torch.unique(sample.point_y)
            exists = [torch.equal(comb, c) for c in combination]
            
            if not any(exists):
                combination.append(comb)
                n_combination.append(1)
                idx_combination.append([sample.id_scan])
            else:
                idx = np.where(exists)[0][0]
                n_combination[idx] += 1
                idx_combination[idx].append(sample.id_scan)
                
        #Choose uniformly combinations from most to less represented
        choosen_per_combination = []
        for comb in np.argsort(n_combination)[::-1]:
            if len(choosen_per_combination) < n:
                choosen_per_combination.append(comb)
                
        #Add randomly choosen combinations (with probability = n samples per combinations)
        if len(choosen_per_combination) < n:
            for comb in np.random.choice(len(n_combination),
                                         n - len(choosen_per_combination),
                                         p = np.array(n_combination) / np.sum(n_combination)):
                choosen_per_combination.append(comb)
        
        #Append indexes
        ucomb, ncomb = np.unique(choosen_per_combination, return_counts = True)
        for uu, nn in zip(ucomb, ncomb):
        #for comb in choosen_per_combination:
            if get_firsts:
                indices.append(idx_combination[uu][:nn])
            else:
                indices.append(np.random.choice(idx_combination[uu], nn)[0])
        
    return np.hstack(indices)