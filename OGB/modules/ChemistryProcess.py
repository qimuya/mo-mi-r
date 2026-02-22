# ChemistryProcess.py
import numpy as np
import torch
from rdkit import Chem
from ogb.utils import smiles2graph
from torch_geometric.data import Data
import hashlib
def _num_atoms(smiles: str) -> int:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return 0
    return sum(1 for a in m.GetAtoms() if a.GetAtomicNum() > 0)
def _stable_hash_to_bucket(s: str, num_buckets: int) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h, 16) % num_buckets

def graph_from_substructure(
    subs,
    return_mask: bool = False,
    return_type: str = "numpy",
    min_atoms_heavy: int = 2,
    return_light_counts: bool = False,
    light_hash_bins: int = 256,
):
    assert return_type in ["numpy", "torch", "pyg"], "Invaild return type"

    B = len(subs)
    sub_struct_set = set()
    for s in subs:
        sub_struct_set.update(s)

    heavy_list, light_list = [], []
    for smi in sub_struct_set:
        na = _num_atoms(smi)
        if na >= min_atoms_heavy:
            heavy_list.append(smi)
        else:
            light_list.append(smi)

    heavy_list = sorted(heavy_list)
    light_list = sorted(light_list)
    sub_to_idx = {x: idx for idx, x in enumerate(heavy_list)}
    mask = np.zeros([B, len(heavy_list)], dtype=bool)
    for i, s in enumerate(subs):
        idxs = [sub_to_idx[t] for t in s if t in sub_to_idx]
        if len(idxs) > 0:
            mask[i, idxs] = True
    heavy_result = None
    if len(heavy_list) > 0:
        sub_graph = [smiles2graph(x) for x in heavy_list]

        edge_idxes, edge_feats, node_feats, lstnode, batch = [], [], [], 0, []
        for idx, g in enumerate(sub_graph):
            edge_idxes.append(g["edge_index"] + lstnode)
            edge_feats.append(g["edge_feat"])
            node_feats.append(g["node_feat"])
            lstnode += g["num_nodes"]
            batch.append(np.ones(g["num_nodes"], dtype=np.int64) * idx)

        heavy_result = {
            "edge_index": np.concatenate(edge_idxes, axis=-1),
            "edge_attr": np.concatenate(edge_feats, axis=0),
            "batch": np.concatenate(batch, axis=0),
            "x": np.concatenate(node_feats, axis=0),
            "num_nodes": lstnode,
        }

        if return_type in ["torch", "pyg"]:
            for k, v in heavy_result.items():
                if k != "num_nodes":
                    heavy_result[k] = torch.from_numpy(v)
        if return_type == "pyg":
            heavy_result = Data(**heavy_result)
    light_counts = None
    if return_light_counts:
        H = int(light_hash_bins)
        light_counts = np.zeros([B, H], dtype=np.float32)
        for i, s in enumerate(subs):
            for t in s:
                if t in light_list:
                    b = _stable_hash_to_bucket(t, H)
                    light_counts[i, b] += 1.0
    if return_mask and return_light_counts:
        return heavy_result, mask, light_counts
    elif return_mask:
        return heavy_result, mask
    elif return_light_counts:
        return heavy_result, light_counts
    else:
        return heavy_result
