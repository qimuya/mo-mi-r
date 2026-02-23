# ChemistryProcess.py
import numpy as np
import torch
from torch_geometric.data import Data

from rdkit import Chem
from rdkit.Chem import BRICS, Recap

from drugood.utils import smile2graph 

_SM2G_CACHE = {}
_ATOMCNT_CACHE = {}

def _real_atom_count(smiles: str) -> int:
    if smiles in _ATOMCNT_CACHE:
        return _ATOMCNT_CACHE[smiles]
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        _ATOMCNT_CACHE[smiles] = 0
        return 0
    cnt = 0
    for a in m.GetAtoms():
        if a.GetAtomicNum() > 0:
            cnt += 1
    _ATOMCNT_CACHE[smiles] = cnt
    return cnt
def cached_smile2graph(smi: str):
    g = _SM2G_CACHE.get(smi, None)
    if g is None:
        g = smile2graph(smi)
        _SM2G_CACHE[smi] = g
    return g

def clear_sm2g_cache():
    _SM2G_CACHE.clear()
def get_substructure(mol=None, smile=None, decomp="brics"):
    assert mol is not None or smile is not None, "need at least one info of mol"
    assert decomp in ["brics", "recap"], "Invalid decomposition method"

    if mol is None:
        mol = Chem.MolFromSmiles(smile)

    if mol is None:
        return set()

    if decomp == "brics":
        substructures = BRICS.BRICSDecompose(mol)
    else:
        recap_tree = Recap.RecapDecompose(mol)
        leaves = recap_tree.GetLeaves()
        substructures = set(leaves.keys()) if leaves is not None else set()
    return substructures

def substructure_batch(smiles, return_mask, return_type="numpy", device=None, decomp="brics"):
    sub_structures = [get_substructure(smile=x, decomp=decomp) for x in smiles]
    return graph_from_substructure(sub_structures, return_mask, return_type, device=device)
def graph_from_substructure(subs, return_mask=False, return_type="numpy", device=None, min_atoms_keep: int = 3):
    assert return_type in ["numpy", "torch", "pyg"], "Invalid return_type"
    sub_struct_set = set()
    for s in subs:
        if not s:
            continue
        toks = list(s) if isinstance(s, (set, list, tuple)) else [s]
        for t in toks:
            smi = str(t)
            if _real_atom_count(smi) >= min_atoms_keep:
                sub_struct_set.add(smi)

    sub_struct_list = sorted(list(sub_struct_set))  # stable order
    S = len(sub_struct_list)
    B = len(subs)
    sub_to_idx = {x: idx for idx, x in enumerate(sub_struct_list)}
    mask = np.zeros((B, S), dtype=bool)
    for bi, s in enumerate(subs):
        if not s:
            continue
        toks = list(s) if isinstance(s, (set, list, tuple)) else [s]
        idxs = []
        for t in toks:
            smi = str(t)
            if _real_atom_count(smi) < min_atoms_keep:
                continue
            if smi in sub_to_idx:
                idxs.append(sub_to_idx[smi])
        if len(idxs) > 0:
            mask[bi, idxs] = True
    if S == 0:
        if return_type == "numpy":
            result = {
                "edge_index": np.zeros((2, 0), dtype=np.int64),
                "edge_attr": np.zeros((0, 0), dtype=np.float32),
                "batch": np.zeros((0,), dtype=np.int64),
                "x": np.zeros((0, 39), dtype=np.float32),
                "num_nodes": 0,
            }
            return (result, mask) if return_mask else result

        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_attr  = torch.zeros((0, 0), dtype=torch.float32, device=device)
        batch_vec  = torch.zeros((0,), dtype=torch.long, device=device)
        x          = torch.zeros((0, 39), dtype=torch.float32, device=device)

        result = {"edge_index": edge_index, "edge_attr": edge_attr, "batch": batch_vec, "x": x, "num_nodes": 0}
        if return_type == "pyg":
            result = Data(**result)
        return (result, mask) if return_mask else result
    edge_idxes, edge_feats, node_feats, batch_vec = [], [], [], []
    lstnode = 0

    for si, smi in enumerate(sub_struct_list):
        g = cached_smile2graph(smi)  # DGLGraph (usually CPU)

        x0 = g.ndata["x"]  # [n, 39]
        if "x" in g.edata:
            e0 = g.edata["x"]  # [m, 10]
        else:
            m = g.num_edges()
            e0 = torch.zeros((m, 0), dtype=torch.float32)  # CPU

        u0, v0 = g.edges()  # tensors (CPU)

        if return_type == "numpy":
            u_np = u0.detach().cpu().numpy().astype(np.int64)
            v_np = v0.detach().cpu().numpy().astype(np.int64)
            edge_index_np = np.stack([u_np, v_np], axis=0) + lstnode

            edge_idxes.append(edge_index_np)
            edge_feats.append(e0.detach().cpu().numpy().astype(np.float32))
            node_feats.append(x0.detach().cpu().numpy().astype(np.float32))

            n = int(x0.shape[0])
            batch_vec.append(np.full((n,), si, dtype=np.int64))
            lstnode += n
        else:
            x = x0.to(device) if device is not None else x0
            e = e0.to(device) if device is not None else e0
            u = u0.to(device) if device is not None else u0
            v = v0.to(device) if device is not None else v0

            edge_index = (torch.stack([u, v], dim=0).long() + lstnode)  # [2, m]

            edge_idxes.append(edge_index)
            edge_feats.append(e.float())
            node_feats.append(x.float())

            n = int(x.size(0))
            batch_vec.append(torch.full((n,), si, dtype=torch.long, device=x.device))
            lstnode += n
    if return_type == "numpy":
        result = {
            "edge_index": np.concatenate(edge_idxes, axis=1) if edge_idxes else np.zeros((2, 0), dtype=np.int64),
            "edge_attr":  np.concatenate(edge_feats, axis=0) if edge_feats else np.zeros((0, 0), dtype=np.float32),
            "batch":      np.concatenate(batch_vec, axis=0) if batch_vec else np.zeros((0,), dtype=np.int64),
            "x":          np.concatenate(node_feats, axis=0) if node_feats else np.zeros((0, 39), dtype=np.float32),
            "num_nodes":  int(lstnode),
        }
        return (result, mask) if return_mask else result

    edge_index = torch.cat(edge_idxes, dim=1) if edge_idxes else torch.zeros((2, 0), dtype=torch.long, device=device)
    edge_attr  = torch.cat(edge_feats, dim=0) if edge_feats else torch.zeros((0, 0), dtype=torch.float32, device=device)
    batch      = torch.cat(batch_vec, dim=0) if batch_vec else torch.zeros((0,), dtype=torch.long, device=device)
    x          = torch.cat(node_feats, dim=0) if node_feats else torch.zeros((0, 39), dtype=torch.float32, device=device)

    result = {"edge_index": edge_index, "edge_attr": edge_attr, "batch": batch, "x": x, "num_nodes": int(lstnode)}
    if return_type == "pyg":
        result = Data(**result)

    return (result, mask) if return_mask else result
