import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import geoopt
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import softmax as pyg_softmax
from torch_scatter import scatter_add
from .ChemistryProcess import graph_from_substructure

class kappaLinear(nn.Module):
    def __init__(self, manifold, in_dim: int, out_dim: int, dropout: float = 0.0, use_bias: bool = True):
        super().__init__()
        self.manifold = manifold
        self.dropout = float(dropout)
        self.use_bias = bool(use_bias)

        self.weight = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        res = self.manifold.mobius_matvec(drop_weight, x, project=True)
        if self.use_bias:
            bias = self.manifold.proju(self.manifold.origin(self.bias.shape), self.bias)
            kappa_bias = self.manifold.expmap0(bias, project=True)
            res = self.manifold.mobius_add(res, kappa_bias, project=True)
        return res

class kappaGCNConv(MessagePassing):
    def __init__(self, manifold, in_dim: int, out_dim: int):
        super().__init__(aggr="add")
        self.manifold = manifold
        self.lin = kappaLinear(manifold=self.manifold, in_dim=in_dim, out_dim=out_dim, use_bias=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_index, _ = add_self_loops(edge_index)

        x = self.lin(x)
        x_tan0 = self.manifold.logmap0(x)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index, x=x_tan0, norm=norm)
        out = self.manifold.expmap0(out, project=True)
        return out

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * x_j

class Encoder(nn.Module):
    def __init__(
        self,
        k: float,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 2,
        learnable: bool = True,
    ):
        super().__init__()
        assert n_layers >= 1

        self.manifold = geoopt.Stereographic(k=k, learnable=learnable)
        self.n_layers = int(n_layers)

        if self.n_layers == 1:
            self.convs = nn.ModuleList([kappaGCNConv(self.manifold, in_dim, out_dim)])
        else:
            layers = [kappaGCNConv(self.manifold, in_dim, hidden_dim)]
            for _ in range(self.n_layers - 2):
                layers.append(kappaGCNConv(self.manifold, hidden_dim, hidden_dim))
            layers.append(kappaGCNConv(self.manifold, hidden_dim, out_dim))
            self.convs = nn.ModuleList(layers)

    def _manifold_init(self, x: torch.Tensor) -> torch.Tensor:
        origin = self.manifold.origin(x.shape)
        u = self.manifold.proju(origin, x)
        return self.manifold.expmap0(u, project=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x_manifold = self._manifold_init(x)
        for conv in self.convs:
            x_manifold = conv(x_manifold, edge_index)
        return x_manifold

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x_manifold = self._manifold_init(x)
        for conv in self.convs:
            x_manifold = conv(x_manifold, edge_index)
        return self.manifold.logmap0(x_manifold)

class EuclidGatingEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers_d: int = 2, dropout: float = 0.0):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout = float(dropout)

        if n_layers_d == 1:
            self.convs.append(GCNConv(in_dim, out_dim))
        else:
            self.convs.append(GCNConv(in_dim, hidden_dim))
            for _ in range(n_layers_d - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, out_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = x.float()
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
        return global_mean_pool(x, batch)

class GeometricDistanceGate(nn.Module):
    def __init__(
        self,
        num_experts: int,
        d: int,
        init_tau: float = 1.0,
        tau_min: float = 0.05,
        tau_max: float = 10.0,
        proto_init: str = "normal",
        proto_std: float = 0.02,
        normalize_h: bool = False,
    ):
        super().__init__()
        self.K = int(num_experts)
        self.d = int(d)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.normalize_h = bool(normalize_h)

        self.u = nn.Parameter(torch.empty(self.K, self.d))
        if proto_init == "orthogonal" and self.K <= self.d:
            nn.init.orthogonal_(self.u)
            self.u.data *= 0.1
        else:
            nn.init.normal_(self.u, mean=0.0, std=float(proto_std))

        init = float(init_tau)
        init = max(init, self.tau_min + 1e-6)
        raw0 = np.log(np.expm1(init - self.tau_min))
        self._tau_raw = nn.Parameter(torch.ones(self.K) * raw0)

    def _tau(self):
        tau = F.softplus(self._tau_raw) + self.tau_min
        return tau.clamp(self.tau_min, self.tau_max)

    def forward(self, h: torch.Tensor, manifolds):
        if self.normalize_h:
            h = F.normalize(h, dim=-1)

        assert len(manifolds) == self.K, f"need {self.K} manifolds, got {len(manifolds)}"
        B = h.size(0)
        tau = self._tau()

        dists = []
        for k in range(self.K):
            M = manifolds[k]

            h0 = M.proju(M.origin(h.shape), h)
            z_k = M.expmap0(h0, project=True)

            u_k = self.u[k].unsqueeze(0).expand(B, -1)
            u0 = M.proju(M.origin(u_k.shape), u_k)
            y_k = M.expmap0(u0, project=True)

            d_k = M.dist(z_k, y_k)
            dists.append(d_k)

        dists = torch.stack(dists, dim=-1)
        scores = -dists / tau.unsqueeze(0)
        weights = F.softmax(scores, dim=-1)
        return weights, dists, tau

class RiemannianExpert(nn.Module):
    def __init__(self, k: float, in_dim: int, hidden_dim: int, out_dim: int, n_layers_r: int = 2, learnable: bool = True):
        super().__init__()
        self.encoder = Encoder(k, in_dim, hidden_dim, out_dim, n_layers=n_layers_r, learnable=learnable)

    @property
    def manifold(self):
        return self.encoder.manifold

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        z = self.encoder.encode(x, edge_index)
        g = global_mean_pool(z, batch)
        return g

class MoEGraphEncoder(nn.Module):
    def __init__(
        self,
        init_curvs,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers_r: int = 2,
        n_layers_d: int = 2,
        gate_hidden_dim: int = 32,
        init_tau: float = 1.0,
        tau_min: float = 0.05,
        tau_max: float = 10.0,
        proto_init: str = "normal",
        proto_std: float = 0.02,
        normalize_h: bool = False,
    ):
        super().__init__()
        self.init_curvs = list(init_curvs)
        self.num_experts = len(self.init_curvs)
        self.out_dim = int(out_dim)

        self.experts = nn.ModuleList([
            RiemannianExpert(
                k=float(k),
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                n_layers_r=n_layers_r,
                learnable=(float(k) != 0.0),
            )
            for k in self.init_curvs
        ])

        self.gate_encoder = EuclidGatingEncoder(
            in_dim=in_dim,
            hidden_dim=gate_hidden_dim,
            out_dim=out_dim,
            n_layers_d=n_layers_d,
            dropout=0.0
        )

        self.gating = GeometricDistanceGate(
            num_experts=self.num_experts,
            d=out_dim,
            init_tau=init_tau,
            tau_min=tau_min,
            tau_max=tau_max,
            proto_init=proto_init,
            proto_std=proto_std,
            normalize_h=normalize_h
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        emb_list = [expert(x, edge_index, batch) for expert in self.experts]
        features = torch.cat(emb_list, dim=-1)

        gate_feat = self.gate_encoder(x, edge_index, batch)

        manifolds = [expert.manifold for expert in self.experts]

        weights, dists, tau = self.gating(gate_feat, manifolds)

        dim_per_expert = features.size(-1) // self.num_experts
        weights_expanded = weights.repeat_interleave(dim_per_expert, dim=1)
        scaled = features * weights_expanded

        return scaled, weights, dists, tau

class AttentionAggregation(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, batch_idx: torch.Tensor, dim_size: int = None):
        scores = self.attn_net(x)
        alpha = pyg_softmax(scores, batch_idx, num_nodes=dim_size)
        out = scatter_add(x * alpha, batch_idx, dim=0, dim_size=dim_size)
        return out, alpha

class Framework(nn.Module):
    def __init__(
        self,
        sub_model: nn.Module,
        num_tasks: int,
        sub_dim: int,
        dropout: float = 0.0,
        light_hash_bins: int = 256,
        light_hidden: int = 128,
        light_dropout: float = 0.0,
    ):
        super().__init__()
        self.sub_model = sub_model
        self.sub_dim = int(sub_dim)

        self.light_hash_bins = int(light_hash_bins)

        self.light_mlp = nn.Sequential(
            nn.Linear(self.light_hash_bins, light_hidden),
            nn.ReLU(),
            nn.Dropout(light_dropout) if 0.0 < light_dropout < 1.0 else nn.Identity(),
            nn.Linear(light_hidden, self.sub_dim),
        )

        self.aggregator = AttentionAggregation(self.sub_dim, max(1, self.sub_dim // 2))

        layers = [nn.Linear(self.sub_dim, self.sub_dim)]
        if 0.0 < dropout < 1.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.sub_dim, num_tasks))
        self.predictor = nn.Sequential(*layers)

    def sub_feature_from_graphs(self, subs, device, return_mask: bool = False):
        heavy_graph, mask, light_counts = graph_from_substructure(
            subs,
            return_mask=True,
            return_type="pyg",
            min_atoms_heavy=2,
            return_light_counts=True,
            light_hash_bins=self.light_hash_bins,
        )

        light_counts = torch.from_numpy(light_counts).to(device)

        if heavy_graph is None:
            K = self.sub_model.num_experts if hasattr(self.sub_model, "num_experts") else 1
            empty_feat = torch.zeros((0, self.sub_dim), device=device)
            empty_w = torch.zeros((0, K), device=device)
            mask_t = torch.from_numpy(mask).to(device).bool()
            
            if return_mask:
                return empty_feat, empty_w, mask_t, light_counts
            else:
                return empty_feat, light_counts

        heavy_graph = heavy_graph.to(device)

        scaled, weights, dists, tau = self.sub_model(heavy_graph)

        mask_t = torch.from_numpy(mask).to(device).bool()

        if return_mask:
            return scaled, weights, mask_t, light_counts
        else:
            return scaled, light_counts

    def forward(self, substructures, batched_data, return_detail=False):
        device = batched_data.x.device if hasattr(batched_data, "x") else torch.device("cpu")

        sub_feat, sub_w, mask, light_counts = self.sub_feature_from_graphs(
            substructures, device=device, return_mask=True
        )

        B, N_heavy = mask.shape
        detail_info = {}

        if N_heavy == 0:
            mol_emb_heavy = torch.zeros((B, self.sub_dim), device=device)
            K = sub_w.size(-1) if sub_w.numel() > 0 else (self.sub_model.num_experts if hasattr(self.sub_model, "num_experts") else 1)
            w_mol = torch.ones((B, K), device=device) / float(K)
        else:
            nonzero_idx = mask.nonzero(as_tuple=False)
            batch_idx = nonzero_idx[:, 0]
            feat_idx  = nonzero_idx[:, 1]

            x_selected = sub_feat[feat_idx]
            w_selected = sub_w[feat_idx]

            mol_emb_heavy, alpha = self.aggregator(x_selected, batch_idx, dim_size=B)

            w_mol = scatter_add(w_selected * alpha, batch_idx, dim=0, dim_size=B)
            w_mol = w_mol / (w_mol.sum(dim=-1, keepdim=True).clamp_min(1e-12))

            if return_detail:
                detail_info['heavy_feat_idx'] = feat_idx.cpu().numpy()
                detail_info['batch_idx'] = batch_idx.cpu().numpy()
                detail_info['expert_weights'] = w_selected.detach().cpu().numpy()
                detail_info['attn_score'] = alpha.detach().cpu()

        light_in = torch.log1p(light_counts.float())
        light_emb = self.light_mlp(light_in)

        mol_emb = mol_emb_heavy
        logits = self.predictor(mol_emb)

        if return_detail:
            return logits, w_mol, detail_info
        else:
            return logits, w_mol