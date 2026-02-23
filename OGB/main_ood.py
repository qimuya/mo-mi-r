import os
import json
from tqdm import tqdm
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from ogb.graphproppred import Evaluator
from geoopt.optim import RiemannianAdam
import random
from modules.DataLoading import pyg_molsubdataset
from modules.RGNN import MoEGraphEncoder as MoEGraph, Framework

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.warning")

def set_seed_everywhere(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TORCH_ALLOW_TF32_CUBLAS"] = "0"
    os.environ["TORCH_ALLOW_TF32_CUDNN"] = "0"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(
        f"[seed setup] seed={seed} "
        f"PYTHONHASHSEED={os.environ['PYTHONHASHSEED']} "
        f"CUBLAS_WORKSPACE_CONFIG={os.environ['CUBLAS_WORKSPACE_CONFIG']}"
    )

def parse_subs_batch(batch_sub):
    subs = []
    for s in batch_sub:
        s = s.strip()
        if len(s) == 0:
            subs.append(set())
        else:
            subs.append(set(s.split("|||")))
    return subs

@torch.no_grad()
def eval_one_epoch(loader, evaluator, model, device, verbose=False):
    model.eval()
    y_pred_list, y_gt_list = [], []
    iterx = tqdm(loader, leave=False) if verbose else loader

    for batch_sub, batch_graph in iterx:
        subs = parse_subs_batch(batch_sub)
        batch_graph = batch_graph.to(device)

        out = model(subs, batch_graph)
        
        if isinstance(out, (tuple, list)):
            pred = out[0]
        else:
            pred = out

        y_pred_list.append(pred.detach().cpu())
        y_gt_list.append(batch_graph.y.reshape(pred.shape).detach().cpu())

    y_pred = torch.cat(y_pred_list, dim=0).numpy()
    y_gt = torch.cat(y_gt_list, dim=0).numpy()
    
    return evaluator.eval({"y_true": y_gt, "y_pred": y_pred})

def build_sub_model_from_config(sub_cfg, dataset_node_feat_dim=None, args=None):
    typ = sub_cfg.get("type", "").lower()
    paras = sub_cfg.get("paras", {}) if sub_cfg.get("paras") is not None else {}

    if args is not None:
        if args.init_curvs is not None:
            paras["init_curvs"] = args.init_curvs
        paras["n_layers_d"] = args.n_layers_gate
        paras["n_layers_r"] = args.n_layers_expert
        paras["hidden_dim"] = args.hidden_dim_moe
        paras["out_dim"] = args.out_dim_moe
        paras["gate_hidden_dim"] = args.hidden_dim_gate
        paras["init_tau"] = args.init_tau
        paras["tau_min"] = args.tau_min
        paras["tau_max"] = args.tau_max
        paras["proto_init"] = args.proto_init
        paras["proto_std"] = args.proto_std
        paras["normalize_h"] = args.normalize_h

    if "rgnn-moe" in typ or "moegraph" in typ:
        init_curvs = paras.get("init_curvs", paras.get("init_curv", [0.0, -1.0, 1.0]))
        in_dim = dataset_node_feat_dim if dataset_node_feat_dim is not None else paras.get("in_dim")
        hidden_dim = paras.get("hidden_dim", 64)
        out_dim = paras.get("out_dim", 64)
        n_layers_r = paras.get("n_layers_r", 2)
        n_layers_d = paras.get("n_layers_d", 2)
        gate_hidden_dim = paras.get("gate_hidden_dim", 32)

        return MoEGraph(
            init_curvs=init_curvs,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers_r=n_layers_r,
            n_layers_d=n_layers_d,
            gate_hidden_dim=gate_hidden_dim,
            init_tau=paras.get("init_tau", 1.5),
            tau_min=paras.get("tau_min", 0.05),
            tau_max=paras.get("tau_max", 10.0),
            proto_init=paras.get("proto_init", "normal"),
            proto_std=paras.get("proto_std", 0.02),
            normalize_h=paras.get("normalize_h", False),
        )

    raise ValueError(f"Unknown sub backend type: {sub_cfg.get('type')}")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--sub_backend", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="ogbg-molbbbp")
    parser.add_argument("--decomp_method", choices=["brics", "recap"], default="brics")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=300)

    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr_gate", type=float, default=None)
    parser.add_argument("--lr_cls", type=float, default=None)
    parser.add_argument("--lr_riemann", type=float, default=1e-2)

    parser.add_argument("--wd_riemann", type=float, default=0.0)
    parser.add_argument("--wd_gate", type=float, default=0.0)
    parser.add_argument("--wd_cls", type=float, default=0.0)

    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--init_curvs", type=float, nargs="+", default=None)
    parser.add_argument("--n_layers_gate", type=int, default=2)
    parser.add_argument("--n_layers_expert", type=int, default=2)
    parser.add_argument("--hidden_dim_moe", type=int, default=32)
    parser.add_argument("--out_dim_moe", type=int, default=32)
    parser.add_argument("--hidden_dim_gate", type=int, default=32)

    parser.add_argument("--init_tau", type=float, default=1.5)
    parser.add_argument("--tau_min", type=float, default=0.05)
    parser.add_argument("--tau_max", type=float, default=10.0)
    parser.add_argument("--proto_init", type=str, default="normal", choices=["normal", "orthogonal"])
    parser.add_argument("--proto_std", type=float, default=0.02)
    parser.add_argument("--normalize_h", action="store_true")
    
    parser.add_argument("--sharp_coef", type=float, default=0.0)
    parser.add_argument("--div_coef", type=float, default=0.0)
    
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed_everywhere(args.seed)

    if args.lr_gate is None:
        args.lr_gate = args.lr
    if args.lr_cls is None:
        args.lr_cls = args.lr

    device = torch.device("cpu") if args.device < 0 or (not torch.cuda.is_available()) else torch.device(f"cuda:{args.device}")
    print(f"[device] {device}  lr_gate={args.lr_gate}  lr_cls={args.lr_cls}  lr_riemann={args.lr_riemann}")
    print(f"[gate_reg] sharp={args.sharp_coef} div={args.div_coef}")

    with open(args.sub_backend) as f:
        sub_cfg = json.load(f)

    total_smiles, total_subs, dataset = pyg_molsubdataset(args.dataset, args.decomp_method)

    def canon_sub_to_str(s):
        if isinstance(s, set):
            toks = sorted(list(s))
        elif isinstance(s, (list, tuple)):
            toks = sorted(list(s))
        else:
            toks = [str(s)]
        return "|||".join(toks)

    canon_subs = [canon_sub_to_str(s) for s in total_subs]
    evaluator = Evaluator(args.dataset)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"]
    valid_idx = split_idx["valid"]
    test_idx = split_idx["test"]

    def idx_to_int(x):
        return x.item() if hasattr(x, "item") else int(x)

    train_subs = [canon_subs[idx_to_int(x)] for x in train_idx]
    valid_subs = [canon_subs[idx_to_int(x)] for x in valid_idx]
    test_subs  = [canon_subs[idx_to_int(x)] for x in test_idx]

    train_graphs = [dataset[idx_to_int(x)] for x in train_idx]
    valid_graphs = [dataset[idx_to_int(x)] for x in valid_idx]
    test_graphs  = [dataset[idx_to_int(x)] for x in test_idx]

    dataset_node_feat_dim = dataset.num_node_features if hasattr(dataset, "num_node_features") else None
    sub_model = build_sub_model_from_config(sub_cfg, dataset_node_feat_dim=dataset_node_feat_dim, args=args).to(device)

    paras = sub_cfg.get("paras", {}) if sub_cfg.get("paras") is not None else {}
    init_curvs = args.init_curvs if args.init_curvs is not None else paras.get("init_curvs", paras.get("init_curv", [0.0, -1.0, 1.0]))
    out_dim = args.out_dim_moe if args.out_dim_moe is not None else paras.get("out_dim", paras.get("out_dim_moe", 16))

    sub_dim = len(init_curvs) * int(out_dim)
    main_model = Framework(
        sub_model=sub_model,
        num_tasks=dataset.num_tasks,
        sub_dim=sub_dim,
        dropout=args.dropout
    ).to(device)

    train_loader = DataLoader(list(zip(train_subs, train_graphs)), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(list(zip(valid_subs, valid_graphs)), batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(list(zip(test_subs, test_graphs)), batch_size=args.batch_size, shuffle=False)

    use_group_optim = hasattr(sub_model, "experts") and (RiemannianAdam is not None)

    if use_group_optim:
        experts_params = []
        for e in sub_model.experts:
            experts_params += list(e.parameters())
        r_optim = RiemannianAdam(experts_params, lr=args.lr_riemann, weight_decay=args.wd_riemann, stabilize=30)

        optimizer_gate = torch.optim.Adam(
            [
                {"params": sub_model.gate_encoder.parameters(), "weight_decay": args.wd_gate},
                {"params": sub_model.gating.parameters(), "weight_decay": 0.0},
            ],
            lr=args.lr_gate
        )

        cls_params = list(main_model.predictor.parameters())
        if hasattr(main_model, "aggregator") and main_model.aggregator is not None:
            cls_params += list(main_model.aggregator.parameters())
        if hasattr(main_model, "light_mlp") and main_model.light_mlp is not None:
            cls_params += list(main_model.light_mlp.parameters())

        optimizer_cls = torch.optim.Adam(cls_params, lr=args.lr_cls, weight_decay=args.wd_cls)
        print("Optimizer: RiemannianAdam(experts) + Adam(gate) + Adam(cls)")
    else:
        optimizer = torch.optim.Adam(main_model.parameters(), lr=args.lr, weight_decay=args.wd_cls)
        r_optim = optimizer
        optimizer_gate = optimizer
        optimizer_cls = optimizer
        print("Optimizer: single Adam")

    task_type = dataset.task_type
    cls_criterion = torch.nn.BCEWithLogitsLoss()
    reg_criterion = torch.nn.MSELoss()

    train_curve, valid_curve, test_curve = [], [], []

    for ep in range(args.epoch):
        main_model.train()
        total_loss = 0.0
        count = 0

        gate_batches = 0
        H_w_sum = 0.0
        H_bar_sum = 0.0
        reg_sum = 0.0

        for batch_sub, batch_graph in tqdm(train_loader, desc=f"Train ep{ep}", leave=False):
            subs = parse_subs_batch(batch_sub)
            batch_graph = batch_graph.to(device)

            if (not hasattr(batch_graph, "x")) or (batch_graph.x is None) or (batch_graph.x.shape[0] <= 1):
                continue

            r_optim.zero_grad()
            optimizer_gate.zero_grad()
            optimizer_cls.zero_grad()

            out = main_model(subs, batch_graph)
            if isinstance(out, (tuple, list)):
                pred, w_mol = out
            else:
                pred = out
                w_mol = None

            is_labeled = batch_graph.y == batch_graph.y
            if is_labeled.sum() == 0:
                continue

            if "classification" in task_type:
                base_loss = cls_criterion(
                    pred.to(torch.float32)[is_labeled],
                    batch_graph.y.to(torch.float32)[is_labeled]
                )
            else:
                base_loss = reg_criterion(
                    pred.to(torch.float32)[is_labeled],
                    batch_graph.y.to(torch.float32)[is_labeled]
                )

            reg_loss = 0.0
            if (args.sharp_coef > 0.0 or args.div_coef > 0.0) and (w_mol is not None):
                w = w_mol.clamp_min(1e-12)
                H_w = -(w * w.log()).sum(dim=-1).mean()
                w_bar = w.mean(dim=0)
                H_bar = -(w_bar * w_bar.log()).sum()

                if args.sharp_coef > 0.0:
                    reg_loss = reg_loss + args.sharp_coef * H_w
                if args.div_coef > 0.0:
                    reg_loss = reg_loss + args.div_coef * (-H_bar)

                gate_batches += 1
                H_w_sum += float(H_w.item())
                H_bar_sum += float(H_bar.item())
                reg_sum += float(reg_loss.item()) if torch.is_tensor(reg_loss) else float(reg_loss)

            loss = base_loss + reg_loss
            loss.backward()

            r_optim.step()
            optimizer_gate.step()
            optimizer_cls.step()

            total_loss += float(loss.item())
            count += 1

        avg_loss = total_loss / count if count > 0 else 0.0
        print(f"[Epoch {ep}] train loss: {avg_loss:.6f}")

        train_perf = eval_one_epoch(train_loader, evaluator, main_model, device)
        valid_perf = eval_one_epoch(valid_loader, evaluator, main_model, device)
        test_perf  = eval_one_epoch(test_loader, evaluator, main_model, device)

        metric = dataset.eval_metric
        train_score = float(train_perf[metric])
        valid_score = float(valid_perf[metric])
        test_score  = float(test_perf[metric])

        print({"Train": train_score, "Validation": valid_score, "Test": test_score})

        train_curve.append(train_score)
        valid_curve.append(valid_score)
        test_curve.append(test_score)

if __name__ == "__main__":
    main()