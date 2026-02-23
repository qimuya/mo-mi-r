import os
import time
import json
import ast
import random
from copy import deepcopy

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from mmcv import Config

from drugood.datasets import build_dataset, build_dataloader
from modules import dataset as _dataset_register
from modules.RGNN import MoEGraphEncoder, Framework

try:
    from geoopt.optim import RiemannianAdam
except Exception:
    RiemannianAdam = None

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

    print(f"[seed] {seed}")

def get_device(device_id: int):
    if device_id < 0 or (not torch.cuda.is_available()):
        return torch.device("cpu")
    return torch.device(f"cuda:{device_id}")

def parse_subs_list(batch_subs):
    out = []
    for s in batch_subs:
        if s is None:
            out.append([])
            continue
        if isinstance(s, (list, tuple, set)):
            toks = list(s)
            out.append(sorted([str(x) for x in toks]))
            continue

        s = str(s).strip()
        if len(s) == 0:
            out.append([])
            continue

        try:
            obj = ast.literal_eval(s)
        except Exception:
            out.append([s])
            continue

        if isinstance(obj, set):
            out.append(sorted([str(x) for x in obj]))
        elif isinstance(obj, (list, tuple)):
            out.append(sorted([str(x) for x in obj]))
        else:
            out.append([str(obj)])
    return out

@torch.no_grad()
def eval_one_epoch(loader, model, device, verbose=False):
    model.eval()
    y_pred, y_gt = [], []
    it = tqdm(loader, disable=not verbose)
    for data in it:
        subs = parse_subs_list(data["subs"])
        graphs = data["input"].to(device)
        labels = data["gt_label"].to(device).float().view(-1)

        logits = model(subs, graphs).view(-1)
        probs = torch.sigmoid(logits)

        y_pred.append(probs.detach().cpu())
        y_gt.append(labels.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0).view(-1).numpy()
    y_gt = torch.cat(y_gt, dim=0).view(-1).numpy()

    try:
        return float(roc_auc_score(y_gt, y_pred))
    except Exception:
        return float("nan")

def build_rgnn_moe_model(
    *,
    in_dim: int,
    num_tasks: int,
    init_curvs,
    hidden_dim: int,
    out_dim: int,
    n_layers_expert: int,
    n_layers_gate: int,
    gate_hidden_dim: int,
    dropout: float,
    init_tau: float,
    tau_min: float,
    tau_max: float,
    proto_init: str,
    proto_std: float,
    normalize_h: bool,
):
    sub_model = MoEGraphEncoder(
        init_curvs=init_curvs,
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        n_layers_r=n_layers_expert,
        n_layers_d=n_layers_gate,
        gate_hidden_dim=gate_hidden_dim,
        init_tau=init_tau,
        tau_min=tau_min,
        tau_max=tau_max,
        proto_init=proto_init,
        proto_std=proto_std,
        normalize_h=normalize_h,
    )
    sub_dim = len(init_curvs) * out_dim
    model = Framework(
        sub_model=sub_model,
        num_tasks=num_tasks,
        sub_dim=sub_dim,
        dropout=dropout,
    )
    return model

def init_args():
    import argparse

    parser = argparse.ArgumentParser("Train RGNN-MoE (Gate3) on DrugOOD")

    parser.add_argument("--data_config", type=str, required=True)
    parser.add_argument("--dataset_tag", type=str, default="exp")

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval_verbose", action="store_true")

    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr_gate", type=float, default=None)
    parser.add_argument("--lr_cls", type=float, default=None)
    parser.add_argument("--lr_riemann", type=float, default=1e-3)
    parser.add_argument("--wd_gate", type=float, default=0.0)
    parser.add_argument("--wd_cls", type=float, default=0.0)
    parser.add_argument("--wd_riemann", type=float, default=0.0)

    parser.add_argument("--init_curvs", type=float, nargs="+", default=[0.0, -1.0, 1.0])
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--out_dim", type=int, default=32)
    parser.add_argument("--n_layers_expert", type=int, default=2)
    parser.add_argument("--n_layers_gate", type=int, default=2)
    parser.add_argument("--gate_hidden_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--init_tau", type=float, default=1.5)
    parser.add_argument("--tau_min", type=float, default=0.05)
    parser.add_argument("--tau_max", type=float, default=10.0)
    parser.add_argument("--proto_init", type=str, default="normal", choices=["normal", "orthogonal"])
    parser.add_argument("--proto_std", type=float, default=0.02)
    parser.add_argument("--normalize_h", action="store_true")

    parser.add_argument("--work_dir", type=str, default="log_rgnn_moe")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--save_model_name", type=str, default="final_model.pth")
    parser.add_argument("--save_log_name", type=str, default="train_log.json")

    return parser.parse_args()

def main():
    args = init_args()
    set_seed_everywhere(args.seed)
    device = get_device(args.device)
    print("[device]", device)

    if args.lr is not None:
        if args.lr_gate is None:
            args.lr_gate = args.lr
        if args.lr_cls is None:
            args.lr_cls = args.lr

    if args.lr_gate is None:
        args.lr_gate = 1e-3
    if args.lr_cls is None:
        args.lr_cls = 1e-3

    data_cfg = Config.fromfile(args.data_config)
    print(data_cfg.pretty_text)

    train_set = build_dataset(data_cfg.data.train)
    valid_set = build_dataset(data_cfg.data.ood_val)
    test_set = build_dataset(data_cfg.data.ood_test)
    data_cfg.data.ood_val.test_mode = True
    data_cfg.data.ood_test.test_mode = True

    train_loader = build_dataloader(
        train_set, data_cfg.data.samples_per_gpu,
        data_cfg.data.workers_per_gpu, num_gpus=1,
        dist=False, round_up=False, seed=args.seed, shuffle=True
    )
    valid_loader = build_dataloader(
        valid_set, data_cfg.data.samples_per_gpu,
        data_cfg.data.workers_per_gpu, num_gpus=1,
        dist=False, round_up=False, seed=args.seed, shuffle=False
    )
    test_loader = build_dataloader(
        test_set, data_cfg.data.samples_per_gpu,
        data_cfg.data.workers_per_gpu, num_gpus=1,
        dist=False, round_up=False, seed=args.seed, shuffle=False
    )

    in_dim = 39
    num_tasks = int(getattr(data_cfg.data, "num_class", 1))

    model = build_rgnn_moe_model(
        in_dim=in_dim,
        num_tasks=num_tasks,
        init_curvs=args.init_curvs,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        n_layers_expert=args.n_layers_expert,
        n_layers_gate=args.n_layers_gate,
        gate_hidden_dim=args.gate_hidden_dim,
        dropout=args.dropout,
        init_tau=args.init_tau,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        proto_init=args.proto_init,
        proto_std=args.proto_std,
        normalize_h=args.normalize_h,
    ).to(device)

    use_group_optim = (hasattr(model, "sub_model")
                       and hasattr(model.sub_model, "experts")
                       and (RiemannianAdam is not None))

    if use_group_optim:
        experts_params = []
        for e in model.sub_model.experts:
            experts_params += list(e.parameters())
        optim_riem = RiemannianAdam(
            experts_params, lr=args.lr_riemann, weight_decay=args.wd_riemann, stabilize=30
        )

        optim_gate = AdamW(
            [
                {"params": model.sub_model.gate_encoder.parameters(), "weight_decay": args.wd_gate},
                {"params": model.sub_model.gating.parameters(), "weight_decay": 0.0},
            ],
            lr=args.lr_gate
        )

        cls_params = list(model.predictor.parameters())
        if hasattr(model, "aggregator") and model.aggregator is not None:
            cls_params += list(model.aggregator.parameters())
        optim_cls = AdamW(cls_params, lr=args.lr_cls, weight_decay=args.wd_cls)

        print("[optim] RiemannianAdam(experts) + AdamW(gate) + AdamW(cls)")
    else:
        optim_all = AdamW(model.parameters(), lr=min(args.lr_gate, args.lr_cls), weight_decay=args.wd_cls)
        optim_riem = optim_all
        optim_gate = optim_all
        optim_cls = optim_all
        print("[optim] fallback: single AdamW(all params)")

    bce = torch.nn.BCEWithLogitsLoss()

    run_dir = os.path.join(args.work_dir, args.dataset_tag, str(time.time()))
    os.makedirs(run_dir, exist_ok=True)
    save_model_path = os.path.join(run_dir, args.save_model_name)
    save_log_path = os.path.join(run_dir, args.save_log_name)
    print("[run_dir]", run_dir)

    curve = {"train": [], "valid": [], "test": []}

    for ep in range(args.epochs):
        model.train()
        loss_sum = 0.0
        steps = 0

        for data in tqdm(train_loader, desc=f"Train ep{ep}"):
            subs = parse_subs_list(data["subs"])
            graphs = data["input"].to(device)
            labels = data["gt_label"].to(device).float().view(-1)

            logits = model(subs, graphs).view(-1)
            mask = torch.isfinite(labels)
            if mask.sum() == 0:
                continue
            loss = bce(logits[mask], labels[mask])

            optim_riem.zero_grad()
            optim_gate.zero_grad()
            optim_cls.zero_grad()
            loss.backward()
            optim_riem.step()
            optim_gate.step()
            optim_cls.step()

            loss_sum += float(loss.item())
            steps += 1

        avg_loss = loss_sum / max(steps, 1)
        print(f"[Epoch {ep}] train loss = {avg_loss:.6f}")

        train_auc = eval_one_epoch(train_loader, model, device, verbose=args.eval_verbose)
        valid_auc = eval_one_epoch(valid_loader, model, device, verbose=args.eval_verbose)
        test_auc = eval_one_epoch(test_loader, model, device, verbose=args.eval_verbose)

        curve["train"].append(train_auc)
        curve["valid"].append(valid_auc)
        curve["test"].append(test_auc)

        print({"Train": train_auc, "Valid": valid_auc, "Test": test_auc})

        with open(save_log_path, "w") as f:
            json.dump(
                {
                    "args": vars(args),
                    "run_dir": run_dir,
                    "curve": curve,
                },
                f,
                indent=2,
            )

    if not args.no_save:
        torch.save(model.state_dict(), save_model_path)
        print(f"[SAVE] model -> {save_model_path}")

    print(f"[FINAL] Train AUC = {train_auc:.6f}, Valid AUC = {valid_auc:.6f}, Test AUC = {test_auc:.6f}")

if __name__ == "__main__":
    main()