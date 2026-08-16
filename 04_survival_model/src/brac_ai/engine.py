from __future__ import annotations
import csv
import math
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from .losses import cox_ph_loss, multitask_cox_loss
from .metrics import concordance_index

TENSOR_KEYS = (
    "image", "morphology_map",
    "os_time", "dfs_time", "os_event", "dfs_event",
)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in TENSOR_KEYS:
        moved[key] = batch[key].to(device, non_blocking=True)
    return moved


def forward_model(model, batch):
    return model(batch["image"], batch["morphology_map"])


def predict(model, loader: DataLoader, device: torch.device, show_progress=False, desc="Prediction", mininterval=1.0):
    model.eval()
    out = {k: [] for k in ("sample_id","patient_id","os_time","os_event","os_risk","dfs_time","dfs_event","dfs_risk")}
    iterator = tqdm(loader, total=len(loader), desc=desc, unit="batch", leave=False, mininterval=mininterval, dynamic_ncols=True) if show_progress else loader
    with torch.inference_mode():
        for raw in iterator:
            batch = move_batch(raw, device)
            os_risk, dfs_risk = forward_model(model, batch)
            out["sample_id"].extend(raw["sample_id"])
            out["patient_id"].extend(raw["patient_id"])
            for key, values in {
                "os_time": batch["os_time"], "os_event": batch["os_event"], "os_risk": os_risk,
                "dfs_time": batch["dfs_time"], "dfs_event": batch["dfs_event"], "dfs_risk": dfs_risk,
            }.items():
                out[key].extend(values.detach().cpu().reshape(-1).tolist())
    return {k: v if k in {"sample_id","patient_id"} else np.asarray(v, dtype=np.float64) for k,v in out.items()}


def aggregate_by_patient(pred, method="mean"):
    if method not in {"mean","median","max"}:
        raise ValueError(f"Unsupported patient aggregation: {method}")
    pids = np.asarray(pred["patient_id"], dtype=str)
    reducer = {"mean":np.mean,"median":np.median,"max":np.max}[method]
    out = {k: [] for k in pred}
    for pid in list(dict.fromkeys(pids.tolist())):
        idx = np.flatnonzero(pids == pid)
        out["sample_id"].append("|".join(str(pred["sample_id"][i]) for i in idx))
        out["patient_id"].append(pid)
        for endpoint in ("os","dfs"):
            times = np.asarray(pred[f"{endpoint}_time"])[idx]
            events = np.asarray(pred[f"{endpoint}_event"])[idx]
            if not np.allclose(times, times[0], rtol=0, atol=1e-6) or not np.all(events == events[0]):
                raise ValueError(f"Inconsistent {endpoint.upper()} outcomes across slides for patient {pid}")
            out[f"{endpoint}_time"].append(float(times[0]))
            out[f"{endpoint}_event"].append(float(events[0]))
            out[f"{endpoint}_risk"].append(float(reducer(np.asarray(pred[f"{endpoint}_risk"])[idx])))
    return {k: v if k in {"sample_id","patient_id"} else np.asarray(v,dtype=np.float64) for k,v in out.items()}


def cox_losses(pred, aggregation="mean", ties="efron"):
    p = aggregate_by_patient(pred, aggregation)
    t = {k: torch.as_tensor(np.asarray(p[k]), dtype=torch.float64) for k in ("os_risk","dfs_risk","os_time","dfs_time","os_event","dfs_event")}
    os_loss = cox_ph_loss(t["os_risk"], t["os_time"], t["os_event"], ties)
    dfs_loss = cox_ph_loss(t["dfs_risk"], t["dfs_time"], t["dfs_event"], ties)
    return {"os_cox_loss":float(os_loss), "dfs_cox_loss":float(dfs_loss), "joint_cox_loss":float(os_loss+dfs_loss)}


def summarize(pred, aggregation="mean"):
    p = aggregate_by_patient(pred, aggregation)
    os_c = concordance_index(p["os_time"], p["os_risk"], p["os_event"])
    dfs_c = concordance_index(p["dfs_time"], p["dfs_risk"], p["dfs_event"])
    valid = [x for x in (os_c,dfs_c) if math.isfinite(x)]
    return {
        "n_samples": len(pred["sample_id"]), "n_patients": len(p["patient_id"]),
        "os_events": int(np.sum(p["os_event"])), "dfs_events": int(np.sum(p["dfs_event"])),
        "os_cindex": float(os_c), "dfs_cindex": float(dfs_c),
        "mean_cindex": float(np.mean(valid)) if valid else float("nan"),
    }


def save_predictions(pred, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields=("sample_id","patient_id","os_time","os_event","os_risk","dfs_time","dfs_event","dfs_risk")
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(fields)
        for row in zip(*(pred[k] for k in fields)): w.writerow(row)


def _capture_rng(device):
    return torch.get_rng_state().clone(), torch.cuda.get_rng_state_all() if device.type == "cuda" else None


def _restore_rng(state, device):
    cpu, cuda = state; torch.set_rng_state(cpu)
    if device.type == "cuda" and cuda is not None: torch.cuda.set_rng_state_all(cuda)


def _aggregate_train(patient_ids, os_risk, dfs_risk, os_time, dfs_time, os_event, dfs_event, method):
    unique=list(dict.fromkeys(patient_ids))
    def reduce(x):
        return x.mean() if method=="mean" else (x.median() if method=="median" else x.max())
    result=[[] for _ in range(6)]
    outcomes=(os_time,dfs_time,os_event,dfs_event)
    for pid in unique:
        idx=torch.tensor([i for i,v in enumerate(patient_ids) if v==pid], dtype=torch.long, device=os_risk.device)
        result[0].append(reduce(os_risk[idx])); result[1].append(reduce(dfs_risk[idx]))
        for offset, values in enumerate(outcomes,start=2):
            pv=values[idx]
            if not torch.allclose(pv, pv[0].expand_as(pv), rtol=0, atol=1e-6):
                raise ValueError(f"Inconsistent outcome across slides for patient {pid}")
            result[offset].append(pv[0])
    return tuple(torch.stack(x) for x in result)


def train_full_risk_update(model, loader, optimizer, device, scaler, use_amp, gradient_clip, ties="efron", aggregation="mean", show_progress=False, prefix="Train", mininterval=1.0):
    model.train(); model.enforce_frozen_backbone_eval(); optimizer.zero_grad(set_to_none=True)
    os_scores=[]; dfs_scores=[]; os_times=[]; dfs_times=[]; os_events=[]; dfs_events=[]; patient_ids=[]; rng=[]
    iterator=tqdm(loader,total=len(loader),desc=f"{prefix} · Pass 1/2 risk",unit="batch",leave=False,mininterval=mininterval,dynamic_ncols=True) if show_progress else loader
    with torch.no_grad():
        for raw in iterator:
            batch=move_batch(raw,device); rng.append(_capture_rng(device))
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=use_amp):
                os_risk,dfs_risk=forward_model(model,batch)
            os_scores.append(os_risk.detach().float().reshape(-1)); dfs_scores.append(dfs_risk.detach().float().reshape(-1))
            os_times.append(batch["os_time"].float().reshape(-1)); dfs_times.append(batch["dfs_time"].float().reshape(-1))
            os_events.append(batch["os_event"].float().reshape(-1)); dfs_events.append(batch["dfs_event"].float().reshape(-1))
            patient_ids.extend(str(x) for x in raw["patient_id"])
    full_os=torch.cat(os_scores).requires_grad_(True); full_dfs=torch.cat(dfs_scores).requires_grad_(True)
    inputs=_aggregate_train(patient_ids,full_os,full_dfs,torch.cat(os_times),torch.cat(dfs_times),torch.cat(os_events),torch.cat(dfs_events),aggregation)
    loss=multitask_cox_loss(*inputs,ties=ties); os_grad,dfs_grad=torch.autograd.grad(loss,(full_os,full_dfs))
    iterator=zip(loader,rng)
    if show_progress: iterator=tqdm(iterator,total=len(rng),desc=f"{prefix} · Pass 2/2 replay",unit="batch",leave=False,mininterval=mininterval,dynamic_ncols=True)
    offset=0
    for raw,state in iterator:
        batch=move_batch(raw,device); _restore_rng(state,device)
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=use_amp):
            os_risk,dfs_risk=forward_model(model,batch); n=os_risk.numel()
            surrogate=(os_risk.float().reshape(-1)*os_grad[offset:offset+n]).sum()+(dfs_risk.float().reshape(-1)*dfs_grad[offset:offset+n]).sum()
        scaler.scale(surrogate).backward(); offset+=n
    if offset != full_os.numel(): raise RuntimeError("Replay loader order/length changed")
    if gradient_clip:
        scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),float(gradient_clip))
    scaler.step(optimizer); scaler.update()
    return float(loss.detach().cpu())
