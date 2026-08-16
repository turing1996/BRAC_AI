from __future__ import annotations
import argparse, csv, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from .checkpoint import load_checkpoint, save_checkpoint
from .config import load_config
from .data import assert_patient_disjoint, discover_ids, make_dataset
from .engine import aggregate_by_patient, cox_losses, predict, save_predictions, summarize, train_full_risk_update
from .model import build_model
from .plotting import plt


def seed_everything(seed, deterministic=True):
    if deterministic: os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=deterministic; torch.backends.cudnn.benchmark=not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=False)


def _seed_worker(worker_id):
    seed=torch.initial_seed()%(2**32); np.random.seed(seed); random.seed(seed)


def make_loader(dataset,batch_size,workers,device,seed):
    g=torch.Generator().manual_seed(seed)
    return DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=workers,pin_memory=device.type=="cuda",persistent_workers=workers>0,worker_init_fn=_seed_worker,generator=g)


def build_optimizer(model,t):
    groups={"backbone":[],"morphology":[],"heads":[]}
    for name,p in model.named_parameters():
        if name.startswith("ViTbranch1."): groups["backbone"].append(p)
        elif name.startswith("morphology_encoder."): groups["morphology"].append(p)
        else: groups["heads"].append(p)
    return torch.optim.AdamW([
        {"params":groups["backbone"],"lr":float(t["backbone_learning_rate"]),"group_name":"backbone"},
        {"params":groups["morphology"],"lr":float(t["morphology_learning_rate"]),"group_name":"morphology"},
        {"params":groups["heads"],"lr":float(t["head_learning_rate"]),"group_name":"heads"},
    ],weight_decay=float(t["weight_decay"]))


def configure_stage(model,t,epoch):
    if epoch <= int(t["freeze_backbone_epochs"]):
        model.configure_backbone_trainability(0); return "warmup"
    model.configure_backbone_trainability(int(t["unfreeze_last_blocks"])); return "fine_tune"


def trainable_params(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def lrs(opt): return {str(g["group_name"]):float(g["lr"]) for g in opt.param_groups}

def fmt(sec):
    sec=max(0,int(round(sec))); h,r=divmod(sec,3600); m,s=divmod(r,60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def write_history(history,out,curves=True):
    if not history:return
    tmp=out/"history.csv.tmp"
    with tmp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(history[0].keys())); w.writeheader(); w.writerows(history); f.flush(); os.fsync(f.fileno())
    tmp.replace(out/"history.csv")
    if curves:
        ep=[r["epoch"] for r in history]
        fig,ax=plt.subplots(1,3,figsize=(15,4))
        ax[0].plot(ep,[r["train_joint_cox_loss"] for r in history]); ax[0].set(title="Training",xlabel="Epoch",ylabel="Joint Cox loss")
        ax[1].plot(ep,[r["validation_joint_cox_loss"] for r in history]); ax[1].set(title="TCGA validation",xlabel="Epoch",ylabel="Joint Cox loss")
        ax[2].plot(ep,[r["validation_mean_cindex"] for r in history]); ax[2].set(title="TCGA validation",xlabel="Epoch",ylabel="Mean C-index")
        fig.tight_layout(); fig.savefig(out/"training_curves.png",dpi=180); plt.close(fig)


def run(config, no_progress=False):
    t=config["training"]; seed=int(t["seed"]); seed_everything(seed,bool(t.get("deterministic_algorithms",True)))
    device=torch.device("cuda" if str(t.get("device","auto"))=="auto" and torch.cuda.is_available() else ("cpu" if str(t.get("device","auto"))=="auto" else str(t["device"])))
    train_root=Path(config["data"]["tcga_train"]); val_root=Path(config["data"]["tcga_validation"])
    train_ids=discover_ids(train_root); val_ids=discover_ids(val_root); assert_patient_disjoint(train_ids,val_ids)
    train_set=make_dataset(train_root,config,train_ids); val_set=make_dataset(val_root,config,val_ids)
    workers=int(t["num_workers"]); train_loader=make_loader(train_set,int(t["batch_size"]),workers,device,seed); val_loader=make_loader(val_set,int(t["eval_batch_size"]),workers,device,seed+1)
    out=Path(t["output_dir"])
    if out.exists() and any(out.iterdir()) and not bool(t.get("allow_existing_output",False)):
        raise FileExistsError(f"Output directory is not empty: {out}")
    out.mkdir(parents=True,exist_ok=True)
    model=build_model(config["model"],initialize_pretrained=True).to(device); opt=build_optimizer(model,t)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",patience=int(t["lr_patience"]),factor=float(t["lr_factor"]))
    use_amp=bool(t.get("amp",True) and device.type=="cuda"); scaler=torch.amp.GradScaler(device.type,enabled=use_amp)
    show=bool(t.get("progress",{}).get("enabled",True)) and not no_progress; mininterval=float(t.get("progress",{}).get("mininterval_seconds",1.0))
    total=int(t["epochs"]); updates=int(t["full_risk_updates_per_epoch"]); agg=str(t.get("patient_risk_aggregation","mean")); ties=str(t.get("cox_ties","efron"))
    print(f"Device={device} | model=morphology_mlp | train={len(train_set)} | validation={len(val_set)} | updates/epoch={updates}",flush=True)
    best=math.inf; best_path=out/"best.pt"; no_improve=0; history=[]; start=time.perf_counter()
    for epoch in range(1,total+1):
        es=time.perf_counter(); stage=configure_stage(model,t,epoch); tp=trainable_params(model); update_losses=[]
        for u in range(1,updates+1):
            loss=train_full_risk_update(model,train_loader,opt,device,scaler,use_amp,t.get("gradient_clip"),ties,agg,show,f"E{epoch:02d} U{u}/{updates}",mininterval)
            update_losses.append(loss)
            if show:tqdm.write(f"E{epoch:02d} full-risk update {u}/{updates} | joint Cox loss {loss:.5f}")
        train_loss=float(np.mean(update_losses))
        vp=predict(model,val_loader,device,show,f"E{epoch:02d} validation",mininterval); vl=cox_losses(vp,agg,ties); vm=summarize(vp,agg); vjoint=float(vl["joint_cox_loss"])
        scheduler.step(vjoint if math.isfinite(vjoint) else math.inf)
        improved=math.isfinite(vjoint) and vjoint < best-float(t.get("early_stopping_min_delta",0.0))
        if improved:
            best=vjoint; no_improve=0; save_checkpoint(best_path,model,epoch,config,{**vl,**vm,"selection_metric":"joint_cox_loss"})
        else:no_improve+=1
        lr=lrs(opt); sec=time.perf_counter()-es; elapsed=time.perf_counter()-start; eta=elapsed/epoch*max(0,total-epoch)
        row={
            "epoch":epoch,"training_stage":stage,"train_joint_cox_loss":train_loss,"validation_joint_cox_loss":vjoint,
            "validation_os_cox_loss":vl["os_cox_loss"],"validation_dfs_cox_loss":vl["dfs_cox_loss"],
            "validation_os_cindex":vm["os_cindex"],"validation_dfs_cindex":vm["dfs_cindex"],"validation_mean_cindex":vm["mean_cindex"],
            "is_best":improved,"best_validation_joint_cox_loss":best,"epochs_without_improvement":no_improve,
            "trainable_parameters":tp,"backbone_learning_rate":lr["backbone"],"morphology_learning_rate":lr["morphology"],"head_learning_rate":lr["heads"],
            "cross_attention_gamma":float(model.cross_attention.gamma.detach().cpu().item()),"epoch_seconds":sec,
        }
        history.append(row); write_history(history,out,bool(t.get("progress",{}).get("update_curves_each_epoch",True)))
        summary=(f"Epoch {epoch:02d}/{total} | {stage} | train {train_loss:.5f} | val {vjoint:.5f}"+(" | BEST" if improved else "")+
                 f" | OS C {vm['os_cindex']:.3f} | DFS C {vm['dfs_cindex']:.3f} | mean C {vm['mean_cindex']:.3f}"+
                 f" | LR bb {lr['backbone']:.1e} morph {lr['morphology']:.1e} head {lr['heads']:.1e}"+
                 f" | trainable {tp/1e6:.2f}M | gamma {row['cross_attention_gamma']:.3f} | time {fmt(sec)} | ETA {fmt(eta)}")
        tqdm.write(summary) if show else print(summary,flush=True)
        if int(t.get("early_stopping_patience",0)) and no_improve>=int(t["early_stopping_patience"]):
            print(f"Early stopping at epoch {epoch}; best validation joint Cox loss={best:.6f}",flush=True); break
    if not best_path.is_file(): raise RuntimeError("No valid checkpoint was produced")
    load_checkpoint(model,best_path,device)
    for name,loader in (("training",train_loader),("validation",val_loader)):
        p=predict(model,loader,device,show,f"Final {name}",mininterval); pp=aggregate_by_patient(p,agg)
        save_predictions(p,out/f"{name}_predictions_samples.csv"); save_predictions(pp,out/f"{name}_predictions.csv")
        if name=="validation":
            metrics={**cox_losses(p,agg,ties),**summarize(p,agg),"selection_metric":"joint_cox_loss"}
            (out/"validation_metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=True),encoding="utf-8")
    return best_path


def main():
    parser=argparse.ArgumentParser(description="Train minimal BRAC morphology-MLP survival model")
    parser.add_argument("--config",default="config.yaml"); parser.add_argument("--no-progress",action="store_true"); args=parser.parse_args()
    run(load_config(args.config),args.no_progress)

if __name__=="__main__":main()
