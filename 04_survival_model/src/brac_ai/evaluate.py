from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from .checkpoint import load_checkpoint
from .config import load_config
from .data import discover_ids, make_dataset
from .engine import aggregate_by_patient, cox_losses, predict, save_predictions, summarize
from .model import build_model
from .train import make_loader


def main():
    p=argparse.ArgumentParser(description="Evaluate minimal morphology-MLP checkpoint")
    p.add_argument("--config",default="config.yaml"); p.add_argument("--checkpoint",required=True)
    p.add_argument("--cohort",choices=("tcga_train","tcga_validation","external"),required=True); p.add_argument("--output"); p.add_argument("--no-progress",action="store_true")
    a=p.parse_args(); cfg=load_config(a.config); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=build_model(cfg["model"],initialize_pretrained=False).to(device); load_checkpoint(model,a.checkpoint,device)
    root=Path(cfg["data"][a.cohort]); ids=discover_ids(root); ds=make_dataset(root,cfg,ids)
    loader=make_loader(ds,int(cfg["training"]["eval_batch_size"]),int(cfg["training"]["num_workers"]),device,int(cfg["training"]["seed"])+100)
    show=bool(cfg["training"].get("progress",{}).get("enabled",True)) and not a.no_progress
    pred=predict(model,loader,device,show,f"Evaluate {a.cohort}",float(cfg["training"].get("progress",{}).get("mininterval_seconds",1.0)))
    agg=str(cfg["training"].get("patient_risk_aggregation","mean")); ties=str(cfg["training"].get("cox_ties","efron")); patient=aggregate_by_patient(pred,agg)
    out=Path(a.output or Path(cfg["training"]["output_dir"])/f"evaluation_{a.cohort}"); out.mkdir(parents=True,exist_ok=True)
    save_predictions(pred,out/"predictions_samples.csv"); save_predictions(patient,out/"predictions.csv")
    metrics={**cox_losses(pred,agg,ties),**summarize(pred,agg)}; (out/"metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=True),encoding="utf-8")
    print(json.dumps(metrics,indent=2,allow_nan=True))

if __name__=="__main__":main()
