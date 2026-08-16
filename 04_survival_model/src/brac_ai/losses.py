from __future__ import annotations

import torch


def cox_ph_loss(
    log_risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    ties: str = "efron",
) -> torch.Tensor:
    """Negative Cox partial log-likelihood averaged over observed events."""
    if ties not in {"breslow", "efron"}:
        raise ValueError(f"Unsupported Cox ties method: {ties}")

    log_risk = log_risk.reshape(-1)
    time = time.reshape(-1).to(device=log_risk.device)
    event = event.reshape(-1).to(device=log_risk.device, dtype=log_risk.dtype)
    if not (log_risk.numel() == time.numel() == event.numel()):
        raise ValueError("log_risk, time and event must contain the same number of observations")
    if log_risk.numel() == 0:
        raise ValueError("Cox loss requires at least one observation")

    event_mask = event > 0.5
    n_events = event_mask.sum()
    if int(n_events.detach().cpu()) == 0:
        return log_risk.sum() * 0.0

    pll = log_risk.new_zeros(())
    for event_time in torch.unique(time[event_mask]):
        tied_events = event_mask & (time == event_time)
        risk_set = time >= event_time
        deaths = int(tied_events.sum().detach().cpu())
        log_risk_set_sum = torch.logsumexp(log_risk[risk_set], dim=0)
        pll = pll + log_risk[tied_events].sum()

        if ties == "breslow" or deaths == 1:
            pll = pll - deaths * log_risk_set_sum
            continue

        log_tied_risk_sum = torch.logsumexp(log_risk[tied_events], dim=0)
        tied_fraction = torch.exp(log_tied_risk_sum - log_risk_set_sum).clamp(max=1.0)
        for rank in range(deaths):
            if rank == 0:
                denominator = log_risk_set_sum
            else:
                reduction = ((rank / deaths) * tied_fraction).clamp(
                    max=1.0 - torch.finfo(log_risk.dtype).eps
                )
                denominator = log_risk_set_sum + torch.log1p(-reduction)
            pll = pll - denominator
    return -pll / n_events.to(dtype=log_risk.dtype)


def multitask_cox_loss(
    os_risk: torch.Tensor,
    dfs_risk: torch.Tensor,
    os_time: torch.Tensor,
    dfs_time: torch.Tensor,
    os_event: torch.Tensor,
    dfs_event: torch.Tensor,
    ties: str = "efron",
) -> torch.Tensor:
    return cox_ph_loss(os_risk, os_time, os_event, ties=ties) + cox_ph_loss(
        dfs_risk, dfs_time, dfs_event, ties=ties
    )
