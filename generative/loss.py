import torch 
import torch.nn.functional as F
import torch.nn as nn

def dafl_kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_kl = F.kl_div(p, q, reduction='batchmean')
    return l_kl

def kl_loss(logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits/T, dim=1)
    p = F.softmax(targets/T, dim=1)
    return F.kl_div(q, p, reduction=reduction) * (T*T)

class KLDiv(nn.Module):
    def __init__(self, T=1.0, reduction='batchmean'):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kl_loss(logits, targets, T=self.T, reduction=self.reduction)


# --------------------------------------------------------------------------- #
#  Energy-weighted knowledge distillation                                      #
#                                                                              #
#  Instead of forcing the *generator* to minimise the free energy (which is    #
#  redundant with L_OH / L_CLS and gameable by logit explosion), we use the    #
#  teacher free energy as a read-only, per-sample *reliability* signal in the  #
#  KD step. Low-energy (in-distribution) synthetic samples are trusted; high-  #
#  energy (OOD / garbage) ones are discounted. The gate is self-calibrating    #
#  per batch (no energy targets, no margins), so it is much lighter than the   #
#  KDCI confounder-correction machinery while addressing the same bias.        #
# --------------------------------------------------------------------------- #

def teacher_energy(logits, T=1.0):
    """Free energy E(x) = -T * logsumexp(f(x)/T). Lower energy => more
    in-distribution (higher density under the teacher)."""
    return -T * torch.logsumexp(logits / T, dim=1)

def energy_kd_weights(logits, beta=1.0, T=1.0, eps=1e-8):
    """Per-sample KD weights from teacher free energy.

    w_i = sigmoid( -(E_i - mu_B) / (beta * sigma_B) ), then renormalised to
    mean 1 over the batch so the overall loss scale (and effective LR) is
    preserved. mu_B / sigma_B are the batch energy mean / std, making the gate
    parameter-light: the only knob is `beta` (gate sharpness, default 1).

    Returns a detached weight vector of shape [B]; weights are a curation
    signal, not something to backpropagate through.
    """
    energy = teacher_energy(logits, T=T).detach()
    mu = energy.mean()
    sigma = energy.std().clamp_min(eps)
    w = torch.sigmoid(-(energy - mu) / (beta * sigma))
    w = w * (w.numel() / (w.sum() + eps))  # normalise to mean 1
    return w

def weighted_kl(s_out, t_out, weights=None, T=1.0):
    """Per-sample KD (KL(teacher || student) at temperature T) optionally
    reweighted by `weights`, then averaged over the batch."""
    q = F.log_softmax(s_out / T, dim=1)
    p = F.softmax(t_out / T, dim=1)
    per_sample = F.kl_div(q, p, reduction='none').sum(1) * (T * T)  # [B]
    if weights is not None:
        per_sample = per_sample * weights
    return per_sample.mean()


# --------------------------------------------------------------------------- #
#  Energy-adaptive distillation temperature                                    #
#                                                                              #
#  In DFKD the teacher is overconfident on synthetic samples, so its soft      #
#  targets collapse towards one-hot and the dark knowledge vanishes (measured  #
#  teacher entropy ~0.05). A *global* KD temperature cannot fix this because   #
#  the saturation is heterogeneous across samples. The free energy E(x) is the #
#  exact, parameter-free measure of per-sample saturation (energy is the       #
#  temperature-scaled logsumexp), so we use it to set a per-sample temperature #
#  that softens the more saturated samples more, restoring inter-class         #
#  structure uniformly across the batch. Unlike energy *weighting*, this does  #
#  not discard samples -- it recovers the information in each one.             #
# --------------------------------------------------------------------------- #

def energy_adaptive_kl(s_out, t_out, tau_base=4.0, alpha=1.0, eps=1e-8):
    """Per-sample temperature KD, with tau driven by the teacher free energy.

    tau_i = tau_base * exp(-alpha * z_i),  z_i = (E_i - mu_E) / sigma_E
    (more saturated => lower energy => higher tau => more softening).
    `alpha = 0` recovers standard global-temperature KD at `tau_base`, which is
    the natural ablation. tau is clamped to [tau_base/4, tau_base*4] for
    stability (no extra hyper-parameters).
    """
    energy = teacher_energy(t_out, T=1.0).detach()                      # [B]
    z = (energy - energy.mean()) / energy.std().clamp_min(eps)          # [B]
    tau = (tau_base * torch.exp(-alpha * z)).clamp(tau_base * 0.25, tau_base * 4.0)
    tau = tau.unsqueeze(1)                                              # [B, 1]
    p = F.softmax(t_out / tau, dim=1)
    logq = F.log_softmax(s_out / tau, dim=1)
    per_sample = F.kl_div(logq, p, reduction='none').sum(1) * (tau.squeeze(1) ** 2)
    return per_sample.mean()