from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F


# --------- Base interface ---------
class HFunc(ABC):
    """
    Contract:
      - input: logR (B, T-1), where R = exp(logR) > 0
      - output: per-element loss, same shape
    """

    l_logr: float = -30.0
    u_logr: float = 30.0

    def loss_from_logR(self, logR: torch.Tensor) -> torch.Tensor:
        # Default fallback: compute Eq(10) literally using h(R), h'(R), h'(1/R)
        # L_h = h'(R)R - h(R) - h'(1/R)
        logR = self._clamp_logR(logR)
        R = torch.exp(logR)
        invR = torch.exp(-logR)
        return self.h_prime(R) * R - self.h(R) - self.h_prime(invR)

    @abstractmethod
    def h(self, R: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def h_prime(self, R: torch.Tensor) -> torch.Tensor: ...

    def _clamp_logR(self, logR: torch.Tensor) -> torch.Tensor:
        # Prevent exp overflow for power-based instances
        return torch.clamp(logR, self.l_logr, self.u_logr)


# --------- Concrete instances (Table 2) ---------
# These are implemented in a numerically stable way where possible.
# Table 2 lists LR(DPO), KLIEP, LSIF, BA, SBA forms.


@dataclass(frozen=True)
class LR_DPO(HFunc):
    # L = E[ log(1 + R) ]  (recovers DPO)
    l_logr: float = -30.0
    u_logr: float = 30.0

    def loss_from_logR(self, logR: torch.Tensor) -> torch.Tensor:
        logR = self._clamp_logR(logR)
        return F.softplus(logR)  # log(1+exp(logR)) == log(1+R)

    def h(self, R: torch.Tensor) -> torch.Tensor:
        # Not used
        return R

    def h_prime(self, R: torch.Tensor) -> torch.Tensor:
        # Not used
        return torch.ones_like(R)


@dataclass(frozen=True)
class KLIEP(HFunc):
    # L = E[ R + log R ]
    l_logr: float = -30.0
    u_logr: float = 30.0

    def loss_from_logR(self, logR: torch.Tensor) -> torch.Tensor:
        logR = self._clamp_logR(logR)
        return torch.exp(logR) + logR

    def h(self, R: torch.Tensor) -> torch.Tensor:
        return R * torch.log(R) - R

    def h_prime(self, R: torch.Tensor) -> torch.Tensor:
        return torch.log(R)


@dataclass(frozen=True)
class LSIF(HFunc):
    # L = E[ R^2 - 2R ]
    l_logr: float = -30.0
    u_logr: float = 30.0

    def loss_from_logR(self, logR: torch.Tensor) -> torch.Tensor:
        logR = self._clamp_logR(logR)
        R = torch.exp(logR)
        return R * R - 2.0 * R

    def h(self, R: torch.Tensor) -> torch.Tensor:
        return (R - 1.0) ** 2

    def h_prime(self, R: torch.Tensor) -> torch.Tensor:
        return 2.0 * (R - 1.0)


@dataclass(frozen=True)
class BA(HFunc):
    # BA: Table 2 uses lambda > -1 and provides a closed-form L_h.
    lam: float  # λ
    l_logr: float = -30.0
    u_logr: float = 30.0

    def loss_from_logR(self, logR: torch.Tensor) -> torch.Tensor:
        # L = E[ R^{λ+1} - ((λ+1)/λ) * R^{-λ} ]
        if abs(self.lam) < 1e-6:
            return KLIEP().loss_from_logR(logR)  # BA(0) -> KLIEP per paper discussion
        logR = self._clamp_logR(logR)
        lam = self.lam
        term1 = torch.exp((lam + 1.0) * logR)  # R^{λ+1}
        term2 = torch.exp((-lam) * logR)  # R^{-λ}
        return term1 - ((lam + 1.0) / lam) * term2

    def h(self, R: torch.Tensor) -> torch.Tensor:
        lam = self.lam
        return (R ** (1.0 + lam) - R) / lam

    def h_prime(self, R: torch.Tensor) -> torch.Tensor:
        lam = self.lam
        return ((1.0 + lam) * (R**lam) - 1.0) / lam


@dataclass(frozen=True)
class SBA(HFunc):
    # SBA: scaled BA with scale s (paper suggests s=4 to match DPO scale at init).
    lam: float
    s: float = 4.0
    l_logr: float = -30.0
    u_logr: float = 30.0

    def loss_from_logR(self, logR: torch.Tensor) -> torch.Tensor:
        # L = E[ (1/(s(λ+1))) R^{λ+1} - (1/(sλ)) R^{-λ} ]
        if abs(self.lam) < 1e-6:
            # λ→0 limit: scaled KLIEP-ish; simplest is just scale KLIEP by 1/s
            return (1.0 / self.s) * KLIEP().loss_from_logR(logR)
        logR = self._clamp_logR(logR)
        lam, s = self.lam, self.s
        term1 = torch.exp((lam + 1.0) * logR) / (s * (lam + 1.0))
        term2 = torch.exp((-lam) * logR) / (s * lam)
        return term1 - term2

    def h(self, R: torch.Tensor) -> torch.Tensor:
        lam, s = self.lam, self.s
        return (R ** (1.0 + lam) - R) / (s * lam * (lam + 1.0))

    def h_prime(self, R: torch.Tensor) -> torch.Tensor:
        lam, s = self.lam, self.s
        return (((1.0 + lam) * (R**lam) - 1.0) / lam) / (s * (lam + 1.0))


# --------- Registry / factory ---------
def make_h(name: str, **kwargs) -> HFunc:
    name = name.lower()
    l_logr = float(kwargs.get("l_logr", -30.0))
    u_logr = float(kwargs.get("u_logr", 30.0))

    if name in ("lr", "dpo", "logistic"):
        return LR_DPO(l_logr=l_logr, u_logr=u_logr)
    if name == "kliep":
        return KLIEP(l_logr=l_logr, u_logr=u_logr)
    if name == "lsif":
        return LSIF(l_logr=l_logr, u_logr=u_logr)
    if name == "ba":
        return BA(lam=float(kwargs["lam"]), l_logr=l_logr, u_logr=u_logr)
    if name == "sba":
        return SBA(
            lam=float(kwargs["lam"]),
            s=float(kwargs.get("s", 4.0)),
            l_logr=l_logr,
            u_logr=u_logr,
        )
    raise ValueError(f"Unknown h: {name}")
