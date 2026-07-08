r"""Transaction-cost model.

Ignoring costs is the fastest way to turn a losing strategy into a "profitable" backtest,
so the cost model is first-class. Total cost of trading ``Q`` shares at price ``p`` is the
sum of three terms:

.. math::

    \underbrace{\max(c_{\min},\ c_{\text{ps}} \cdot Q)}_{\text{commission}}
    \;+\;
    \underbrace{\tfrac{s}{2}\, p\, Q}_{\text{half-spread}}
    \;+\;
    \underbrace{\eta\, \sigma\, \sqrt{Q/\text{ADV}}\; p\, Q}_{\text{market impact}}.

* **Commission** — a per-share fee with a per-order minimum.
* **Half-spread** — crossing half the bid-ask spread, expressed in basis points ``s`` of
  price. (A marketable order pays roughly half the quoted spread versus the mid.)
* **Market impact** — the square-root law: impact as a *fraction* of price scales like
  :math:`\eta \sigma \sqrt{Q/\text{ADV}}`, where :math:`\sigma` is daily return volatility
  and ADV is average daily volume. Multiplying by notional ``pQ`` gives dollars. This is
  the standard, empirically-motivated form used across the industry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Parameters of the three-component transaction-cost model.

    Attributes
    ----------
    commission_per_share:
        Per-share commission (currency units).
    commission_min:
        Minimum commission charged per order.
    half_spread_bps:
        Half the bid-ask spread in basis points of price (1 bp = 1e-4).
    impact_eta:
        Dimensionless coefficient of the square-root impact law.
    """

    commission_per_share: float = 0.005
    commission_min: float = 1.0
    half_spread_bps: float = 2.0
    impact_eta: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "commission_per_share",
            "commission_min",
            "half_spread_bps",
            "impact_eta",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def cost(
        self,
        quantity: float,
        price: float,
        volatility: float | None = None,
        adv: float | None = None,
    ) -> float:
        """Total transaction cost of trading ``quantity`` shares at ``price``.

        ``volatility`` (daily, fractional) and ``adv`` drive the market-impact term; if
        either is missing or non-positive, impact is taken as zero (commission and spread
        still apply). Cost is symmetric in the sign of ``quantity``.
        """
        q = abs(quantity)
        if q == 0.0 or price <= 0.0:
            return 0.0

        commission = max(self.commission_min, self.commission_per_share * q)
        half_spread = self.half_spread_bps * 1e-4 * price * q

        impact = 0.0
        if volatility is not None and adv is not None and volatility > 0.0 and adv > 0.0:
            impact_fraction = self.impact_eta * volatility * math.sqrt(q / adv)
            impact = impact_fraction * price * q

        return commission + half_spread + impact
