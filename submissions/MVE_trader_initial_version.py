"""
trader_prosperity4.py  —  IMC Prosperity 4 submission
======================================================
HOW TO UPDATE PARAMS:
  1. Run calibrate_strategy.ipynb (Kernel → Restart & Run All)
  2. The final cell prints a block starting with:
         PARAMS: Dict[str, dict] = {
  3. Copy that entire block and paste it here, replacing the PARAMS
     block below. Every key name matches exactly — no restructuring needed.
"""

import json
import math
from typing import Dict, List, Tuple

from datamodel import OrderDepth, TradingState, Order


# ══════════════════════════════════════════════════════════════════════════
# PARAMS  ←  paste calibration output here (replace this entire block)
# ══════════════════════════════════════════════════════════════════════════
PARAMS: Dict[str, dict] = {
    "EMERALDS": {
        "product": "EMERALDS",
        "recommended_strategy": "market_making",
        "use_ml_signal": True,
        "fair_value": 10000.0,
        "fair_value_std": 0.7233,
        "return_autocorr_1": -0.4875,
        "mid_p01": 9996.0,
        "mid_p99": 10004.0,
        "quote_offset": 8,
        "base_quantity": 3,
        "bot_bid": 9992,
        "bot_ask": 10008,
        "obi_skew_threshold": 0.0,
        "obi_skew_ticks": 1,
        "inventory_lean": 0.4,
        "max_position": 60,
        "position_limit_hard": 80,
        "zscore_entry": 0.224,
        "zscore_exit": 0.0,
        "zscore_stop": 4.249,
        "ema_cross_confirm": 0.0782,
        "ml_auc": 0.786,
        "signal_threshold_base": 0.017,
        "signal_threshold_strong": 0.023,
        "top_features": [
            {
                "feature": "ask_volume_1",
                "importance": 0.01141
            },
            {
                "feature": "ask_price_1",
                "importance": 0.01122
            },
            {
                "feature": "ema_cross",
                "importance": 0.00931
            },
            {
                "feature": "ofi_10",
                "importance": 0.00845
            },
            {
                "feature": "obi_mean_10",
                "importance": 0.00778
            },
            {
                "feature": "total_bid_depth",
                "importance": 0.00681
            },
            {
                "feature": "total_ask_depth",
                "importance": 0.0063
            },
            {
                "feature": "volatility_10",
                "importance": 0.0045
            },
            {
                "feature": "ofi_20",
                "importance": 0.002
            },
            {
                "feature": "volatility_20",
                "importance": 0.00196
            },
            {
                "feature": "volatility_5",
                "importance": 0.00063
            },
            {
                "feature": "depth_imbalance",
                "importance": 0.00046
            },
            {
                "feature": "momentum_10",
                "importance": 6e-05
            }
        ]
    },
    "TOMATOES": {
        "product": "TOMATOES",
        "recommended_strategy": "market_making",
        "use_ml_signal": True,
        "fair_value": 4992.76,
        "fair_value_std": 19.7471,
        "return_autocorr_1": -0.4203,
        "mid_p01": 4952.0,
        "mid_p99": 5030.0,
        "quote_offset": 6,
        "base_quantity": 3,
        "bot_bid": 4987,
        "bot_ask": 5001,
        "obi_skew_threshold": 0.0,
        "obi_skew_ticks": 1,
        "inventory_lean": 0.4,
        "max_position": 60,
        "position_limit_hard": 80,
        "zscore_entry": 1.856,
        "zscore_exit": 0.635,
        "zscore_stop": 3.147,
        "ema_cross_confirm": 0.7946,
        "ml_auc": 0.5872,
        "signal_threshold_base": 0.756,
        "signal_threshold_strong": 0.849,
        "top_features": [
            {
                "feature": "ask_volume_2",
                "importance": 0.04681
            },
            {
                "feature": "ofi_20",
                "importance": 0.01044
            },
            {
                "feature": "ask_gap_12",
                "importance": 0.00349
            },
            {
                "feature": "ofi_10",
                "importance": 0.0025
            },
            {
                "feature": "ask_price_1",
                "importance": 0.00228
            },
            {
                "feature": "spread_pct",
                "importance": 0.00186
            },
            {
                "feature": "momentum_5",
                "importance": 0.00155
            },
            {
                "feature": "bid_gap_12",
                "importance": 0.00151
            },
            {
                "feature": "bid_volume_2",
                "importance": 0.00115
            },
            {
                "feature": "bid_price_1",
                "importance": 0.00077
            },
            {
                "feature": "depth_imbalance",
                "importance": 0.00028
            },
            {
                "feature": "spread",
                "importance": 8e-05
            },
            {
                "feature": "microprice_dev",
                "importance": 2e-05
            }
        ]
    }
}


# ══════════════════════════════════════════════════════════════════════════
# SIGNAL COMPUTER  (pure Python, state persisted via traderData each tick)
# ══════════════════════════════════════════════════════════════════════════

class SignalComputer:
    def __init__(self, window: int = 20):
        self.window       = window
        self.mid_history: List[float] = []
        self.obi_history: List[float] = []

    def update(self, order_depth: OrderDepth) -> dict:
        bids = order_depth.buy_orders
        asks = order_depth.sell_orders
        if not bids or not asks:
            return {}

        best_bid = max(bids.keys())
        best_ask = min(asks.keys())
        mid      = (best_bid + best_ask) / 2.0
        spread   = best_ask - best_bid
        bid_vol1 = bids[best_bid]
        ask_vol1 = abs(asks[best_ask])   # Prosperity stores sell_orders as negative

        self.mid_history.append(mid)
        if len(self.mid_history) > self.window * 2:
            self.mid_history.pop(0)

        obi = (bid_vol1 - ask_vol1) / (bid_vol1 + ask_vol1 + 1e-9)
        self.obi_history.append(obi)
        if len(self.obi_history) > self.window:
            self.obi_history.pop(0)

        microprice     = (best_bid * ask_vol1 + best_ask * bid_vol1) / (bid_vol1 + ask_vol1 + 1e-9)
        microprice_dev = microprice - mid

        sorted_bids = sorted(bids.keys(), reverse=True)[:3]
        sorted_asks = sorted(asks.keys())[:3]
        bid_depth   = sum(bids[p]      for p in sorted_bids)
        ask_depth   = sum(abs(asks[p]) for p in sorted_asks)
        depth_imb   = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-9)

        zscore = 0.0
        if len(self.mid_history) >= self.window:
            recent   = self.mid_history[-self.window:]
            mu       = sum(recent) / len(recent)
            variance = sum((x - mu) ** 2 for x in recent) / len(recent)
            sigma    = math.sqrt(variance) if variance > 0 else 1e-9
            zscore   = (mid - mu) / sigma

        ema8      = self._ema(self.mid_history, 8)
        ema21     = self._ema(self.mid_history, 21)
        ema_cross = ema8 - ema21

        momentum = 0.0
        if len(self.mid_history) >= 10:
            prev     = self.mid_history[-10]
            momentum = (mid - prev) / (prev + 1e-9)

        # Weighted OBI — explicit += avoids Python ternary precedence bugs
        bv  = bids.get(sorted_bids[0], 0) * 3 if len(sorted_bids) > 0 else 0
        bv += bids.get(sorted_bids[1], 0) * 2 if len(sorted_bids) > 1 else 0
        bv += bids.get(sorted_bids[2], 0) * 1 if len(sorted_bids) > 2 else 0
        av  = abs(asks.get(sorted_asks[0], 0)) * 3 if len(sorted_asks) > 0 else 0
        av += abs(asks.get(sorted_asks[1], 0)) * 2 if len(sorted_asks) > 1 else 0
        av += abs(asks.get(sorted_asks[2], 0)) * 1 if len(sorted_asks) > 2 else 0
        obi_weighted = (bv - av) / (bv + av + 1e-9)

        vol = self._rolling_vol(self.mid_history, self.window)

        return {
            # Names match the feature names used by the ML top_features list
            "mid":              mid,
            "best_bid":         best_bid,
            "best_ask":         best_ask,
            "spread":           spread,
            "spread_pct":       spread / (mid + 1e-9),
            "bid_volume_1":     bid_vol1,
            "ask_volume_1":     ask_vol1,
            "bid_volume_2":     bids.get(sorted_bids[1], 0) if len(sorted_bids) > 1 else 0,
            "ask_volume_2":     abs(asks.get(sorted_asks[1], 0)) if len(sorted_asks) > 1 else 0,
            "bid_price_1":      best_bid,
            "ask_price_1":      best_ask,
            "bid_price_2":      sorted_bids[1] if len(sorted_bids) > 1 else best_bid,
            "ask_price_2":      sorted_asks[1] if len(sorted_asks) > 1 else best_ask,
            "bid_gap_12":       (best_bid - sorted_bids[1]) if len(sorted_bids) > 1 else 0,
            "ask_gap_12":       (sorted_asks[1] - best_ask) if len(sorted_asks) > 1 else 0,
            "obi_l1":           obi,
            "obi_weighted":     obi_weighted,
            "obi_mean":         sum(self.obi_history) / len(self.obi_history),
            "obi_mean_5":       sum(self.obi_history[-5:])  / len(self.obi_history[-5:])  if self.obi_history else 0,
            "obi_mean_10":      sum(self.obi_history[-10:]) / len(self.obi_history[-10:]) if self.obi_history else 0,
            "microprice":       microprice,
            "microprice_dev":   microprice_dev,
            "total_bid_depth":  bid_depth,
            "total_ask_depth":  ask_depth,
            "depth_imbalance":  depth_imb,
            "zscore_20":        zscore,
            "ema_cross":        ema_cross,
            "momentum_10":      momentum,
            "volatility_5":     self._rolling_vol(self.mid_history, 5),
            "volatility_20":    vol,
            "mom_x_vol":        momentum * vol,
            "ofi_10":           0.0,   # populated from trade data — 0 at runtime (no trade feed)
            "ofi_20":           0.0,
        }

    @staticmethod
    def _ema(series: list, span: int) -> float:
        if not series:
            return 0.0
        alpha = 2.0 / (span + 1)
        val   = series[0]
        for x in series[1:]:
            val = alpha * x + (1 - alpha) * val
        return val

    @staticmethod
    def _rolling_vol(series: list, window: int) -> float:
        if len(series) < 2:
            return 0.0
        recent  = series[-window:]
        returns = [(recent[i] - recent[i-1]) / (recent[i-1] + 1e-9)
                   for i in range(1, len(recent))]
        if not returns:
            return 0.0
        mu  = sum(returns) / len(returns)
        var = sum((r - mu) ** 2 for r in returns) / len(returns)
        return math.sqrt(var)

    def to_dict(self) -> dict:
        return {"mid": self.mid_history, "obi": self.obi_history}

    def from_dict(self, d: dict) -> None:
        self.mid_history = d.get("mid", [])
        self.obi_history = d.get("obi", [])


# ══════════════════════════════════════════════════════════════════════════
# STRATEGY
# ══════════════════════════════════════════════════════════════════════════

def market_making_orders(symbol: str, signals: dict, position: int, p: dict) -> List[Order]:
    """
    Passive market-making using parameters directly from PARAMS[product].
    All p["key"] names correspond exactly to what the notebook writes.
    """
    # Fair value: stable products use the calibrated constant; drifting ones
    # blend microprice for a forward-looking estimate
    if p["fair_value_std"] < 2.0:
        fair_value = p["fair_value"]
    else:
        fair_value = 0.6 * signals["mid"] + 0.4 * signals["microprice"]

    offset     = p["quote_offset"]
    base_qty   = p["base_quantity"]
    max_pos    = p["max_position"]
    hard_limit = p["position_limit_hard"]
    bot_bid    = p["bot_bid"]
    bot_ask    = p["bot_ask"]
    obi        = signals.get("obi_l1", 0.0)
    best_bid   = signals.get("best_bid", bot_bid)
    best_ask   = signals.get("best_ask", bot_ask)
    inv_lean   = p["inventory_lean"]

    orders: List[Order] = []

    # ── Aggressive take: bot at exactly fair value (free edge) ─────────────
    if p["fair_value_std"] < 2.0:   # only for stable-price products
        fv = int(p["fair_value"])
        if best_bid == fv and position > -max_pos:
            qty = min(base_qty, max_pos + position)
            if qty > 0:
                orders.append(Order(symbol, fv, -qty))
                return orders
        if best_ask == fv and position < max_pos:
            qty = min(base_qty, max_pos - position)
            if qty > 0:
                orders.append(Order(symbol, fv, qty))
                return orders

    # ── Momentum adjustment for drifting products ──────────────────────────
    momentum  = signals.get("momentum_10", 0.0)
    mom_ticks = int(max(-2, min(2, momentum * 1e5)))
    fair_value += mom_ticks

    # ── Inventory unwind adjustment ────────────────────────────────────────
    pos_frac   = position / max(max_pos, 1)
    unwind_adj = -int(pos_frac * 3)
    fair_value += unwind_adj

    # ── OBI skew ──────────────────────────────────────────────────────────
    obi_skew = 0
    if abs(obi) > p["obi_skew_threshold"]:
        obi_skew = p["obi_skew_ticks"] * (1 if obi > 0 else -1)
    adjusted_fv = fair_value + obi_skew

    # ── Inventory-scaled quantities ────────────────────────────────────────
    bid_qty = max(1, int(base_qty * (1 - max(0,  pos_frac) * inv_lean * 2)))
    ask_qty = max(1, int(base_qty * (1 - max(0, -pos_frac) * inv_lean * 2)))
    if position >= hard_limit:
        bid_qty = 0
    if position <= -hard_limit:
        ask_qty = 0

    bid_price = int(adjusted_fv) - offset
    ask_price = int(adjusted_fv) + offset

    # ── Clamp inside the bot spread for queue priority ─────────────────────
    bid_price = min(bid_price, best_bid)        # don't cross the market
    ask_price = max(ask_price, best_ask)
    bid_price = max(bid_price, bot_bid + 1)     # beat the bot's bid
    ask_price = min(ask_price, bot_ask - 1)     # beat the bot's ask

    if bid_price >= ask_price:
        bid_price = int(adjusted_fv) - 1
        ask_price = int(adjusted_fv) + 1

    if bid_qty > 0:
        orders.append(Order(symbol, bid_price,  bid_qty))
    if ask_qty > 0:
        orders.append(Order(symbol, ask_price, -ask_qty))
    return orders


def ml_signal_vote(signals: dict, p: dict) -> Tuple[float, str]:
    """
    Weighted directional vote using top_features from PARAMS.
    Feature names in top_features must match keys in signals dict.
    """
    top_features  = p.get("top_features", [])
    strong_thresh = p.get("signal_threshold_strong", 0.7)
    base_thresh   = p.get("signal_threshold_base",   0.6)

    if not top_features:
        return 0.5, "flat"

    BULLISH = {"obi_l1", "obi_weighted", "obi_mean", "obi_mean_5", "obi_mean_10",
               "microprice_dev", "ema_cross", "momentum_10", "ofi_10", "ofi_20",
               "depth_imbalance", "bid_volume_1", "bid_volume_2"}
    BEARISH = {"zscore_20", "ask_volume_1", "ask_volume_2",
               "ask_price_1", "ask_price_2", "spread_pct"}

    total_w = weighted = 0.0
    for fi in top_features[:8]:
        name = fi["feature"]
        imp  = fi["importance"]
        val  = signals.get(name, 0.0)
        if name in BULLISH:
            vote = 1.0 if val > 0 else -1.0
        elif name in BEARISH:
            vote = -1.0 if val > 0 else 1.0
        else:
            vote = 0.0
        weighted += imp * vote
        total_w  += imp

    if total_w < 1e-9:
        return 0.5, "flat"

    conf = 0.5 + (weighted / total_w) * 0.5
    if conf >= strong_thresh:
        return conf, "long"
    if conf <= 1 - strong_thresh:
        return conf, "short"
    if conf >= base_thresh:
        return conf, "long"
    if conf <= 1 - base_thresh:
        return conf, "short"
    return 0.5, "flat"


def apply_position_limits(orders: List[Order], position: int, hard_limit: int) -> List[Order]:
    safe    = []
    running = position
    for o in orders:
        if abs(running + o.quantity) <= hard_limit:
            safe.append(o)
            running += o.quantity
    return safe


# ══════════════════════════════════════════════════════════════════════════
# TRADER CLASS
# ══════════════════════════════════════════════════════════════════════════

class Trader:
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        try:
            saved = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            saved = {}

        computers: Dict[str, SignalComputer] = {}
        for product in PARAMS:
            sc = SignalComputer(window=20)
            if product in saved:
                sc.from_dict(saved[product])
            computers[product] = sc

        result: Dict[str, List[Order]] = {}

        for product, p in PARAMS.items():
            if product not in state.order_depths:
                result[product] = []
                continue

            position = state.position.get(product, 0)
            signals  = computers[product].update(state.order_depths[product])

            if not signals:
                result[product] = []
                continue

            hard_limit = p["position_limit_hard"]
            orders     = market_making_orders(product, signals, position, p)

            # ML lean: tilt quotes 1 tick when confidence is high
            if p.get("use_ml_signal") and (p.get("ml_auc") or 0) > 0.55:
                _, ml_dir = ml_signal_vote(signals, p)
                if ml_dir != "flat":
                    lean   = 1 if ml_dir == "long" else -1
                    orders = [Order(o.symbol, o.price + lean, o.quantity) for o in orders]

            result[product] = apply_position_limits(orders, position, hard_limit)

        new_state   = {pr: computers[pr].to_dict() for pr in PARAMS}
        trader_data = json.dumps(new_state, separators=(",", ":"))
        return result, 0, trader_data
