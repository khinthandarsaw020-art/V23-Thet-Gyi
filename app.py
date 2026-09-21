"""
V23.1 LIVE TRADING BOT
=======================
V23.1 Strategy Engine + V20.6.2 Infrastructure
- Real-time API polling (6lottery)
- Telegram signals (24/7)
- Level betting (1-15)
- Flask health endpoints (HF Spaces compatible)
"""

from __future__ import annotations

import math
import time
import json
import sqlite3
import threading
import logging
import os
import requests
import statistics
from dataclasses import dataclass, asdict, field
from collections import deque, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Iterable

try:
    import numpy as np
except Exception:
    np = None

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

from flask import Flask


# ══════════════════════════════════════════════════════════
#  CREDENTIALS
# ══════════════════════════════════════════════════════════
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID        = os.environ.get("CHAT_ID", "")
LOTTERY_AUTH   = os.environ.get("LOTTERY_AUTH", "")

COLOUR_MAP = {
    0: "Violet+Red", 1: "Green", 2: "Red", 3: "Green", 4: "Red",
    5: "Violet+Green", 6: "Red", 7: "Green", 8: "Red", 9: "Green",
}

# ══════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════
CONFIG = {
    "api_url": "https://6lotteryapi.com/api/webapi/GetNoaverageEmerdList",
    "candle_max_size": 3000,
    "candle_db_path": "candles.db",
    "prediction_db_path": "predictions.db",
    "poll_interval": 2.0,
    "payout_rate": 0.96,
    "profit_reset_threshold": 100000,
    "min_quality_for_signal": "C",
    "min_data_before_signal": 30,
    "warmup_target": 30,
}


# ══════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════
def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(x)))


def sigmoid(x):
    x = max(-35.0, min(35.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def entropy_binary(p):
    p = clamp(p)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def outcome_to_int(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return None
        if value == 0:
            return 0
        if value == 1:
            return 1
        return None
    s = str(value).strip().lower()
    if s in {"big", "b", "1", "up", "true"}:
        return 1
    if s in {"small", "s", "0", "down", "false"}:
        return 0
    return None


def int_to_outcome(v):
    return "Big" if int(v) == 1 else "Small"


def normalize_prob(p, fallback=0.5):
    try:
        p = float(p)
        if math.isfinite(p):
            return clamp(p)
    except Exception:
        pass
    return fallback


# ══════════════════════════════════════════════════════════
#  CANDLE DB
# ══════════════════════════════════════════════════════════
class CandleDB:
    def __init__(self, db_path="candles.db", max_size=3000):
        self.db_path = db_path
        self.max_size = max_size
        self.lock = threading.Lock()
        self.candles = deque(maxlen=max_size)
        self.prev_close = 5.0
        self.last_period = None
        self.insert_count = 0
        self._init_db()
        self._load_recent()

    def _get_conn(self):
        return sqlite3.connect(self.db_path, timeout=10.0)

    def _init_db(self):
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS candles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        period TEXT UNIQUE, digit INTEGER,
                        open REAL, high REAL, low REAL, close REAL,
                        color TEXT, big_small TEXT, timestamp INTEGER
                    )
                """)
                conn.commit()
        except Exception as e:
            print(f"DB Init Error: {e}", flush=True)

    def _load_recent(self):
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT period, digit, open, high, low, close, color, big_small, timestamp
                    FROM candles ORDER BY id DESC LIMIT ?
                """, (self.max_size,))
                rows = cur.fetchall()
            for row in reversed(rows):
                self.candles.append({
                    "period": row[0], "digit": row[1], "open": row[2],
                    "high": row[3], "low": row[4], "close": row[5],
                    "color": row[6], "big_small": row[7], "timestamp": row[8],
                })
                self.prev_close = row[5]
                self.last_period = row[0]
            print(f"Loaded {len(self.candles)} candles", flush=True)
        except Exception as e:
            print(f"Load Error: {e}", flush=True)

    def add(self, digit, period=None):
        try:
            if period == self.last_period:
                return None
            open_p = self.prev_close
            close_p = float(digit)
            high_p = max(open_p, close_p) + 0.5
            low_p = min(open_p, close_p) - 0.5
            candle = {
                "period": str(period or int(time.time())),
                "digit": digit, "open": open_p, "high": high_p,
                "low": low_p, "close": close_p,
                "color": "green" if digit >= 5 else "red",
                "big_small": "Big" if digit >= 5 else "Small",
                "timestamp": int(time.time()),
            }
            with self.lock:
                self.candles.append(candle)
                self.prev_close = close_p
                self.last_period = candle["period"]
                try:
                    with self._get_conn() as conn:
                        conn.execute("""
                            INSERT OR REPLACE INTO candles
                            (period, digit, open, high, low, close, color, big_small, timestamp)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (candle["period"], candle["digit"], candle["open"],
                              candle["high"], candle["low"], candle["close"],
                              candle["color"], candle["big_small"], candle["timestamp"]))
                        conn.commit()
                except Exception:
                    pass
                self.insert_count += 1
            return candle
        except Exception:
            return None

    def get_candles(self):
        with self.lock:
            return list(self.candles)

    def get_count(self):
        with self.lock:
            return len(self.candles)


# ══════════════════════════════════════════════════════════
#  LEVEL BETTING
# ══════════════════════════════════════════════════════════
LEVEL_TABLE = {
    1:  {"bet1": 1000,   "bet2": 2000},
    2:  {"bet1": 1000,   "bet2": 2000},
    3:  {"bet1": 2000,   "bet2": 4000},
    4:  {"bet1": 2000,   "bet2": 4000},
    5:  {"bet1": 3000,   "bet2": 6000},
    6:  {"bet1": 4000,   "bet2": 8000},
    7:  {"bet1": 6000,   "bet2": 12000},
    8:  {"bet1": 8000,   "bet2": 16000},
    9:  {"bet1": 10000,  "bet2": 20000},
    10: {"bet1": 14000,  "bet2": 28000},
    11: {"bet1": 19000,  "bet2": 38000},
    12: {"bet1": 25000,  "bet2": 50000},
    13: {"bet1": 34000,  "bet2": 68000},
    14: {"bet1": 46000,  "bet2": 92000},
    15: {"bet1": 62000,  "bet2": 124000},
}


def get_level_bet(level):
    if level in LEVEL_TABLE:
        return LEVEL_TABLE[level]
    a = LEVEL_TABLE[14]["bet1"]
    b = LEVEL_TABLE[15]["bet1"]
    for _ in range(level - 15):
        a, b = b, a + b
    return {"bet1": b, "bet2": b * 2}


class BettingManager:
    def __init__(self):
        self.reset_all()

    def reset_all(self):
        self.level = 1
        self.level_state = "WAITING_BET1"
        self.bot_step = 1
        self.total_signals = 0
        self.total_wins = 0
        self.total_losses = 0
        self.recent_results = deque(maxlen=100)
        self.total_profit = 0.0
        self.total_loss_amount = 0.0
        self.current_profit = 0.0
        self.max_loss_amount = 0.0
        self.max_profit_seen = 0.0
        self.cycles_completed = 0
        self.max_level_reached = 1
        self.profit_resets = 0

    def get_current_bet(self):
        info = get_level_bet(self.level)
        if self.level_state == "WAITING_BET1":
            return info["bet1"], "BET1"
        return info["bet2"], "BET2"

    def get_wr(self):
        total = self.total_wins + self.total_losses
        return (self.total_wins / total * 100) if total > 0 else 0.0

    def on_result(self, won):
        old_level, old_state = self.level, self.level_state
        if self.level_state == "WAITING_BET1":
            if won:
                self.level_state = "WAITING_BET2"
                return "BET1_WIN", old_level, old_state
            else:
                self.level += 1
                self.level_state = "WAITING_BET1"
                self.max_level_reached = max(self.max_level_reached, self.level)
                return "BET1_LOSE", old_level, old_state
        else:
            if won:
                self.level = 1
                self.level_state = "WAITING_BET1"
                self.cycles_completed += 1
                return "RESET", old_level, old_state
            else:
                self.level += 1
                self.level_state = "WAITING_BET1"
                self.max_level_reached = max(self.max_level_reached, self.level)
                return "BET2_LOSE", old_level, old_state

    def update_bot_step(self, won):
        self.bot_step = 1 if won else self.bot_step + 1

    def apply_result(self, won):
        bet_amount, bet_type = self.get_current_bet()
        if won:
            profit = bet_amount * CONFIG["payout_rate"]
            self.total_profit += profit
            self.current_profit += profit
            self.total_wins += 1
        else:
            profit = -bet_amount
            self.total_loss_amount += bet_amount
            self.current_profit -= bet_amount
            self.total_losses += 1

        self.max_loss_amount = min(self.max_loss_amount, self.current_profit)
        self.max_profit_seen = max(self.max_profit_seen, self.current_profit)

        action, old_level, old_state = self.on_result(won)
        self.recent_results.append(1 if won else 0)
        self.update_bot_step(won)

        return {
            "bet_amount": bet_amount, "bet_type": bet_type,
            "profit": profit, "action": action,
            "old_level": old_level, "new_level": self.level,
        }

    def check_profit_reset(self):
        if self.current_profit >= CONFIG["profit_reset_threshold"]:
            report = {
                "net_profit": self.current_profit,
                "total_profit": self.total_profit,
                "total_loss": self.total_loss_amount,
                "max_dd": self.max_loss_amount,
                "max_level": self.max_level_reached,
            }
            self.total_profit = 0.0
            self.total_loss_amount = 0.0
            self.current_profit = 0.0
            self.max_loss_amount = 0.0
            self.max_profit_seen = 0.0
            self.level = 1
            self.level_state = "WAITING_BET1"
            self.bot_step = 1
            self.max_level_reached = 1
            self.cycles_completed = 0
            self.profit_resets += 1
            return report
        return None


# ══════════════════════════════════════════════════════════
#  V23.1 ENGINE
# ══════════════════════════════════════════════════════════
@dataclass
class AgentSignal:
    name: str
    p_big: float
    confidence: float
    group: str
    regime_fit: float = 1.0
    context_fit: float = 1.0


@dataclass
class Prediction:
    timestamp: float
    round_id: Optional[str]
    p_big_raw: float
    p_big_meta: float
    p_big_calibrated: float
    p_big_final: float
    signal: str
    confidence: float
    quality: str
    sequence_score: float
    loss_streak_risk: float
    win_gap: int
    context_key: str
    agents: Dict[str, float] = field(default_factory=dict)


class FeatureBuilder:
    def __init__(self, max_window=128):
        self.max_window = max_window

    def _vals(self, history):
        vals = []
        for x in history:
            v = outcome_to_int(x)
            if v is not None:
                vals.append(v)
        return vals[-self.max_window:]

    def features(self, history):
        v = self._vals(history)
        n = len(v)
        f = {
            "n": float(n), "p_big_5": 0.5, "p_big_10": 0.5,
            "p_big_20": 0.5, "p_big_50": 0.5, "run_len": 0.0,
            "run_dir": 0.0, "transition_up": 0.5, "transition_down": 0.5,
            "entropy_10": 1.0, "entropy_20": 1.0, "trend": 0.0, "cycle": 0.0,
        }
        if not v:
            return f

        for k in (5, 10, 20, 50):
            z = v[-min(k, n):]
            f[f"p_big_{k}"] = sum(z) / len(z)

        last = v[-1]
        run = 1
        for i in range(n - 2, -1, -1):
            if v[i] == last:
                run += 1
            else:
                break
        f["run_len"] = float(run)
        f["run_dir"] = 1.0 if last == 1 else -1.0

        if n >= 2:
            trans = [(v[i - 1], v[i]) for i in range(1, n)]
            up = [1 for a, b in trans if a == 0 and b == 1]
            dn = [1 for a, b in trans if a == 1 and b == 0]
            f["transition_up"] = (len(up) + 1) / (sum(1 for a, b in trans if a == 0) + 2)
            f["transition_down"] = (len(dn) + 1) / (sum(1 for a, b in trans if a == 1) + 2)

        for k in (10, 20):
            z = v[-min(k, n):]
            f[f"entropy_{k}"] = entropy_binary(sum(z) / len(z))

        if n >= 10:
            a = sum(v[-5:]) / 5
            b = sum(v[-10:-5]) / 5
            f["trend"] = a - b

        if n >= 8:
            z = v[-8:]
            same = sum(1 for i in range(1, len(z)) if z[i] == z[i - 1])
            f["cycle"] = same / 7.0

        return f


class BaseAgent:
    name = "base"
    group = "base"

    def predict(self, history, features):
        return AgentSignal(self.name, 0.5, 0.0, self.group)


class MultiMarkovAgent(BaseAgent):
    name = "multi_markov"
    group = "markov"

    def predict(self, history, features):
        if not history:
            return AgentSignal(self.name, 0.5, 0.0, self.group)

        def cond(order):
            if len(history) <= order:
                return 0.5
            key = tuple(history[-order:])
            a = b = 1.0
            for i in range(order, len(history)):
                if tuple(history[i-order:i]) == key:
                    if history[i] == 1:
                        a += 1
                    else:
                        b += 1
            return a / (a + b)

        probs = [cond(k) for k in (1, 2, 3)]
        p = 0.45 * probs[0] + 0.35 * probs[1] + 0.20 * probs[2]
        conf = abs(p - 0.5) * 2
        return AgentSignal(self.name, p, conf, self.group)


class PatternAgent(BaseAgent):
    name = "pattern"
    group = "pattern"

    def predict(self, history, features):
        if len(history) < 4:
            return AgentSignal(self.name, 0.5, 0.0, self.group)
        current = tuple(history[-3:])
        hits = []
        for i in range(3, len(history)):
            if tuple(history[i-3:i]) == current:
                hits.append(history[i])
        if not hits:
            return AgentSignal(self.name, 0.5, 0.0, self.group)
        p = (sum(hits) + 1) / (len(hits) + 2)
        conf = min(1.0, abs(p - 0.5) * 2 * math.sqrt(len(hits) / 8))
        return AgentSignal(self.name, p, conf, self.group)


class TrendAgent(BaseAgent):
    name = "trend"
    group = "trend"

    def predict(self, history, features):
        p5 = features["p_big_5"]
        p20 = features["p_big_20"]
        p = clamp(0.5 + 0.55 * (p5 - 0.5) + 0.20 * (p20 - 0.5))
        conf = abs(p - 0.5) * 2
        return AgentSignal(self.name, p, conf, self.group)


class MeanRevAgent(BaseAgent):
    name = "mean_rev"
    group = "mean_reversion"

    def predict(self, history, features):
        p20 = features["p_big_20"]
        p = clamp(0.5 - 0.45 * (p20 - 0.5))
        conf = abs(p - 0.5) * 2
        return AgentSignal(self.name, p, conf, self.group)


class TransitionAgent(BaseAgent):
    name = "transition"
    group = "transition"

    def predict(self, history, features):
        if not history:
            return AgentSignal(self.name, 0.5, 0.0, self.group)
        last = history[-1]
        p = features["transition_up"] if last == 0 else 1.0 - features["transition_down"]
        conf = abs(p - 0.5) * 2
        return AgentSignal(self.name, p, conf, self.group)


class DigitBiasAgent(BaseAgent):
    name = "digit_bias"
    group = "micro"

    def predict(self, history, features):
        if not history:
            return AgentSignal(self.name, 0.5, 0.0, self.group)
        p = clamp(0.5 + 0.15 * (features["p_big_5"] - 0.5))
        conf = abs(p - 0.5) * 2
        return AgentSignal(self.name, p, conf, self.group)


class FFTCycleAgent(BaseAgent):
    name = "fft_cycle"
    group = "cycle"

    def predict(self, history, features):
        if np is None or len(history) < 16:
            return AgentSignal(self.name, 0.5, 0.0, self.group)
        x = np.asarray(history[-64:], dtype=float)
        x = x - np.mean(x)
        if np.std(x) < 1e-9:
            return AgentSignal(self.name, 0.5, 0.0, self.group)
        spec = np.abs(np.fft.rfft(x))
        if len(spec) < 3:
            return AgentSignal(self.name, 0.5, 0.0, self.group)
        spec[0] = 0.0
        idx = int(np.argmax(spec[1:]) + 1)
        period = max(2, round(len(x) / idx))
        phase = len(x) % period
        p = clamp(0.5 + 0.12 * math.cos(2 * math.pi * phase / period))
        strength = float(spec[idx] / (np.sum(spec) + 1e-9))
        conf = clamp(strength * 3.0)
        return AgentSignal(self.name, p, conf, self.group)


class RegimeClassifier:
    def classify(self, features):
        p10 = features["p_big_10"]
        p20 = features["p_big_20"]
        trend = features["trend"]
        ent = features["entropy_20"]
        run = features["run_len"]

        if ent < 0.65 and run >= 4:
            return "clustered"
        if abs(trend) > 0.20:
            return "trending"
        if abs(p20 - 0.5) > 0.12 and ent < 0.90:
            return "biased"
        if ent > 0.97:
            return "noisy"
        return "balanced"


class ReliabilityBook:
    def __init__(self, max_contexts=200):
        self.agent_stats = defaultdict(lambda: [1.0, 1.0])
        self.context_stats = defaultdict(lambda: [1.0, 1.0])
        self.max_contexts = max_contexts

    def update(self, agent, correct, context):
        a, b = self.agent_stats[agent]
        self.agent_stats[agent] = [a + (1 if correct else 0), b + (0 if correct else 1)]

        a, b = self.context_stats[context]
        self.context_stats[context] = [a + (1 if correct else 0), b + (0 if correct else 1)]

        if len(self.context_stats) > self.max_contexts:
            self.context_stats.pop(next(iter(self.context_stats)))

    def agent_reliability(self, agent):
        a, b = self.agent_stats[agent]
        return a / (a + b)

    def context_reliability(self, context):
        a, b = self.context_stats[context]
        return a / (a + b)


class EvidenceFusion:
    def __init__(self):
        self.base_weights = {
            "markov": 1.25, "pattern": 1.10, "trend": 0.90,
            "mean_reversion": 0.80, "transition": 1.00,
            "cycle": 0.55, "micro": 0.35,
        }

    def fuse(self, signals, reliability, regime, context):
        groups = defaultdict(list)
        for s in signals:
            rel = reliability.agent_reliability(s.name)
            ctx_rel = reliability.context_reliability(context)
            regime_fit = self._regime_fit(s.group, regime)
            w = self.base_weights.get(s.group, 0.5) * (0.50 + 0.50 * rel)
            w *= (0.75 + 0.25 * ctx_rel)
            w *= (0.60 + 0.40 * regime_fit)
            w *= (0.35 + 0.65 * s.confidence)
            groups[s.group].append((s.p_big, w))

        group_probs = {}
        group_weights = {}
        for g, items in groups.items():
            den = sum(w for _, w in items)
            group_probs[g] = sum(p * w for p, w in items) / den if den else 0.5
            group_weights[g] = den

        if not group_probs:
            return 0.5, {}

        den = sum(group_weights.values())
        p = sum(group_probs[g] * group_weights[g] for g in group_probs) / den
        return clamp(p), group_probs

    @staticmethod
    def _regime_fit(group, regime):
        table = {
            "trending": {"trend": 1.0, "markov": 0.9, "pattern": 0.8, "mean_reversion": 0.4},
            "clustered": {"markov": 1.0, "pattern": 1.0, "transition": 0.9, "trend": 0.7},
            "noisy": {"cycle": 0.5, "trend": 0.5, "pattern": 0.6, "markov": 0.6},
            "biased": {"mean_reversion": 0.7, "trend": 0.9, "markov": 0.9},
            "balanced": {"markov": 0.8, "pattern": 0.8, "transition": 0.8},
        }
        return table.get(regime, {}).get(group, 0.75)


class SequenceStabilityEngine:
    def __init__(self, max_history=2000, context_order=4):
        self.results = deque(maxlen=max_history)
        self.context_order = context_order
        self.context_stats = defaultdict(lambda: [1.0, 1.0])
        self.total_predictions = 0
        self.total_wins = 0
        self.total_losses = 0

    def record(self, correct):
        previous_results = list(self.results)
        context = self._context_from(previous_results)
        y = 1 if correct else 0
        self.results.append(y)
        self.total_predictions += 1
        if y:
            self.total_wins += 1
        else:
            self.total_losses += 1
        a, b = self.context_stats[context]
        self.context_stats[context] = [a + y, b + (1 - y)]

    def current_loss_streak(self):
        r = list(self.results)
        n = 0
        for x in reversed(r):
            if x == 0:
                n += 1
            else:
                break
        return n

    def current_win_streak(self):
        r = list(self.results)
        n = 0
        for x in reversed(r):
            if x == 1:
                n += 1
            else:
                break
        return n

    def win_gap(self):
        r = list(self.results)
        for i in range(len(r) - 1, -1, -1):
            if r[i] == 1:
                return len(r) - 1 - i
        return len(r)

    def _context_from(self, r):
        if not r:
            return "START"
        return "".join("W" if x else "L" for x in r[-self.context_order:])

    def conditional_win_probability(self, extra_context=None):
        context = extra_context if extra_context is not None else self._context_from(list(self.results))
        a, b = self.context_stats[context]
        return a / (a + b)

    def transition_probability(self, prev):
        r = list(self.results)
        if prev is None:
            return (sum(r) + 1.0) / (len(r) + 2.0) if r else 0.5
        xs = []
        for i in range(1, len(r)):
            if r[i - 1] == prev:
                xs.append(r[i])
        return (sum(xs) + 1.0) / (len(xs) + 2.0) if xs else 0.5

    def context_probability(self, context):
        a, b = self.context_stats[context]
        return a / (a + b)

    def sequence_features(self):
        r = list(self.results)
        n = len(r)
        loss = self.current_loss_streak()
        win = self.current_win_streak()
        gap = self.win_gap()
        p_win = (sum(r) + 1) / (n + 2) if n else 0.5
        p_ww = self.transition_probability(1)
        p_w_after_l = self.transition_probability(0)
        ctx = self._context_from(r)
        p_ctx = self.context_probability(ctx)
        close_wins = 0
        if len(r) >= 2:
            for i, x in enumerate(r):
                if x == 1:
                    if i > 0 and r[i - 1] == 1:
                        close_wins += 1
                    elif i > 1 and r[i - 2] == 1:
                        close_wins += 1
        close_rate = close_wins / max(1, sum(r))
        return {
            "seq_n": float(n), "seq_p_win": p_win, "seq_p_ww": p_ww,
            "seq_p_w_after_l": p_w_after_l, "seq_p_context": p_ctx,
            "seq_loss_streak": float(loss), "seq_win_streak": float(win),
            "seq_win_gap": float(gap), "seq_close_win_rate": close_rate,
            "seq_entropy": entropy_binary(p_win),
        }

    def score(self):
        f = self.sequence_features()
        loss = f["seq_loss_streak"]
        gap = f["seq_win_gap"]
        pww = f["seq_p_ww"]
        pctx = f["seq_p_context"]
        loss_risk = clamp(1.0 - math.exp(-loss / 2.0))
        gap_risk = clamp((gap - 2) / 6.0) if gap > 2 else 0.0
        continuation = clamp(0.45 * pww + 0.35 * pctx + 0.20 * f["seq_p_win"])
        stability = clamp(0.45 * continuation + 0.30 * (1.0 - loss_risk) + 0.25 * (1.0 - gap_risk))
        return {
            "sequence_score": stability, "continuation_score": continuation,
            "loss_streak_risk": loss_risk, "gap_risk": gap_risk,
            "win_gap": int(gap), "p_ww": pww, "p_context_win": pctx,
            "context": self._context_from(list(self.results)),
        }

    def adjust_probability(self, p_model, base_confidence):
        p_model = clamp(p_model)
        conf = clamp(base_confidence)
        s = self.score()
        edge = p_model - 0.5
        shrink = 1.0 - 0.52 * s["loss_streak_risk"] - 0.25 * s["gap_risk"]
        continuation_bonus = (s["continuation_score"] - 0.5) * 0.18 * conf
        adjusted_edge = edge * max(0.15, shrink) + continuation_bonus * (1.0 if edge >= 0 else -1.0)
        p_final = clamp(0.5 + adjusted_edge)
        max_delta = 0.10 + 0.08 * conf
        p_final = clamp(p_model + clamp(p_final - p_model, -max_delta, max_delta))
        return p_final, s

    def recent_sequence(self, n=20):
        return "".join("W" if x else "L" for x in list(self.results)[-n:])


class MetaLearner:
    def __init__(self, min_train=120):
        self.min_train = min_train
        self.model = None

    def fit_walk_forward(self, X, y):
        if not SKLEARN_AVAILABLE or len(X) < self.min_train:
            self.model = None
            return
        self.model = HistGradientBoostingClassifier(
            max_iter=160, learning_rate=0.045, max_leaf_nodes=9,
            l2_regularization=1.5, random_state=42,
        )
        self.model.fit(X, y)

    def predict(self, x):
        if self.model is None:
            return 0.5
        try:
            return float(self.model.predict_proba([x])[0][1])
        except Exception:
            return 0.5


class ProbabilityCalibrator:
    def __init__(self):
        self.iso = None

    def fit(self, probs, y):
        if not SKLEARN_AVAILABLE or len(probs) < 80 or len(set(y)) < 2:
            self.iso = None
            return
        try:
            self.iso = IsotonicRegression(out_of_bounds="clip")
            self.iso.fit(probs, y)
        except Exception:
            self.iso = None

    def transform(self, p):
        if self.iso is None:
            return clamp(p)
        try:
            return clamp(float(self.iso.predict([p])[0]))
        except Exception:
            return clamp(p)


class ChangePointGuard:
    def __init__(self, window=30):
        self.window = window

    def score(self, history):
        if len(history) < self.window * 2:
            return 0.0
        a = sum(history[-self.window:]) / self.window
        b = sum(history[-2*self.window:-self.window]) / self.window
        return clamp(abs(a - b) * 2.5)


class BetProgressionSimulator:
    def __init__(self, max_level=10000):
        self.max_level = max(2, int(max_level))
        self.reset()

    def reset(self):
        self.level = 1
        self.stage = 1
        self.cycles_completed = 0
        self.total_wins = 0
        self.total_losses = 0
        self.peak_level = 1
        self.recovery_wins = 0
        self.failed_recoveries = 0
        self.history = []

    @property
    def state(self):
        return f"L{self.level}_B{self.stage}"

    def preview(self, correct):
        level = self.level
        stage = self.stage
        if correct:
            if stage == 1:
                return {"next_level": level, "next_stage": 2, "cycle_completed": False}
            return {"next_level": 1, "next_stage": 1, "cycle_completed": True}
        return {"next_level": min(self.max_level, level + 1), "next_stage": 1, "cycle_completed": False}

    def record(self, correct):
        correct = bool(correct)
        before = self.state
        before_level = self.level
        out = self.preview(correct)
        if correct:
            self.total_wins += 1
            if before_level > 1:
                self.recovery_wins += 1
            if out["cycle_completed"]:
                self.cycles_completed += 1
        else:
            self.total_losses += 1
            if before_level > 1 and self.stage == 2:
                self.failed_recoveries += 1
        self.level = int(out["next_level"])
        self.stage = int(out["next_stage"])
        self.peak_level = max(self.peak_level, self.level)
        self.history.append(1 if correct else 0)
        return {
            "before": before, "outcome": "W" if correct else "L",
            "after": self.state, "cycle_completed": bool(out["cycle_completed"]),
            "level": self.level, "stage": self.stage,
        }

    def risk(self):
        level_risk = clamp((self.level - 1) / 5.0)
        failed_recovery_risk = clamp(self.failed_recoveries / max(1, self.total_wins + self.total_losses))
        peak_risk = clamp((self.peak_level - 1) / 8.0)
        total = clamp(0.65 * level_risk + 0.20 * failed_recovery_risk + 0.15 * peak_risk)
        return {
            "progression_risk": total, "level_risk": level_risk,
            "failed_recovery_risk": failed_recovery_risk,
            "peak_level_risk": peak_risk,
        }

    def metrics(self):
        h = self.history
        n = len(h)
        dangerous = sum(1 for i in range(max(0, n - 6)) if h[i:i+7] == [1,0,0,0,0,0,1])
        recovery = sum(1 for i in range(max(0, n - 7)) if h[i:i+7] == [1,0,0,0,0,1,1])
        return {
            "n": n, "current_state": self.state,
            "current_level": self.level, "current_stage": self.stage,
            "cycles_completed": self.cycles_completed,
            "total_wins": self.total_wins, "total_losses": self.total_losses,
            "peak_level": self.peak_level, "recovery_wins": self.recovery_wins,
            "failed_recoveries": self.failed_recoveries,
            **self.risk(),
        }

    def sequence_score(self):
        return clamp(1.0 - self.risk()["progression_risk"])


class PredictionEngineV23:
    VERSION = "V23-Sequence-Stability-Progression"

    def __init__(self, max_history=2000, db_path=":memory:"):
        self.max_history = max_history
        self.history = deque(maxlen=max_history)
        self.predictions = deque(maxlen=max_history)

        self.features = FeatureBuilder()
        self.regime = RegimeClassifier()
        self.reliability = ReliabilityBook()
        self.fusion = EvidenceFusion()
        self.sequence = SequenceStabilityEngine(max_history=max_history)
        self.progression = BetProgressionSimulator()
        self.meta = MetaLearner()
        self.calibrator = ProbabilityCalibrator()
        self.change_guard = ChangePointGuard()

        self.agents = [
            MultiMarkovAgent(), PatternAgent(), TrendAgent(),
            MeanRevAgent(), TransitionAgent(), FFTCycleAgent(), DigitBiasAgent(),
        ]

        self.lock = threading.RLock()
        self.last_prediction = None

        self.db_path = db_path
        self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False) if db_path == ":memory:" else None
        self._init_db()

    def _connect(self):
        if self.db_path == ":memory:":
            return self._memory_conn
        return sqlite3.connect(self.db_path, check_same_thread=False, timeout=10.0)

    def _init_db(self):
        con = self._connect()
        con.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, round_id TEXT,
                p_raw REAL, p_meta REAL, p_cal REAL, p_final REAL,
                signal TEXT, confidence REAL, quality TEXT,
                sequence_score REAL, loss_streak_risk REAL,
                win_gap INTEGER, context_key TEXT,
                actual TEXT, correct INTEGER
            )
        """)
        con.commit()
        if self.db_path != ":memory:":
            con.close()

    def _save_prediction(self, pred):
        con = self._connect()
        con.execute("""
            INSERT INTO predictions (
                ts, round_id, p_raw, p_meta, p_cal, p_final, signal,
                confidence, quality, sequence_score, loss_streak_risk,
                win_gap, context_key, actual, correct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pred.timestamp, pred.round_id, pred.p_big_raw, pred.p_big_meta,
            pred.p_big_calibrated, pred.p_big_final, pred.signal,
            pred.confidence, pred.quality, pred.sequence_score,
            pred.loss_streak_risk, pred.win_gap, pred.context_key, None, None,
        ))
        con.commit()
        if self.db_path != ":memory:":
            con.close()

    def _history_features(self):
        return self.features.features(list(self.history))

    def _context_key(self, f, regime):
        seq = self.sequence.recent_sequence(4)
        return f"{regime}|{seq}|run{int(f['run_len'])}|pb{int(round(f['p_big_10']*10))}"

    def _all_features(self, f, seq, regime):
        cp = self.change_guard.score(list(self.history))
        return [
            f["p_big_5"], f["p_big_10"], f["p_big_20"], f["p_big_50"],
            f["run_len"], f["run_dir"], f["transition_up"], f["transition_down"],
            f["entropy_10"], f["entropy_20"], f["trend"], f["cycle"],
            seq["seq_p_win"], seq["seq_p_ww"], seq["seq_p_w_after_l"],
            seq["seq_p_context"], seq["seq_loss_streak"], seq["seq_win_streak"],
            seq["seq_win_gap"], seq["seq_close_win_rate"], seq["seq_entropy"],
            cp,
            {"balanced": 0, "trending": 1, "biased": 2, "clustered": 3, "noisy": 4}.get(regime, 0),
        ]

    def predict(self, round_id=None):
        with self.lock:
            f = self._history_features()
            regime = self.regime.classify(f)
            context = self._context_key(f, regime)
            seqf = self.sequence.sequence_features()

            signals = [a.predict(list(self.history), f) for a in self.agents]

            p_raw, _ = self.fusion.fuse(signals, self.reliability, regime, context)

            x = self._all_features(f, seqf, regime)
            p_meta = self.meta.predict(x)
            if p_meta == 0.5 and self.meta.model is None:
                p_meta = p_raw
            else:
                p_meta = clamp(0.55 * p_meta + 0.45 * p_raw)

            p_cal = self.calibrator.transform(p_meta)

            base_conf = abs(p_cal - 0.5) * 2
            p_final, seqscore = self.sequence.adjust_probability(p_cal, base_conf)

            prog = self.progression.risk()
            prog_shrink = 1.0 - 0.45 * prog["progression_risk"]
            p_final = clamp(0.5 + (p_final - 0.5) * max(0.30, prog_shrink))
            seqscore["progression_risk"] = prog["progression_risk"]
            seqscore["progression_score"] = 1.0 - prog["progression_risk"]
            seqscore["progression_level"] = float(self.progression.level)
            seqscore["progression_stage"] = float(self.progression.stage)

            cp = self.change_guard.score(list(self.history))
            confidence = clamp(base_conf * (1.0 - 0.35 * cp))

            risk = seqscore["loss_streak_risk"]
            progression_risk = seqscore.get("progression_risk", 0.0)
            edge = abs(p_final - 0.5)

            if edge >= 0.15 and confidence >= 0.58 and risk < 0.60 and progression_risk < 0.65:
                quality = "A"
            elif edge >= 0.09 and confidence >= 0.45 and risk < 0.78 and progression_risk < 0.82:
                quality = "B"
            elif edge >= 0.045 and confidence >= 0.25:
                quality = "C"
            else:
                quality = "D"

            signal = "Big" if p_final >= 0.5 else "Small"

            pred = Prediction(
                timestamp=time.time(), round_id=round_id,
                p_big_raw=p_raw, p_big_meta=p_meta, p_big_calibrated=p_cal,
                p_big_final=p_final, signal=signal, confidence=confidence,
                quality=quality, sequence_score=seqscore["sequence_score"],
                loss_streak_risk=seqscore["loss_streak_risk"],
                win_gap=seqscore["win_gap"], context_key=context,
                agents={s.name: s.p_big for s in signals},
            )

            self.last_prediction = pred
            self.predictions.append(pred)
            self._save_prediction(pred)
            return pred

    def resolve_last(self, actual):
        with self.lock:
            if not self.last_prediction:
                return None
            actual_i = outcome_to_int(actual)
            if actual_i is None:
                return None

            pred = self.last_prediction
            pred_i = 1 if pred.p_big_final >= 0.5 else 0
            correct = pred_i == actual_i

            self.sequence.record(correct)
            self.progression.record(correct)

            for name, p in pred.agents.items():
                agent_correct = (1 if p >= 0.5 else 0) == actual_i
                self.reliability.update(name, agent_correct, pred.context_key)

            self.history.append(actual_i)

            con = self._connect()
            if pred.round_id is None:
                row = con.execute("""
                    SELECT id FROM predictions WHERE actual IS NULL ORDER BY id DESC LIMIT 1
                """).fetchone()
            else:
                row = con.execute("""
                    SELECT id FROM predictions
                    WHERE round_id = ? AND actual IS NULL ORDER BY id DESC LIMIT 1
                """, (pred.round_id,)).fetchone()

            if row:
                con.execute(
                    "UPDATE predictions SET actual = ?, correct = ? WHERE id = ?",
                    (int_to_outcome(actual_i), int(correct), row[0])
                )
                con.commit()
            if self.db_path != ":memory:":
                con.close()

            return correct

    def sequence_metrics(self, outcomes=None):
        r = outcomes if outcomes is not None else list(self.sequence.results)
        if not r:
            return {
                "n": 0, "win_rate": 0.0, "max_loss_streak": 0,
                "avg_loss_streak": 0.0, "p95_loss_streak": 0.0,
                "max_win_gap": 0, "avg_win_gap": 0.0, "p95_win_gap": 0.0,
                "ww_pair_rate": 0.0, "www_rate": 0.0,
                "close_win_rate_gap_le_2": 0.0,
            }

        n = len(r)
        wins = sum(r)
        win_rate = wins / n

        loss_streaks = []
        cur = 0
        for x in r:
            if x == 0:
                cur += 1
            elif cur:
                loss_streaks.append(cur)
                cur = 0
        if cur:
            loss_streaks.append(cur)

        gaps = []
        last_w = None
        for i, x in enumerate(r):
            if x == 1:
                if last_w is not None:
                    gaps.append(i - last_w)
                last_w = i

        ww = sum(1 for i in range(1, n) if r[i-1] == 1 and r[i] == 1)
        www = sum(1 for i in range(2, n) if r[i-2] == 1 and r[i-1] == 1 and r[i] == 1)

        close_wins = sum(1 for g in gaps if g <= 2)

        def pct(vals, q):
            if not vals:
                return 0.0
            vals = sorted(vals)
            idx = min(len(vals) - 1, max(0, int(math.ceil(q * len(vals))) - 1))
            return float(vals[idx])

        w_origins = max(1, sum(1 for i in range(n - 1) if r[i] == 1))
        transition_ww_den = sum(1 for i in range(1, n) if r[i - 1] == 1)

        return {
            "n": n, "win_rate": win_rate,
            "max_loss_streak": max(loss_streaks) if loss_streaks else 0,
            "avg_loss_streak": statistics.mean(loss_streaks) if loss_streaks else 0.0,
            "p95_loss_streak": pct(loss_streaks, 0.95),
            "max_win_gap": max(gaps) if gaps else 0,
            "avg_win_gap": statistics.mean(gaps) if gaps else 0.0,
            "ww_pair_rate": ww / w_origins,
            "ww_after_w_rate": ww / max(1, transition_ww_den),
            "www_rate": www / max(1, ww),
            "close_win_rate_gap_le_2": close_wins / max(1, len(gaps)),
        }

    def status(self):
        with self.lock:
            sf = self.sequence.sequence_features()
            sc = self.sequence.score()
            f = self._history_features()
            return {
                "version": self.VERSION,
                "history_size": len(self.history),
                "prediction_count": len(self.predictions),
                "regime": self.regime.classify(f),
                "last_prediction": asdict(self.last_prediction) if self.last_prediction else None,
                "sequence": {**sf, **sc, "recent": self.sequence.recent_sequence(30)},
                "progression": self.progression.metrics(),
                "metrics": self.sequence_metrics(),
                "sklearn_available": SKLEARN_AVAILABLE,
            }


# ══════════════════════════════════════════════════════════
#  LIVE ORCHESTRATOR
# ══════════════════════════════════════════════════════════
class V23LiveBot:
    def __init__(self):
        self.lock = threading.Lock()
        self.engine = PredictionEngineV23(
            max_history=2000,
            db_path=CONFIG["prediction_db_path"]
        )
        self.candle_db = CandleDB(
            db_path=CONFIG["candle_db_path"],
            max_size=CONFIG["candle_max_size"],
        )
        self.betting = BettingManager()
        self.is_warmup = True
        self.last_digit = None
        self.current_regime = "unknown"
        self.active_prediction = None

    def send_telegram(self, message):
        if not TELEGRAM_TOKEN or not CHAT_ID:
            print(f"[TG-DISABLED] {message[:100]}", flush=True)
            return

        def _send():
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            for _ in range(3):
                try:
                    res = requests.post(url, json={
                        "chat_id": CHAT_ID,
                        "text": message,
                        "parse_mode": "HTML",
                    }, timeout=15)
                    if res.status_code == 200:
                        return
                except Exception:
                    time.sleep(2)

        threading.Thread(target=_send, daemon=True).start()

    def process_round(self, period, digit, result, is_warmup=False):
        with self.lock:
            notifications = []
            actual_i = 1 if result == "Big" else 0

            self.last_digit = digit
            self.candle_db.add(digit, period=period)

            if is_warmup:
                self.engine.history.append(actual_i)
                if len(self.engine.history) >= CONFIG["warmup_target"]:
                    self.is_warmup = False
                    notifications.append(
                        f"Warmup Complete\nHistory: {len(self.engine.history)}\nLive trading starting..."
                    )
                else:
                    notifications.append(
                        f"Warmup {len(self.engine.history)}/{CONFIG['warmup_target']}"
                    )
                return notifications

            if self.active_prediction is not None:
                correct = self.engine.resolve_last(result)
                won = bool(correct)
                settle = self.betting.apply_result(won)
                profit = settle["profit"]
                action = settle["action"]

                if won:
                    if action == "RESET":
                        notifications.append(
                            f"WIN (+{profit:,.0f})\n"
                            f"BET2 WIN -> RESET\n"
                            f"Level {settle['old_level']} -> 1\n"
                            f"Profit: {self.betting.current_profit:+,.0f}\n"
                            f"WR: {self.betting.get_wr():.1f}%"
                        )
                    else:
                        notifications.append(
                            f"WIN (+{profit:,.0f})\n"
                            f"BET1 WIN -> BET2 wait\n"
                            f"Level: {self.betting.level} | BET2\n"
                            f"Profit: {self.betting.current_profit:+,.0f}"
                        )
                else:
                    notifications.append(
                        f"LOSS ({profit:,.0f})\n"
                        f"Level {settle['old_level']} -> {settle['new_level']}\n"
                        f"Next Bet1: {get_level_bet(self.betting.level)['bet1']:,}\n"
                        f"Profit: {self.betting.current_profit:+,.0f}"
                    )

                reset = self.betting.check_profit_reset()
                if reset:
                    notifications.append(
                        f"PROFIT RESET!\n\n"
                        f"Net: +{reset['net_profit']:,.0f}\n"
                        f"Profit: +{reset['total_profit']:,.0f}\n"
                        f"Loss: -{reset['total_loss']:,.0f}\n"
                        f"Max DD: {reset['max_dd']:,.0f}\n"
                        f"Max Level: {reset['max_level']}\n"
                        f"Reset -> Level 1"
                    )

                self.active_prediction = None

            else:
                self.engine.history.append(actual_i)

            self.engine.predict(round_id=period)
            pred = self.engine.last_prediction
            self.current_regime = self.engine.regime.classify(
                self.engine._history_features()
            )

            quality = pred.quality
            min_q = CONFIG["min_quality_for_signal"]
            quality_order = {"A": 4, "B": 3, "C": 2, "D": 1}

            if quality_order.get(quality, 0) < quality_order.get(min_q, 2):
                notifications.append(
                    f"Period {period[-3:]}\n"
                    f"SKIP (Quality {quality})\n"
                    f"{pred.signal} @ {pred.confidence:.1%}"
                )
            else:
                bet_amount, bet_type = self.betting.get_current_bet()
                self.active_prediction = pred
                self.betting.total_signals += 1

                notifications.append(
                    f"Period {period[-3:]}\n"
                    f"SIGNAL -> {pred.signal.upper()}\n"
                    f"Conf: {pred.confidence:.1%}\n"
                    f"Quality: {quality}\n"
                    f"Regime: {self.current_regime}\n"
                    f"---\n"
                    f"Bot Step: {self.betting.bot_step}x\n"
                    f"Level: {self.betting.level} | {bet_type}\n"
                    f"Bet: {bet_amount:,}\n"
                    f"---\n"
                    f"Max Lv: {self.betting.max_level_reached}\n"
                    f"Max DD: {self.betting.max_loss_amount:,.0f}\n"
                    f"Profit: {self.betting.current_profit:+,.0f}\n"
                    f"WR: {self.betting.get_wr():.1f}%\n"
                    f"Last: {digit} ({COLOUR_MAP.get(digit, '?')})"
                )

            return notifications


# ══════════════════════════════════════════════════════════
#  API POLLER
# ══════════════════════════════════════════════════════════
def run_api_poller(bot):
    print("V23.1 Live Poller Starting...", flush=True)

    seen_periods = set()
    seen_order = deque(maxlen=1000)
    is_first_poll = True

    headers = {
        "accept": "application/json, text/plain, */*",
        "authorization": f"Bearer {LOTTERY_AUTH}",
        "content-type": "application/json;charset=UTF-8",
        "origin": "https://6win598.com",
        "referer": "https://6win598.com/",
        "user-agent": "Mozilla/5.0",
    }

    while True:
        try:
            payload = {
                "pageSize": 10, "pageNo": 1, "typeId": 30, "language": 7,
                "random": "036263f367384d418be07465793c8da8",
                "signature": "55F4FD150F15F090B943374F3C9BE78B",
                "timestamp": int(time.time()),
            }
            res = requests.post(
                CONFIG["api_url"], headers=headers, json=payload, timeout=5
            )
            if res.status_code != 200:
                time.sleep(2)
                continue

            data = res.json().get("data", {}).get("list", [])
            if not data:
                time.sleep(2)
                continue

            sorted_data = sorted(data, key=lambda x: int(x.get("issueNumber", 0)))

            if is_first_poll:
                print(f"First poll: {len(sorted_data)} periods", flush=True)
                for item in sorted_data:
                    period = str(item.get("issueNumber"))
                    num = int(item.get("number"))
                    result = "Big" if num >= 5 else "Small"
                    seen_periods.add(period)
                    seen_order.append(period)
                    bot.process_round(period, num, result, is_warmup=True)
                is_first_poll = False
            else:
                for item in sorted_data:
                    period = str(item.get("issueNumber"))
                    num = int(item.get("number"))
                    result = "Big" if num >= 5 else "Small"

                    if period not in seen_periods:
                        seen_periods.add(period)
                        seen_order.append(period)
                        if len(seen_periods) > 1000:
                            oldest = seen_order.popleft()
                            seen_periods.discard(oldest)

                        print(f"New {period} -> {result} ({num})", flush=True)
                        notifs = bot.process_round(period, num, result)
                        for msg in notifs:
                            bot.send_telegram(msg)
                            time.sleep(0.1)

        except Exception as e:
            print(f"Poll Error: {e}", flush=True)

        time.sleep(CONFIG["poll_interval"])


# ══════════════════════════════════════════════════════════
#  TELEGRAM COMMANDS
# ══════════════════════════════════════════════════════════
def poll_telegram(bot):
    if not TELEGRAM_TOKEN:
        print("TELEGRAM_TOKEN not set", flush=True)
        return

    try:
        requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook?drop_pending_updates=true",
            timeout=10
        )
    except Exception:
        pass

    offset = 0
    errors = 0

    while True:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates?offset={offset}&timeout=20"
            res = requests.get(url, timeout=25)
            if res.status_code == 200:
                errors = 0
                for upd in res.json().get("result", []):
                    offset = upd["update_id"] + 1
                    msg = upd.get("message", {})
                    chat = str(msg.get("chat", {}).get("id", ""))
                    text = msg.get("text", "").strip().lower()

                    if chat != CHAT_ID:
                        continue

                    with bot.lock:
                        b = bot.betting
                        if text == "/status":
                            bet, bt = b.get_current_bet()
                            bot.send_telegram(
                                f"V23.1 LIVE STATUS\n\n"
                                f"Regime: {bot.current_regime}\n"
                                f"Warmup: {'YES' if bot.is_warmup else 'NO'}\n"
                                f"Bot Step: {b.bot_step}x\n"
                                f"Level: {b.level} | {b.level_state}\n"
                                f"Bet: {bet:,} ({bt})\n\n"
                                f"Max Level: {b.max_level_reached}\n"
                                f"Max DD: {b.max_loss_amount:+,.0f}\n"
                                f"Profit: {b.current_profit:+,.0f}\n"
                                f"WR: {b.get_wr():.1f}%\n"
                                f"{b.total_wins}W / {b.total_losses}L\n"
                                f"Signals: {b.total_signals}\n"
                                f"Candles: {bot.candle_db.get_count()}\n"
                                f"History: {len(bot.engine.history)}"
                            )

                        elif text == "/v23":
                            eng = bot.engine
                            bot.send_telegram(
                                f"V23.1 ENGINE\n\n"
                                f"History: {len(eng.history)}\n"
                                f"Predictions: {len(eng.predictions)}\n"
                                f"Meta: {'YES' if eng.meta.model else 'NO'}\n"
                                f"Calibrator: {'YES' if eng.calibrator.iso else 'NO'}\n"
                                f"Progression: {eng.progression.state}\n"
                                f"Sklearn: {'YES' if SKLEARN_AVAILABLE else 'NO'}"
                            )

                        elif text == "/agent_stats":
                            rel = bot.engine.reliability
                            lines = ["Agent Reliability\n"]
                            for name, (a, b_) in rel.agent_stats.items():
                                wr = a / (a + b_) if (a + b_) > 0 else 0.5
                                lines.append(f"- {name}: {wr:.1%}")
                            bot.send_telegram("\n".join(lines))

                        elif text == "/metrics":
                            m = bot.engine.sequence_metrics()
                            bot.send_telegram(
                                f"METRICS\n\n"
                                f"n: {m['n']}\n"
                                f"WR: {m['win_rate']:.1%}\n"
                                f"Max LS: {m['max_loss_streak']}\n"
                                f"P95 LS: {m['p95_loss_streak']:.1f}"
                            )

                        elif text == "/profit":
                            bot.send_telegram(
                                f"PROFIT\n\n"
                                f"Net: {b.current_profit:+,.0f}\n"
                                f"Profit: +{b.total_profit:,.0f}\n"
                                f"Loss: -{b.total_loss_amount:,.0f}\n"
                                f"Max Level: {b.max_level_reached}"
                            )

                        elif text == "/reset":
                            b.reset_all()
                            bot.active_prediction = None
                            bot.send_telegram("Reset Done")

                        elif text == "/help":
                            bot.send_telegram(
                                "V23.1 COMMANDS\n\n"
                                "/status - Bot status\n"
                                "/v23 - Engine\n"
                                "/agent_stats - Agents\n"
                                "/metrics - Metrics\n"
                                "/profit - Profit\n"
                                "/reset - Reset\n"
                                "/help - Commands"
                            )

            else:
                errors += 1
                if errors > 5:
                    time.sleep(5)
                    errors = 0

        except Exception as e:
            print(f"TG Error: {e}", flush=True)
            errors += 1
            if errors > 5:
                time.sleep(5)
                errors = 0

        time.sleep(1)


# ══════════════════════════════════════════════════════════
#  FLASK
# ══════════════════════════════════════════════════════════
app = Flask(__name__)
GLOBAL_BOT = None


@app.route("/")
def home():
    if not GLOBAL_BOT:
        return "<h3>V23.1 starting...</h3>"
    b = GLOBAL_BOT
    with b.lock:
        bet, bt = b.betting.get_current_bet()
        return f"""
        <h2>V23.1 Live Bot</h2>
        <p>Warmup: {'YES' if b.is_warmup else 'NO'}</p>
        <p>Regime: {b.current_regime}</p>
        <p>Level: {b.betting.level} | {b.betting.level_state}</p>
        <p>Bet: {bet:,} ({bt})</p>
        <p>Profit: {b.betting.current_profit:+,.0f}</p>
        <p>WR: {b.betting.get_wr():.1f}%</p>
        <p>History: {len(b.engine.history)}</p>
        """


@app.route("/stats")
def stats():
    if not GLOBAL_BOT:
        return {"status": "initializing"}
    b = GLOBAL_BOT
    with b.lock:
        bet, bt = b.betting.get_current_bet()
        return {
            "version": "V23.1-LIVE",
            "warmup": b.is_warmup,
            "regime": b.current_regime,
            "betting": {
                "level": b.betting.level, "state": b.betting.level_state,
                "bet": bet, "bet_type": bt,
                "profit": round(b.betting.current_profit, 2),
                "wr": round(b.betting.get_wr(), 2),
                "wins": b.betting.total_wins, "losses": b.betting.total_losses,
            },
            "engine": {
                "history": len(b.engine.history),
                "predictions": len(b.engine.predictions),
                "meta_trained": b.engine.meta.model is not None,
            },
            "candles": b.candle_db.get_count(),
        }


@app.route("/health")
def health():
    return {"status": "ok", "version": "V23.1-LIVE"}


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
def main():
    global GLOBAL_BOT
    logging.basicConfig(level=logging.INFO)
    print("V23.1 LIVE BOT STARTING", flush=True)

    GLOBAL_BOT = V23LiveBot()

    threading.Thread(target=run_api_poller, args=(GLOBAL_BOT,), daemon=True).start()
    threading.Thread(target=poll_telegram, args=(GLOBAL_BOT,), daemon=True).start()

    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()