"""Imisoro Compliance Coach environment.

One episode is one tax year (52 weekly steps) for one simulated Rwandan small
business that uses the Imisoro AI assistant. Each week the agent (the
assistant's engagement policy) picks one intervention. The owner's latent state
(tax knowledge, trust in the assistant, message fatigue, bookkeeping, cash set
aside, preparation for the next return) persists and changes with the actions,
and RRA filing deadlines arrive on a calendar that depends on the business's
tax profile. Rewards arrive mostly at deadlines, so the agent must trade off
contact now (cost, fatigue) against compliance later.

The obligation calendar is a simplified model of RRA guidance:
  * VAT (VAT-registered) and PAYE (has employees): monthly, due the 15th.
  * Flat / lump-sum regime: quarterly declaration, due the 15th after quarter end.
  * Real regime: annual income tax return (31 March) + quarterly prepayments
    (30 June, 30 September, 31 December).
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

WEEKS = 52

ACTIONS = [
    "WAIT",               # 0 no contact this week
    "SMS_REMINDER",       # 1 short deadline reminder by SMS (Kinyarwanda/English)
    "EMAIL_DIGEST",       # 2 detailed email with personalised tax calendar
    "RUN_CALCULATOR",     # 3 prompt owner to compute liability with the rule-based calculator
    "EDUCATION_MODULE",   # 4 short bilingual lesson grounded in RRA guidance (RAG answer + quiz)
    "RECORDS_NUDGE",      # 5 prompt to log sales / reconcile EBM receipts
    "SAVINGS_NUDGE",      # 6 prompt to set aside the estimated tax amount
    "ESCALATE_TO_RRA",    # 7 book an RRA call-centre / tax-office session (max 3 per year)
]
N_ACTIONS = len(ACTIONS)

# action -> (cost in reward units, fatigue added, channel)
ACTION_COST = np.array([0.0, 0.10, 0.05, 0.20, 0.30, 0.20, 0.10, 2.0])
ACTION_FATIGUE = np.array([0.0, 0.10, 0.05, 0.12, 0.15, 0.10, 0.08, 0.15])

OBLIGATIONS = ["VAT", "PAYE", "QUARTERLY", "ANNUAL"]
REGIMES = ["FLAT", "LUMP_SUM", "REAL"]


def _week_of(month: int, day: int) -> int:
    """Week index (0..51) of a calendar date in a 365-day year."""
    doy = int(round(month * 30.42 + day)) - 1
    return min(WEEKS - 1, doy // 7)


MONTHLY_WEEKS = [_week_of(m, 15) for m in range(12)]  # 15th of each month
QUARTERLY_WEEKS = [_week_of(m, 15) for m in (0, 3, 6, 9)]  # 15 Jan/Apr/Jul/Oct
ANNUAL_WEEKS = [_week_of(2, 31), _week_of(5, 30), _week_of(8, 30), _week_of(11, 31)]


@dataclass
class Scenario:
    """Distribution the owner profile and dynamics are sampled from."""

    knowledge: tuple[float, float] = (0.10, 0.50)
    trust: tuple[float, float] = (0.40, 0.80)
    records: tuple[float, float] = (0.10, 0.50)
    cash: tuple[float, float] = (0.00, 0.30)
    literacy: tuple[float, float] = (0.20, 0.90)
    p_employees: float = 0.40
    fatigue_mult: float = 1.0


SCENARIOS = {
    "train": Scenario(),
    # generalisation test: less knowledgeable, less trusting, more easily fatigued owners
    "stress": Scenario(knowledge=(0.0, 0.25), trust=(0.25, 0.55), records=(0.0, 0.30),
                       cash=(0.0, 0.15), literacy=(0.10, 0.60), p_employees=0.6,
                       fatigue_mult=1.4),
}


class ImisoroComplianceEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    OBS_NAMES = [
        "week_frac", "weeks_to_deadline", "n_due_next",
        "next_vat", "next_paye", "next_quarterly", "next_annual",
        "regime_flat", "regime_lump", "regime_real",
        "has_employees", "vat_registered", "digital_literacy",
        "prep_progress", "records_score", "cash_ratio",
        "engagement_rate", "contacts_4w", "quiz_score",
        "missed_frac", "escalations_left",
    ]

    def __init__(self, scenario: str = "train", render_mode: str | None = None,
                 max_escalations: int = 3):
        super().__init__()
        self.scenario_name = scenario
        self.sc = SCENARIOS[scenario]
        self.render_mode = render_mode
        self.max_escalations = max_escalations
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(len(self.OBS_NAMES),), dtype=np.float32)
        self._renderer = None

    # ------------------------------------------------------------------ setup
    def _sample_profile(self):
        r = self.np_random
        u = lambda lo_hi: float(r.uniform(*lo_hi))  # noqa: E731
        self.regime = int(r.choice(3, p=[0.35, 0.35, 0.30]))
        self.vat_registered = bool(self.regime == 2 and r.random() < 0.9)
        p_emp = min(1.0, self.sc.p_employees + (0.3 if self.regime == 2 else 0.0))
        self.has_employees = bool(r.random() < p_emp)
        self.literacy = u(self.sc.literacy)
        self.knowledge = u(self.sc.knowledge)
        self.trust = u(self.sc.trust)
        self.records = u(self.sc.records)
        self.cash = u(self.sc.cash)
        self.fatigue = 0.0

        cal: dict[int, list[int]] = {}
        add = lambda w, o: cal.setdefault(w, []).append(o)  # noqa: E731
        if self.vat_registered:
            for w in MONTHLY_WEEKS:
                add(w, 0)
        if self.has_employees:
            for w in MONTHLY_WEEKS:
                add(w, 1)
        if self.regime in (0, 1):
            for w in QUARTERLY_WEEKS:
                add(w, 2)
        else:
            for w in ANNUAL_WEEKS:
                add(w, 3)
        self.calendar = cal
        self.deadline_weeks = sorted(cal)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._sample_profile()
        self.week = 0
        self.prep = 0.0
        self.salience = 0.0
        self.engagement = 0.5
        self.contact_hist: list[int] = []
        self.missed = 0
        self.filed = 0
        self.underpaid = 0
        self.escalations_left = self.max_escalations
        self.opted_out = False
        self.last = {"action": None, "responded": None, "events": [], "reward": 0.0}
        self.ep_return = 0.0
        self.history: list[dict] = []
        if self.render_mode == "human":
            self.render()
        return self._obs(), self._info()

    # ------------------------------------------------------------ observation
    def _next_deadline(self):
        for w in self.deadline_weeks:
            if w >= self.week:
                return w, self.calendar[w]
        return None, []

    def _obs(self) -> np.ndarray:
        w, obl = self._next_deadline()
        wtd = (w - self.week) if w is not None else 8
        onehot = [1.0 if i in obl else 0.0 for i in range(4)]
        quiz = float(np.clip(self.knowledge + self.np_random.normal(0, 0.05), 0, 1))
        o = [
            self.week / WEEKS,
            min(wtd, 8) / 8,
            len(obl) / 3,
            *onehot,
            *[1.0 if self.regime == i else 0.0 for i in range(3)],
            float(self.has_employees), float(self.vat_registered), self.literacy,
            self.prep, self.records, min(self.cash, 1.0),
            self.engagement,
            sum(1 for a in self.contact_hist[-4:] if a != 0) / 4,
            quiz,
            min(self.missed, 5) / 5,
            self.escalations_left / self.max_escalations,
        ]
        return np.clip(np.asarray(o, dtype=np.float32), 0.0, 1.0)

    def _info(self):
        return {"week": self.week, "filed": self.filed, "missed": self.missed,
                "underpaid": self.underpaid, "opted_out": self.opted_out,
                "knowledge": self.knowledge, "trust": self.trust, "fatigue": self.fatigue,
                "compliance_rate": self.filed / max(1, self.filed + self.missed)}

    # ------------------------------------------------------------------- step
    def step(self, action):
        a = int(action)
        r = self.np_random
        reward = -float(ACTION_COST[a])
        events: list[str] = []
        w_next, _ = self._next_deadline()
        wtd = (w_next - self.week) if w_next is not None else 99

        # 1. owner response to the intervention
        responded = False
        if a != 0:
            affinity = self.literacy if a == 2 else (0.95 if a in (1, 7) else 0.8)
            p_resp = (0.3 + 0.7 * self.trust) * (1.0 - self.fatigue) * affinity
            responded = bool(r.random() < p_resp)
            self.engagement = 0.7 * self.engagement + 0.3 * float(responded)
            self.fatigue = min(1.0, self.fatigue + ACTION_FATIGUE[a] * self.sc.fatigue_mult)
            # reminders far from any deadline feel like spam
            if a in (1, 2) and wtd > 3:
                self.trust = max(0.0, self.trust - 0.05)

        if a == 7:
            if self.escalations_left == 0:
                reward -= 1.0
                responded = False
                events.append("escalation quota exhausted")
            else:
                self.escalations_left -= 1
                self.prep = min(1.0, self.prep + (0.6 if responded else 0.3))
                self.knowledge = min(1.0, self.knowledge + 0.05)
                events.append("RRA session booked")
        elif responded:
            if a == 1 and wtd <= 1:
                self.prep = min(1.0, self.prep + 0.08)
                self.salience = 1.0
            elif a == 2:
                self.prep = min(1.0, self.prep + (0.15 if wtd <= 3 else 0.03))
                self.salience = max(self.salience, 0.7 if wtd <= 3 else 0.2)
            elif a == 3:
                self.prep = min(1.0, self.prep + 0.50 * (0.5 + 0.5 * self.records))
            elif a == 4:
                self.knowledge = min(1.0, self.knowledge + 0.12 * (1 - self.knowledge))
            elif a == 5:
                self.records = min(1.0, self.records + 0.30 * (1 - self.records))
            elif a == 6:
                self.cash = min(1.5, self.cash + 0.35)

        # 2. natural (uncoached) dynamics, scaled by knowledge
        if wtd <= 3:
            self.prep = min(1.0, self.prep + 0.25 * self.knowledge)
        self.records = float(np.clip(self.records - 0.03 + 0.05 * self.knowledge, 0, 1))
        self.cash = min(1.5, self.cash + 0.04 * self.knowledge)
        self.fatigue *= 0.75
        self.salience *= 0.6 if wtd > 0 else 1.0

        # 3. deadline resolution
        due = self.calendar.get(self.week, [])
        for o in due:
            p_on_time = np.clip(0.10 + 0.45 * self.prep + 0.25 * self.knowledge
                                + 0.12 * self.salience + 0.15 * self.records, 0.0, 0.98)
            if r.random() < p_on_time:
                self.filed += 1
                if self.cash >= 0.6:
                    reward += 10.0
                    events.append(f"{OBLIGATIONS[o]} filed & paid on time (+10)")
                else:
                    reward += 4.0
                    self.underpaid += 1
                    events.append(f"{OBLIGATIONS[o]} filed, underpaid (+4)")
                self.trust = min(1.0, self.trust + 0.03)
            else:
                self.missed += 1
                reward -= 15.0
                self.trust = max(0.0, self.trust - 0.06)
                events.append(f"{OBLIGATIONS[o]} MISSED - RRA penalty (-15)")
        if due:
            self.cash = max(0.0, self.cash - 0.6)  # cash is a ratio of the next deadline's liability
            self.prep = 0.5 * self.knowledge
            self.salience = 0.0

        # 4. opt-out (terminal) when the owner is overloaded with messages
        terminated = False
        if self.fatigue > 0.45 and r.random() < (self.fatigue - 0.45) * 3.0:
            terminated = True
            self.opted_out = True
            reward -= 25.0
            events.append("owner OPTED OUT of Imisoro (-25)")

        self.contact_hist.append(a)
        self.week += 1
        truncated = self.week >= WEEKS and not terminated
        if truncated:
            reward += 5.0 * self.knowledge  # capability retained after the year
            events.append(f"year end: knowledge bonus +{5 * self.knowledge:.1f}")

        self.ep_return += reward
        self.last = {"action": ACTIONS[a], "responded": responded, "events": events,
                     "reward": reward}
        self.history.append({"week": self.week, "action": a, "reward": reward,
                             "events": events})
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, self._info()

    # ----------------------------------------------------------------- render
    def snapshot(self) -> dict:
        """Full state as JSON-friendly dict (used by the 3D renderer and the API)."""
        return {
            "week": self.week, "regime": REGIMES[self.regime],
            "vat_registered": self.vat_registered, "has_employees": self.has_employees,
            "calendar": {str(k): [OBLIGATIONS[o] for o in v] for k, v in self.calendar.items()},
            "knowledge": self.knowledge, "trust": self.trust, "fatigue": self.fatigue,
            "records": self.records, "cash": min(self.cash, 1.0), "prep": self.prep,
            "engagement": self.engagement, "filed": self.filed, "missed": self.missed,
            "underpaid": self.underpaid, "escalations_left": self.escalations_left,
            "opted_out": self.opted_out, "ep_return": self.ep_return, "last": self.last,
            "history": self.history[-60:], "scenario": self.scenario_name,
        }

    def ansi_line(self) -> str:
        l = self.last
        ev = "; ".join(l["events"]) if l["events"] else ""
        resp = "" if l["responded"] is None else (" (responded)" if l["responded"] else " (ignored)")
        return (f"wk {self.week:2d} | {str(l['action']):16s}{resp:12s} | r={l['reward']:+6.2f} "
                f"G={self.ep_return:+7.2f} | K={self.knowledge:.2f} T={self.trust:.2f} "
                f"F={self.fatigue:.2f} prep={self.prep:.2f} cash={min(self.cash, 1):.2f} | "
                f"filed={self.filed} missed={self.missed} | {ev}")

    def render(self):
        if self.render_mode == "ansi":
            return self.ansi_line()
        if self.render_mode == "human":
            if self._renderer is None:
                from environment.rendering import LiveRenderer
                self._renderer = LiveRenderer.get()
            self._renderer.push(self.snapshot())
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.flush()
