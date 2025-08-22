import os
import json
import math
import pytz
import time
import uuid
import copy
import enum
import argparse
import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Iterable
import psycopg2

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Pandas & numpy are optional but useful for reporting
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None

# -------------------------------------------------------------------------------------------------
# CONFIG / ENV
# -------------------------------------------------------------------------------------------------
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=5432
)
cur = conn.cursor()

# Timezone
EST = pytz.timezone("US/Eastern")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="[%(levelname)s] %(message)s")
logger = logging.getLogger("wnba_self_learning")


# -------------------------------------------------------------------------------------------------
# CONSTANTS / ENUMS
# -------------------------------------------------------------------------------------------------

class Result(str, enum.Enum):
    WIN = "WIN"
    LOSE = "LOSE"
    PUSH = "PUSH"


class BetSide(str, enum.Enum):
    OVER = "OVER"
    UNDER = "UNDER"


# -------------------------------------------------------------------------------------------------
# DATA CLASSES (aligned to your schemas)
# -------------------------------------------------------------------------------------------------

@dataclass
class BetHistoryRow:
    """
    Represents a single prior prediction/bet from the legacy models (CL1, CL2).
    We parse conservatively and normalize types. The 'result' field is critical for learning.
    """
    date: datetime
    player_name: str
    team: Optional[str]
    model_name: str
    predicted_pts: Optional[float]
    actual_pts: Optional[float]
    over_line: Optional[float]
    under_line: Optional[float]
    over_odds: Optional[float]
    under_odds: Optional[float]
    bet: Optional[str]
    result: Optional[str]
    pts_differential: Optional[float]
    note: Optional[str]
    amount: Optional[float]
    profit: Optional[float]

    @staticmethod
    def from_mapping(m: Dict[str, Any]) -> "BetHistoryRow":
        # Normalize/parse
        d = m.get("date")
        if isinstance(d, str):
            # Try ISO first (with optional Z)
            try:
                d = datetime.fromisoformat(d.replace("Z", "+00:00"))
            except Exception:
                try:
                    d = datetime.strptime(d, "%Y-%m-%d")
                except Exception:
                    d = None
        return BetHistoryRow(
            date=d,
            player_name=m.get("player_name"),
            team=m.get("team"),
            model_name=m.get("model_name"),
            predicted_pts=_to_float(m.get("predicted_pts")),
            actual_pts=_to_float(m.get("actual_pts")),
            over_line=_to_float(m.get("over_line")),
            under_line=_to_float(m.get("under_line")),
            over_odds=_to_float(m.get("over_odds")),
            under_odds=_to_float(m.get("under_odds")),
            bet=_norm_side(m.get("bet")),
            result=_norm_result(m.get("result")),
            pts_differential=_to_float(m.get("pts_differential")),
            note=(m.get("note") or None),
            amount=_to_float(m.get("amount")),
            profit=_to_float(m.get("profit")),
        )


@dataclass
class PlayerBoxRow:
    """
    Represents a single player box score row (from local CSVs in this read-only setup).
    """
    game_id: str
    season: int
    season_type: str
    game_date: str
    game_date_time: Optional[str]
    athlete_id: Optional[str]
    athlete_display_name: str
    team_id: Optional[str]
    team_name: Optional[str]
    team_location: Optional[str]
    team_short_display_name: Optional[str]
    minutes: Optional[float]
    field_goals_made: Optional[int]
    field_goals_attempted: Optional[int]
    three_point_field_goals_made: Optional[int]
    three_point_field_goals_attempted: Optional[int]
    free_throws_made: Optional[int]
    free_throws_attempted: Optional[int]
    offensive_rebounds: Optional[int]
    defensive_rebounds: Optional[int]
    rebounds: Optional[int]
    assists: Optional[int]
    steals: Optional[int]
    blocks: Optional[int]
    turnovers: Optional[int]
    fouls: Optional[int]
    plus_minus: Optional[int]
    points: Optional[int]
    starter: Optional[bool]
    ejected: Optional[bool]
    did_not_play: Optional[bool]
    reason: Optional[str]
    active: Optional[bool]
    athlete_jersey: Optional[str]
    athlete_short_name: Optional[str]
    athlete_headshot_href: Optional[str]
    athlete_position_name: Optional[str]
    athlete_position_abbreviation: Optional[str]
    team_display_name: Optional[str]
    team_uid: Optional[str]
    team_slug: Optional[str]
    team_logo: Optional[str]
    team_abbreviation: Optional[str]
    team_color: Optional[str]
    team_alternate_color: Optional[str]
    home_away: Optional[str]
    team_winner: Optional[bool]
    team_score: Optional[int]
    opponent_team_id: Optional[str]
    opponent_team_name: Optional[str]
    opponent_team_location: Optional[str]
    opponent_team_display_name: Optional[str]
    opponent_team_abbreviation: Optional[str]
    opponent_team_logo: Optional[str]
    opponent_team_color: Optional[str]
    opponent_team_alternate_color: Optional[str]
    opponent_team_score: Optional[int]

    @staticmethod
    def from_mapping(m: Dict[str, Any]) -> "PlayerBoxRow":
        def _int(x):
            try:
                if x is None or (isinstance(x, str) and x.strip() == ''):
                    return None
                if pd and pd.isna(x):
                    return None
                return int(x)
            except Exception:
                return None

        def _float(x):
            try:
                if x is None or (isinstance(x, str) and x.strip() == ''):
                    return None
                if pd and pd.isna(x):
                    return None
                return float(x)
            except Exception:
                return None

        return PlayerBoxRow(
            game_id=str(m.get("game_id")),
            season=_int(m.get("season")) or 0,
            season_type=(m.get("season_type") or ""),
            game_date=(m.get("game_date") or ""),
            game_date_time=(m.get("game_date_time") or None),
            athlete_id=(m.get("athlete_id") or None),
            athlete_display_name=(m.get("athlete_display_name") or ""),
            team_id=(m.get("team_id") or None),
            team_name=(m.get("team_name") or None),
            team_location=(m.get("team_location") or None),
            team_short_display_name=(m.get("team_short_display_name") or None),
            minutes=_float(m.get("minutes")),
            field_goals_made=_int(m.get("field_goals_made")),
            field_goals_attempted=_int(m.get("field_goals_attempted")),
            three_point_field_goals_made=_int(m.get("three_point_field_goals_made")),
            three_point_field_goals_attempted=_int(m.get("three_point_field_goals_attempted")),
            free_throws_made=_int(m.get("free_throws_made")),
            free_throws_attempted=_int(m.get("free_throws_attempted")),
            offensive_rebounds=_int(m.get("offensive_rebounds")),
            defensive_rebounds=_int(m.get("defensive_rebounds")),
            rebounds=_int(m.get("rebounds")),
            assists=_int(m.get("assists")),
            steals=_int(m.get("steals")),
            blocks=_int(m.get("blocks")),
            turnovers=_int(m.get("turnovers")),
            fouls=_int(m.get("fouls")),
            plus_minus=_int(m.get("plus_minus")),
            points=_int(m.get("points")),
            starter=_to_bool(m.get("starter")),
            ejected=_to_bool(m.get("ejected")),
            did_not_play=_to_bool(m.get("did_not_play")),
            reason=(m.get("reason") or None),
            active=_to_bool(m.get("active")),
            athlete_jersey=(m.get("athlete_jersey") or None),
            athlete_short_name=(m.get("athlete_short_name") or None),
            athlete_headshot_href=(m.get("athlete_headshot_href") or None),
            athlete_position_name=(m.get("athlete_position_name") or None),
            athlete_position_abbreviation=(m.get("athlete_position_abbreviation") or None),
            team_display_name=(m.get("team_display_name") or None),
            team_uid=(m.get("team_uid") or None),
            team_slug=(m.get("team_slug") or None),
            team_logo=(m.get("team_logo") or None),
            team_abbreviation=(m.get("team_abbreviation") or None),
            team_color=(m.get("team_color") or None),
            team_alternate_color=(m.get("team_alternate_color") or None),
            home_away=(m.get("home_away") or None),
            team_winner=_to_bool(m.get("team_winner")),
            team_score=_int(m.get("team_score")),
            opponent_team_id=(m.get("opponent_team_id") or None),
            opponent_team_name=(m.get("opponent_team_name") or None),
            opponent_team_location=(m.get("opponent_team_location") or None),
            opponent_team_display_name=(m.get("opponent_team_display_name") or None),
            opponent_team_abbreviation=(m.get("opponent_team_abbreviation") or None),
            opponent_team_logo=(m.get("opponent_team_logo") or None),
            opponent_team_color=(m.get("opponent_team_color") or None),
            opponent_team_alternate_color=(m.get("opponent_team_alternate_color") or None),
            opponent_team_score=_int(m.get("opponent_team_score")),
        )


@dataclass
class PlayerProp:
    name: str
    over_line: float
    under_line: float


@dataclass
class BetDecision:
    player: str
    predicted_points: float
    bet: str
    over_line: float
    under_line: float
    confidence: float
    rationale: str
    # optionally include these when available
    dip_probability: Optional[float] = None
    category: Optional[str] = None


# -------------------------------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------------------------------

def _to_float(x: Any) -> Optional[float]:
    """
    Best-effort float converter.
    """
    try:
        if x is None:
            return None
        if pd and pd.isna(x):
            return None
        if isinstance(x, (float, int)):
            if math.isnan(x):
                return None
            return float(x)
        s = str(x).strip()
        if not s:
            return None
        val = float(s)
        if math.isnan(val):
            return None
        return val
    except Exception:
        return None


def _to_bool(x: Any) -> Optional[bool]:
    """
    Best-effort bool converter.
    """
    if x is None:
        return None
    if pd and pd.isna(x):
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"true", "t", "1", "yes", "y"}:
        return True
    if s in {"false", "f", "0", "no", "n"}:
        return False
    return None


def _norm_side(x: Any) -> Optional[str]:
    """
    Normalize bet side to 'OVER'/'UNDER' or None.
    """
    if x is None:
        return None
    s = str(x).strip().upper()
    if s in {"OVER", "O"}:
        return "OVER"
    if s in {"UNDER", "U"}:
        return "UNDER"
    return None


def _norm_result(x: Any) -> Optional[str]:
    """
    Normalize result to 'WIN'/'LOSE'/'PUSH' or None. Accepts 'WON'/'LOST' etc.
    """
    if x is None:
        return None
    s = str(x).strip().upper()
    if s in {"WIN", "WON", "W"}:
        return Result.WIN.value
    if s in {"LOSE", "LOST", "L"}:
        return Result.LOSE.value
    if s in {"PUSH", "P"}:
        return Result.PUSH.value
    return None


# Console styling (simple ANSI)
class Style:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"


# -------------------------------------------------------------------------------------------------
# DB ACCESS (READ-ONLY)
# -------------------------------------------------------------------------------------------------

class DB:
    """
    Lightweight DB accessor. Only performing reads for prior model history.
    """

    def __init__(self, engine: Engine):
        self.engine = engine

    # Generic helper
    def _fetchall(self, sql: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        with self.engine.connect() as conn:
            rows = conn.execute(text(sql), params).mappings().all()
        return [dict(r) for r in rows]

    def fetch_player_history_for_models(
            self,
            player_name: str,
            models: Tuple[str, ...] = ("model_CL1", "model_CL2"),
            limit: int = 500,
    ) -> List[BetHistoryRow]:
        """
        Pull all the rows for a player from CL1/CL2. We keep limit reasonably high so we
        have enough history to learn meaningful patterns.
        """
        sql = (
            """
            SELECT date, player_name, team, model_name, predicted_pts, actual_pts,
                   over_line, under_line, over_odds, under_odds, bet, result,
                   pts_differential, note, amount, profit
            FROM predictions
            WHERE player_name = :player_name
              AND model_name = ANY(:models)
            ORDER BY date DESC
            LIMIT :limit
            """
        )
        rows = self._fetchall(sql, {"player_name": player_name, "models": list(models), "limit": limit})
        return [BetHistoryRow.from_mapping(r) for r in rows]

    def fetch_recent_playerboxes(self, player_name: str, limit: int = 20) -> List[PlayerBoxRow]:
        """
        Pull recent player boxes from local CSVs (paths are configured below).
        """
        self.years = {
            2025: "playerboxes/player_box_2025.csv",
            2024: "playerboxes/player_box_2024.csv",
            2023: "playerboxes/player_box_2023.csv",
            2022: "playerboxes/player_box_2022.csv",
            2021: "playerboxes/player_box_2021.csv",
            2020: "playerboxes/player_box_2020.csv",
            2019: "playerboxes/player_box_2019.csv",
            2018: "playerboxes/player_box_2018.csv",
            2017: "playerboxes/player_box_2017.csv",
            2016: "playerboxes/player_box_2016.csv",
            2015: "playerboxes/player_box_2015.csv",
            2014: "playerboxes/player_box_2014.csv",
            2013: "playerboxes/player_box_2013.csv",
            2012: "playerboxes/player_box_2012.csv",
            2011: "playerboxes/player_box_2011.csv",
            2010: "playerboxes/player_box_2010.csv",
            2009: "playerboxes/player_box_2009.csv"
        }

        results: List[PlayerBoxRow] = []

        # Newest to oldest, stop as soon as we hit the limit
        for year in sorted(self.years.keys(), reverse=True):
            path = self.years[year]
            try:
                if pd is None:
                    raise FileNotFoundError("pandas not installed or CSV unavailable")
                df = pd.read_csv(path)
            except FileNotFoundError:
                continue

            # Filter for the requested player
            player_df = df[df["athlete_display_name"] == player_name].copy()
            if player_df.empty:
                continue

            # Ensure date is sortable; drop any rows with bad/empty dates
            player_df["game_date"] = pd.to_datetime(player_df["game_date"], errors="coerce")
            player_df = player_df.dropna(subset=["game_date"]).sort_values("game_date", ascending=False)

            # Convert rows to PlayerBoxRow and collect until limit
            for _, row in player_df.iterrows():
                box_row = PlayerBoxRow.from_mapping(row.to_dict())
                # Skip if points is None or NaN
                if box_row.points is not None:
                    results.append(box_row)
                if len(results) >= limit:
                    return results

        return results


# -------------------------------------------------------------------------------------------------
# SELF-LEARNING ENGINE
# -------------------------------------------------------------------------------------------------

@dataclass
class PlayerState:
    """
    Per-player mutable state maintained only in-memory during a run. We allow
    bias and preference to keep nudging logistic probabilities, but HPO takes precedence.
    """
    bias: float = 0.0
    over_pref: float = 0.0
    under_pref: float = 0.0
    notes: str = ""


@dataclass
class HistorySummary:
    """
    Aggregated view of legacy model performance for a player.
    """
    o_correct: int = 0
    o_wrong: int = 0
    u_correct: int = 0
    u_wrong: int = 0
    total: int = 0
    side_counts: Dict[str, int] = field(default_factory=dict)
    side_correct_rate: Dict[str, float] = field(default_factory=dict)
    suggested_override: Optional[BetSide] = None
    override_reason: str = ""


class SelfLearner:
    """
    Maintains an **in-memory** per-player state for this run (no DB writes).
    The state shifts based on prior CL1/CL2 mistakes with **recency weighting**.
    Additionally, we implement a strong **Historical Preference Override** (HPO) that
    short-circuits the soft bias system when history is unequivocal.

    - Positive bias → pushes to OVER
    - Negative bias → pushes to UNDER
    - over_pref/under_pref capture directional momentum from historical wins/losses
    - HPO uses explicit bet+result counts to choose or flip sides.
    """

    def __init__(
            self,
            half_life_days: float = 60.0,
            min_history: int = 5,
            dominance: float = 0.70,
            strict_learn: bool = False,
    ):
        """
        Args:
          half_life_days: recency half-life for soft bias learning.
          min_history: minimum # of historical bets needed to trigger HPO.
          dominance: fraction threshold (e.g., 0.70) to decide 'overwhelming' correct/wrong.
          strict_learn: if True, HPO becomes even more assertive (slightly higher confidence).
        """
        self.half_life_days = half_life_days
        self._state: Dict[str, PlayerState] = {}
        self.min_history = max(1, int(min_history))
        self.dominance = float(max(0.5, min(0.95, dominance)))
        self.strict_learn = bool(strict_learn)

    # ---------------------------- Utilities ---------------------------- #
    @staticmethod
    def _sigmoid(x: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def _recency_weight(self, event_time: Optional[datetime], now: Optional[datetime] = None) -> float:
        if event_time is None:
            return 1.0
        if now is None:
            now = datetime.now(EST)
        days = (now - event_time).days if isinstance(event_time, datetime) else 0
        if days <= 0:
            return 1.0
        # half-life decay
        return 0.5 ** (days / max(1e-6, self.half_life_days))

    @staticmethod
    def _actual_side_from_points(row: BetHistoryRow) -> Optional[BetSide]:
        """
        Backup method: If 'result' is missing, infer OVER/UNDER winner by comparing
        actual_pts with over_line. We treat exact ties as pushes.
        """
        if row.actual_pts is None or row.over_line is None:
            return None
        if row.actual_pts > row.over_line + 1e-9:
            return BetSide.OVER
        if row.actual_pts < row.over_line - 1e-9:
            return BetSide.UNDER
        return None  # treat exact tie as None/PUSH

    # ---------------------------- Learning ----------------------------- #

    def _compute_accuracy(self, rows: List[BetHistoryRow]) -> Dict[str, Dict[str, int]]:
        """
        Compute correctness stats for OVER/UNDER using **result** first.
        Falls back to points vs line only when 'result' is absent.
        """
        stats = {"OVER": {"correct": 0, "wrong": 0}, "UNDER": {"correct": 0, "wrong": 0}}
        for r in rows:
            placed = (r.bet or "").upper()
            if placed not in ("OVER", "UNDER"):
                continue

            # Prefer 'result' if available
            if r.result in (Result.WIN.value, Result.LOSE.value, Result.PUSH.value):
                if r.result == Result.WIN.value:
                    stats[placed]["correct"] += 1
                elif r.result == Result.LOSE.value:
                    stats[placed]["wrong"] += 1
                # PUSH is ignored in accuracy tallies
                continue

            # Fall back to inferring from points
            actual_side = self._actual_side_from_points(r)
            if actual_side is None:
                continue
            if placed == actual_side.value:
                stats[placed]["correct"] += 1
            else:
                stats[placed]["wrong"] += 1

        return stats

    def history_summary(self, rows: List[BetHistoryRow]) -> HistorySummary:
        """
        Build a structured summary of historical performance. This feeds the
        Historical Preference Override (HPO).
        """
        st = self._compute_accuracy(rows)
        o_c, o_w = st["OVER"]["correct"], st["OVER"]["wrong"]
        u_c, u_w = st["UNDER"]["correct"], st["UNDER"]["wrong"]
        total = o_c + o_w + u_c + u_w

        # Compute side counts / correct rates
        side_counts: Dict[str, int] = {
            "OVER": o_c + o_w,
            "UNDER": u_c + u_w,
        }
        side_correct_rate: Dict[str, float] = {
            "OVER": (o_c / side_counts["OVER"]) if side_counts["OVER"] > 0 else 0.0,
            "UNDER": (u_c / side_counts["UNDER"]) if side_counts["UNDER"] > 0 else 0.0,
        }

        hs = HistorySummary(
            o_correct=o_c, o_wrong=o_w, u_correct=u_c, u_wrong=u_w,
            total=total, side_counts=side_counts, side_correct_rate=side_correct_rate
        )

        # Determine suggested override, if any
        hs.suggested_override, hs.override_reason = self._historical_preference_decision(hs)
        return hs

    def _historical_preference_decision(self, hs: HistorySummary) -> Tuple[Optional[BetSide], str]:
        """
        Decide a hard override based on dominance thresholds.
        If one side is >= dominance correct rate with enough samples → pick that side.
        If one side is <= (1 - dominance) correct rate with enough samples → flip to the other side.
        """
        # Require minimum total bet history (both sides combined)
        if hs.total < self.min_history:
            return None, f"insufficient history: total={hs.total} < min_history={self.min_history}"

        # Per-side sample sufficiency
        def enough(side: str) -> bool:
            return hs.side_counts.get(side, 0) >= max(3, int(self.min_history // 2))

        # OVER dominance
        if enough("OVER"):
            over_rate = hs.side_correct_rate["OVER"]
            if over_rate >= self.dominance:
                return BetSide.OVER, f"OVER dominance: rate={over_rate:.2f} ≥ dominance={self.dominance:.2f}"
            if over_rate <= (1.0 - self.dominance):
                return BetSide.UNDER, f"OVER failure: rate={over_rate:.2f} ≤ 1-dominance={1.0 - self.dominance:.2f}"

        # UNDER dominance
        if enough("UNDER"):
            under_rate = hs.side_correct_rate["UNDER"]
            if under_rate >= self.dominance:
                return BetSide.UNDER, f"UNDER dominance: rate={under_rate:.2f} ≥ dominance={self.dominance:.2f}"
            if under_rate <= (1.0 - self.dominance):
                return BetSide.OVER, f"UNDER failure: rate={under_rate:.2f} ≤ 1-dominance={1.0 - self.dominance:.2f}"

        return None, "no dominant or failing side"

    def learn_from(self, player: str, rows: List[BetHistoryRow]) -> PlayerState:
        """
        Soft learning (bias & preferences) with recency weighting.
        This still happens even when HPO triggers, so notes reflect stats.
        """
        st = self._state.get(player, PlayerState())
        if not rows:
            self._state[player] = st
            return st

        # Weighted error signal encourages *soft* biasing
        err = self._weighted_error_signal(rows)

        # bias update: clip and LR
        lr_bias = 0.35
        st.bias = float(max(-6.0, min(6.0, st.bias + lr_bias * max(-3.0, min(3.0, err)))))

        # prefs update: push more aggressively when the placed side was wrong
        lr_pref = 0.08
        now = datetime.now(EST)
        for r in rows:
            placed = (r.bet or "").upper()
            wt = self._recency_weight(r.date, now)

            # Prefer explicit result if available
            rnorm = r.result
            if rnorm in (Result.WIN.value, Result.LOSE.value):
                won = (rnorm == Result.WIN.value)
                if placed == "OVER":
                    if won:
                        st.over_pref += lr_pref * 0.5 * wt
                    else:
                        st.over_pref -= lr_pref * 1.0 * wt
                        st.under_pref += lr_pref * 2.0 * wt
                elif placed == "UNDER":
                    if won:
                        st.under_pref += lr_pref * 0.5 * wt
                    else:
                        st.under_pref -= lr_pref * 1.0 * wt
                        st.over_pref += lr_pref * 2.0 * wt
                continue

            # Fall back to points inference
            actual_side = self._actual_side_from_points(r)
            if placed == "UNDER" and actual_side == BetSide.OVER:
                st.over_pref += lr_pref * 2.0 * wt
                st.under_pref -= lr_pref * 1.0 * wt
            elif placed == "OVER" and actual_side == BetSide.UNDER:
                st.over_pref -= lr_pref * 1.0 * wt
                st.under_pref += lr_pref * 2.0 * wt
            elif placed == "OVER" and actual_side == BetSide.OVER:
                st.over_pref += lr_pref * 0.5 * wt
            elif placed == "UNDER" and actual_side == BetSide.UNDER:
                st.under_pref += lr_pref * 0.5 * wt

        # clamp
        st.over_pref = float(max(-5.0, min(5.0, st.over_pref)))
        st.under_pref = float(max(-5.0, min(5.0, st.under_pref)))

        # attach notes with raw stats for explainability
        stats = self._compute_accuracy(rows)
        err_str = f"{err:.2f}"
        st.notes = (
            f"CL1/CL2 OVER(correct={stats['OVER']['correct']},wrong={stats['OVER']['wrong']}), "
            f"UNDER(correct={stats['UNDER']['correct']},wrong={stats['UNDER']['wrong']}), "
            f"err={err_str} → bias={st.bias:.2f}, prefs(O={st.over_pref:.2f},U={st.under_pref:.2f})"
        )
        self._state[player] = st
        return st

    def _weighted_error_signal(self, rows: List[BetHistoryRow]) -> float:
        """
        Soft signal only: positive → historic tendency suggests more OVER.
        negative → historic tendency suggests more UNDER.
        Now grounded in explicit 'result' when possible.
        """
        now = datetime.now(EST)
        signal = 0.0
        for r in rows:
            placed = (r.bet or "").upper()
            if placed not in ("OVER", "UNDER"):
                continue
            wt = self._recency_weight(r.date, now)

            # Use 'result' first
            if r.result in (Result.WIN.value, Result.LOSE.value):
                if placed == "OVER":
                    signal += (0.25 if r.result == Result.WIN.value else -1.0) * wt
                else:  # UNDER
                    signal += (-0.25 if r.result == Result.WIN.value else 1.0) * wt
                continue

            # Fallback inference
            actual_side = self._actual_side_from_points(r)
            if actual_side is None:
                continue
            if placed == "UNDER" and actual_side == BetSide.OVER:
                signal += 1.0 * wt  # they were wrong UNDER → push OVER
            elif placed == "OVER" and actual_side == BetSide.UNDER:
                signal -= 1.0 * wt  # they were wrong OVER → push UNDER
            elif placed == "OVER" and actual_side == BetSide.OVER:
                signal += 0.25 * wt  # mild reinforcement toward OVER
            elif placed == "UNDER" and actual_side == BetSide.UNDER:
                signal -= 0.25 * wt  # mild reinforcement toward UNDER
        return signal

    # -------------------------- Calibration ---------------------------- #

    def calibrate_over_probability(self, raw: float, st: PlayerState) -> float:
        """
        Shift p(OVER) by logit adjustment derived from bias and (over_pref-under_pref).
        """
        raw = float(max(1e-6, min(1 - 1e-6, raw)))
        logit = math.log(raw / (1 - raw))
        adj = 0.8 * st.bias + 0.4 * (st.over_pref - st.under_pref)
        out = self._sigmoid(logit + adj)
        # small exploration to avoid lock-in
        eps = 0.02
        out = (1 - eps) * out + eps * 0.5
        return float(max(0.0, min(1.0, out)))

    # --------------------------- Hard Rules ---------------------------- #

    def force_flip(self, rows: List[BetHistoryRow]) -> Optional[BetSide]:
        """
        Safety valve using raw counts (softened now that HPO exists).
        Triggers if wrong side count is very high relative to correct.
        """
        stats = self._compute_accuracy(rows)
        uw, uc = stats["UNDER"]["wrong"], stats["UNDER"]["correct"]
        ow, oc = stats["OVER"]["wrong"], stats["OVER"]["correct"]

        # Make slightly less aggressive vs HPO; HPO should typically fire first.
        if uw >= 6 and uw >= 3 * max(1, uc):
            return BetSide.OVER
        if ow >= 6 and ow >= 3 * max(1, oc):
            return BetSide.UNDER
        return None


# -------------------------------------------------------------------------------------------------
# POINT PREDICTOR (from playerboxes; no CSVs, only DB)
# -------------------------------------------------------------------------------------------------

class PointPredictor:
    """Simple but robust point total predictor using recent playerbox games."""

    def __init__(self, lookback_games: int = 8):
        self.lookback_games = lookback_games

    @staticmethod
    def _recent(seq: List[Optional[int]], n: int) -> List[int]:
        out: List[int] = []
        for v in seq[:n]:
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                out.append(int(v))
        return out

    @staticmethod
    def _safe_mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    @staticmethod
    def _safe_stdev(xs: List[float]) -> float:
        if len(xs) < 2:
            return 0.0
        try:
            return float(statistics.stdev(xs))
        except Exception:
            return 0.0

    def _recency_weights(self, n: int) -> List[float]:
        # Heavier weight for more recent games
        if n <= 0:
            return []
        base = [i for i in range(n, 0, -1)]
        s = sum(base)
        return [b / s for b in base]

    def predict(self, boxes: List[PlayerBoxRow]) -> Optional[float]:
        if not boxes:
            return None
        # Filter out None values and NaN values
        pts = []
        for b in boxes:
            if b.points is not None and not (isinstance(b.points, float) and math.isnan(b.points)):
                pts.append(b.points)

        pts = pts[: max(1, self.lookback_games)]
        if not pts:
            return None
        n = len(pts)
        weights = self._recency_weights(n)
        w_avg = sum(p * w for p, w in zip(pts, weights))
        spread = self._safe_stdev([float(x) for x in pts])
        trend = 0.0
        if n >= 2:
            trend = float(pts[0] - pts[-1]) / max(1.0, n - 1)
        # volatility penalty to avoid overbetting noisy players
        pred = 0.65 * w_avg + 0.25 * (w_avg + trend) - 0.10 * spread
        return round(float(pred), 1)

    def get_performance_note(self, boxes: List[PlayerBoxRow]) -> str:
        """Generate a 3-word performance note based on recent games."""
        if not boxes:
            return "insufficient data"

        # Filter out None values and NaN values
        pts = []
        for b in boxes:
            if b.points is not None and not (isinstance(b.points, float) and math.isnan(b.points)):
                pts.append(b.points)

        pts = pts[: max(1, self.lookback_games)]
        if not pts:
            return "no valid data"

        avg_pts = self._safe_mean([float(x) for x in pts])
        stdev_pts = self._safe_stdev([float(x) for x in pts])

        # Categorize performance
        if avg_pts >= 20:
            if stdev_pts <= 3:
                return "Very Good Game"
            else:
                return "Good game"
        elif avg_pts >= 15:
            if stdev_pts <= 4:
                return "Above Average Game"
            else:
                return "Average Game"
        elif avg_pts >= 10:
            return "Below Average"
        elif avg_pts >= 5:
            return "Bad Game"
        else:
            return "Very Bad Game"


# -------------------------------------------------------------------------------------------------
# DECISION ENGINE
# -------------------------------------------------------------------------------------------------

@dataclass
class DecisionSettings:
    """
    Tunable thresholds for decision logic.
    """
    base_thresh: float = 0.52
    line_slope_min: float = 0.20
    line_slope_per_point: float = 0.015
    line_slope_cap_line: float = 40.0
    line_slope_floor_line: float = 10.0
    min_thresh: float = 0.48
    max_thresh: float = 0.58


class DecisionEngine:
    """
    Combine point prediction with learned bias and safety rules to choose OVER/UNDER.
    Now with a strong Historical Preference Override (HPO) that takes precedence.
    """

    def __init__(self, learner: SelfLearner, settings: Optional[DecisionSettings] = None):
        self.learner = learner
        self.settings = settings or DecisionSettings()

    @staticmethod
    def raw_over_probability(pred_points: float, over_line: float) -> float:
        """Logistic transform around the over_line."""
        line = over_line
        slope = 0.20 + 0.015 * max(10.0, min(40.0, line))
        margin = pred_points - line
        try:
            return 1.0 / (1.0 + math.exp(-slope * margin))
        except OverflowError:
            return 0.0 if margin < 0 else 1.0

    def _dynamic_threshold(self, over_line: float) -> float:
        s = self.settings
        adj = 0.01 * (over_line - 20.0)
        return float(max(s.min_thresh, min(s.max_thresh, s.base_thresh + adj)))

    def _get_bet_amount(self, confidence: float, decision_type: str) -> Optional[int]:
        """
        Convert confidence to bet amount (1-5 scale).
        Returns None if confidence is too low.

        Args:
            confidence: Confidence level (0.0-1.0)
            decision_type: The bet type ("OVER" or "UNDER") - not used in calculation but kept for consistency

        Returns:
            Integer from 1-5 representing bet amount, or None if not confident enough
        """
        if confidence < 0.52:  # Below this threshold, no bet
            return None

        # Map confidence (0.52-1.0) to bet amount (1-5)
        if confidence >= 0.85:
            return 5  # Very confident
        elif confidence >= 0.75:
            return 4  # High confidence
        elif confidence >= 0.65:
            return 3  # Moderate confidence
        elif confidence >= 0.58:
            return 2  # Low-moderate confidence
        else:
            return 1  # Low confidence (but still above minimum threshold)
    def decide(
            self,
            player: str,
            pred_points: float,
            over_line: float,
            under_line: float,
            prior_rows: List[BetHistoryRow],
    ) -> BetDecision:
        """
        Decision hierarchy:
          1) Historical Preference Override (HPO) based on explicit bet+result history
          2) Force-Flip safety valve on egregious error counts
          3) Logistic base probability + soft calibration (bias/prefs)
        """
        # --- 0) Build a history summary upfront for transparency
        hist = self.learner.history_summary(prior_rows)

        # --- 1) Historical Preference Override (hard)
        if hist.suggested_override is not None:
            # Keep learning notes up-to-date even when overriding.
            st = self.learner.learn_from(player, prior_rows)

            # Confidence from dominance strength + strict_learn
            dominance_strength = 0.0
            side = hist.suggested_override.value
            if side == "OVER" and hist.side_counts["OVER"] > 0:
                dominance_strength = abs(hist.side_correct_rate["OVER"] - 0.5) * 2.0  # 0..1
            elif side == "UNDER" and hist.side_counts["UNDER"] > 0:
                dominance_strength = abs(hist.side_correct_rate["UNDER"] - 0.5) * 2.0  # 0..1

            base_conf = 0.72 if not self.learner.strict_learn else 0.78
            conf = float(min(0.95, base_conf + 0.15 * dominance_strength))

            # Adjust predicted points to align with forced decision
            adjusted_pred_points = pred_points
            if side == "OVER" and pred_points <= over_line:
                # Force prediction above the line for OVER bet
                adjusted_pred_points = over_line + 1.5
            elif side == "UNDER" and pred_points >= over_line:
                # Force prediction below the line for UNDER bet
                adjusted_pred_points = over_line - 1.5

            rationale = (
                f"HPO: Forced {side} due to historical record. "
                f"O(correct={hist.o_correct},wrong={hist.o_wrong}), "
                f"U(correct={hist.u_correct},wrong={hist.u_wrong}); "
                f"reason={hist.override_reason}. "
                f"state(bias={st.bias:.2f}, prefO={st.over_pref:.2f}, prefU={st.under_pref:.2f}). "
                f"Adjusted prediction from {pred_points:.1f} to {adjusted_pred_points:.1f} to align with forced bet."
            )
            return BetDecision(
                player=player,
                predicted_points=float(adjusted_pred_points),  # This should use adjusted_pred_points
                bet=side,
                over_line=float(over_line),
                under_line=float(under_line),
                confidence=conf,
                rationale=rationale,
            )

        # --- 2) Force-Flip safety valve (if HPO didn't fire)
        forced = self.learner.force_flip(prior_rows)
        if forced is not None:
            st = self.learner.learn_from(player, prior_rows)
            rationale = (
                f"FORCED {forced.value} due to CL1/CL2 systematic errors "
                f"(no HPO). "
                f"O(correct={hist.o_correct},wrong={hist.o_wrong}), "
                f"U(correct={hist.u_correct},wrong={hist.u_wrong}). "
                f"state(bias={st.bias:.2f}, prefO={st.over_pref:.2f}, prefU={st.under_pref:.2f})."
            )
            confidence = 0.66
            return BetDecision(
                player=player,
                predicted_points=float(pred_points),
                bet=forced.value,
                over_line=float(over_line),
                under_line=float(under_line),
                confidence=float(max(0.0, min(1.0, confidence))),
                rationale=rationale,
            )

        # --- 3) Soft path: logistic + bias calibration
        p_raw = self.raw_over_probability(pred_points, over_line)
        st = self.learner.learn_from(player, prior_rows)
        p_cal = self.learner.calibrate_over_probability(p_raw, st)
        thresh = self._dynamic_threshold(over_line)

        if p_cal >= thresh:
            final_side = BetSide.OVER.value
        elif p_cal <= (1.0 - thresh):
            final_side = BetSide.UNDER.value
        else:
            # close-call tie-breaker: lean to sign of bias
            final_side = BetSide.OVER.value if st.bias >= 0 else BetSide.UNDER.value

        # confidence uses distance from 0.5 and how far from threshold
        base_confidence = abs(p_cal - 0.5) * 2  # Convert to 0-1 scale based on distance from 50/50
        # Apply dampening to avoid overconfidence
        if base_confidence >= 0.8:
            confidence = 0.65 + (base_confidence - 0.8) * 0.75  # Max out around 0.80
        elif base_confidence >= 0.6:
            confidence = 0.55 + (base_confidence - 0.6) * 0.5  # Moderate confidence
        else:
            confidence = 0.45 + base_confidence * 0.5  # Low confidence

        confidence = float(max(0.45, min(0.85, confidence)))

        rationale = (
            f"Soft path: p_raw={p_raw:.3f} → p_cal={p_cal:.3f} vs thresh={thresh:.3f}. "
            f"O(correct={hist.o_correct},wrong={hist.o_wrong}), "
            f"U(correct={hist.u_correct},wrong={hist.u_wrong}). "
            f"state(bias={st.bias:.2f}, prefO={st.over_pref:.2f}, prefU={st.under_pref:.2f})."
        )

        return BetDecision(
            player=player,
            predicted_points=float(pred_points),
            bet=final_side,
            over_line=float(over_line),
            under_line=float(under_line),
            confidence=float(max(0.0, min(1.0, confidence))),
            rationale=rationale,
        )


# -------------------------------------------------------------------------------------------------
# REPORTING / CONSOLE OUTPUT
# -------------------------------------------------------------------------------------------------

class Reporter:
    """
    Console reporter with diagnostic-heavy output so you can audit the choices.
    """

    def __init__(self, width: int = 96, show_history_block: bool = True):
        self.width = width
        self.show_history_block = show_history_block

    def banner(self, title: str):
        print(Style.MAGENTA + "=" * self.width + Style.RESET)
        print(Style.BOLD + title.center(self.width) + Style.RESET)
        print(Style.MAGENTA + "=" * self.width + Style.RESET)

    def line(self, ch: str = "-"):
        print(ch * self.width)

    def fmt_num(self, v: Optional[float], nd: int = 1) -> str:
        if v is None:
            return "-"
        return f"{v:.{nd}f}"

    def print_history_block(self, hist: HistorySummary):
        if not self.show_history_block:
            return
        print(Style.DIM + "  --- Historical Summary (CL1/CL2) ---" + Style.RESET)
        print(
            f"  OVER:  correct={hist.o_correct}, wrong={hist.o_wrong} "
            f"| UNDER: correct={hist.u_correct}, wrong={hist.u_wrong} "
            f"| total={hist.total}"
        )
        print(
            f"  Side counts: OVER={hist.side_counts.get('OVER', 0)}, "
            f"UNDER={hist.side_counts.get('UNDER', 0)}"
        )
        print(
            f"  Correct rates: OVER={hist.side_correct_rate.get('OVER', 0.0):.2f}, "
            f"UNDER={hist.side_correct_rate.get('UNDER', 0.0):.2f}"
        )
        if hist.suggested_override:
            print(
                Style.BOLD + f"  HPO Suggestion: {hist.suggested_override.value} ({hist.override_reason})" + Style.RESET)

    def print_player_summary(
            self,
            player: str,
            pred_points: Optional[float],
            over_line: Optional[float],
            under_line: Optional[float],
            decision: Optional[BetDecision],
            prior_stats: Dict[str, Dict[str, int]],
            hist: Optional[HistorySummary] = None,
    ):
        print(Style.CYAN + f"Player: {player}" + Style.RESET)
        print(
            f"  Predicted Points: {self.fmt_num(pred_points)}  |  Lines: O={self.fmt_num(over_line)}, U={self.fmt_num(under_line)}"
        )
        o = prior_stats.get("OVER", {"correct": 0, "wrong": 0})
        u = prior_stats.get("UNDER", {"correct": 0, "wrong": 0})
        print(
            f"  CL1/CL2 OVER: correct={o['correct']}, wrong={o['wrong']}  |  UNDER: correct={u['correct']}, wrong={u['wrong']}"
        )
        if hist is not None:
            self.print_history_block(hist)
        if decision is None:
            print(Style.YELLOW + "  Decision: None (insufficient data)" + Style.RESET)
        else:
            color = Style.GREEN if decision.bet == "OVER" else Style.RED
            print(color + f"  Decision: {decision.bet}  (conf={decision.confidence:.2f})" + Style.RESET)
            print("  Rationale: " + decision.rationale)
        self.line()


# -------------------------------------------------------------------------------------------------
# ORCHESTRATOR
# -------------------------------------------------------------------------------------------------

class Orchestrator:
    """
    High-level runner; glues together DB, learner, predictor, decider, and reporter.
    """

    def __init__(self, engine: Engine, min_history: int, dominance: float, strict_learn: bool):
        self.db = DB(engine)
        self.learner = SelfLearner(
            half_life_days=60.0,
            min_history=min_history,
            dominance=dominance,
            strict_learn=strict_learn,
        )
        self.point_model = PointPredictor(lookback_games=8)
        self.decider = DecisionEngine(self.learner)
        self.report = Reporter()

    def _prior_stats(self, rows: List[BetHistoryRow]) -> Dict[str, Dict[str, int]]:
        return self.learner._compute_accuracy(rows)

    def run_from_props(self, props_path: str):
        with open(props_path, "r") as f:
            data = json.load(f)
        players = data.get("players") or data.get("Players") or data
        if isinstance(players, dict) and "players" in players:
            players = players["players"]
        if not isinstance(players, list):
            raise ValueError("props.json must be a list or an object with 'players' list")

        self.report.banner("WNBA Self-Learning Model — Local Analysis")

        for p in players:
            try:
                name = p["name"]
                over_line = float(p.get("over_line"))
                under_line = float(p.get("under_line"))
            except Exception as e:
                logger.warning(f"Skipping malformed entry: {p} ({e})")
                continue

            # 1) pull prior CL1/CL2 bets
            prior_rows = self.db.fetch_player_history_for_models(name, ("model_CL1", "model_CL2"), limit=500)

            # 2) pull recent playerboxes for prediction
            boxes = self.db.fetch_recent_playerboxes(name, limit=20)
            pred_points = self.point_model.predict(boxes)

            # 3) compute decision
            decision: Optional[BetDecision] = None
            hist: Optional[HistorySummary] = self.learner.history_summary(prior_rows)
            if pred_points is not None:
                decision = self.decider.decide(
                    player=name,
                    pred_points=pred_points,
                    over_line=over_line,
                    under_line=under_line,
                    prior_rows=prior_rows,
                )

            # 4) print report
            stats = self._prior_stats(prior_rows)
            # Use the adjusted prediction from the decision if available
            display_pred_points = decision.predicted_points if decision else pred_points
            self.report.print_player_summary(
                player=name,
                pred_points=display_pred_points,  # <-- Use adjusted prediction
                over_line=over_line,
                under_line=under_line,
                decision=decision,
                prior_stats=stats,
                hist=hist,
            )


# -------------------------------------------------------------------------------------------------
# OPTIONAL: SIMULATION / BACKTEST (read-only, local printing)
# -------------------------------------------------------------------------------------------------

class Simulator:
    """
    Small helper for quick sanity checks against the most recent box score row.
    Not a true backtest; just helps validate directional decisions.
    """

    def __init__(self, orchestrator: Orchestrator):
        self.o = orchestrator

    @staticmethod
    def _decide_win(decision: BetDecision, actual_pts: float) -> Result:
        if decision.bet == BetSide.OVER.value:
            if actual_pts > decision.over_line:
                return Result.WIN
            if abs(actual_pts - decision.over_line) < 1e-9:
                return Result.PUSH
            return Result.LOSE
        else:
            if actual_pts < decision.over_line:
                return Result.WIN
            if abs(actual_pts - decision.over_line) < 1e-9:
                return Result.PUSH
            return Result.LOSE

    def simulate_recent_game(self, player: str, over_line: float, under_line: float) -> None:
        """
        Make a decision using current pipeline and score it on the **most recent** available
        playerbox row (as proxy). This is only for sanity checks; not a true backtest.
        """
        prior_rows = self.o.db.fetch_player_history_for_models(player, ("model_CL1", "model_CL2"), limit=500)
        boxes = self.o.db.fetch_recent_playerboxes(player, limit=1)
        if not boxes or boxes[0].points is None:
            print(f"No recent actual points available for {player}.")
            return
        pred_points = self.o.point_model.predict(self.o.db.fetch_recent_playerboxes(player, limit=20))
        if pred_points is None:
            print(f"No prediction available for {player}.")
            return
        decision = self.o.decider.decide(player, pred_points, over_line, under_line, prior_rows)
        actual = float(boxes[0].points)
        res = self._decide_win(decision, actual)
        print(
            f"SIM {player}: bet={decision.bet} conf={decision.confidence:.2f} | pred={pred_points:.1f} line={over_line:.1f} | actual={actual} → {res.value}"
        )


# -------------------------------------------------------------------------------------------------
# MAIN PREDICT FUNCTION
# -------------------------------------------------------------------------------------------------

# Replace the existing predict function (around line 820-900) with this fixed version:

# Replace the existing predict function (around line 820-900) with this fixed version:

# Replace the existing predict function (around line 820-900) with this fixed version:

# Replace the existing predict function (around line 820-900) with this fixed version:

# Replace the existing predict function (around line 820-900) with this fixed version:

# Replace the existing predict function (around line 820-900) with this fixed version:

def predict(player: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main prediction function that returns prediction results for a single player.
    Uses the original model prediction (not HPO-adjusted) to maintain consistency
    between the predicted points and the natural bet suggestion.

    Args:
        player: Dictionary containing player info with keys 'name', 'over_line', 'under_line'
    Returns:
        Dictionary with prediction results including predicted_points, bet, lines, note, and bet_amount
    """
    try:
        # Extract player info
        player_name = player['name']
        over_line = float(player['over_line'])
        under_line = float(player['under_line'])

        # Get current date for logging
        current_date = datetime.now().strftime("%Y-%m-%d")
        print(f"--- Running prediction for {player_name} on {current_date} ---")

        # Build engine and orchestrator
        try:
            engine = build_engine()
        except Exception as e:
            print(f"Database connection failed for {player_name}: {e}")
            print(f"Prediction not generated for {player_name}")
            return None

        orch = Orchestrator(
            engine=engine,
            min_history=5,
            dominance=0.70,
            strict_learn=False,
        )

        # Get player data
        prior_rows = orch.db.fetch_player_history_for_models(player_name, ("model_CL1", "model_CL2"), limit=500)
        boxes = orch.db.fetch_recent_playerboxes(player_name, limit=20)

        # Check if we have enough data
        if not boxes:
            print(f"{player_name} not in dip results")
            print(f"Prediction not generated for {player_name}")
            return None

        # Predict points - this is the original model prediction
        pred_points = orch.point_model.predict(boxes)
        if pred_points is None or math.isnan(pred_points):
            print(f"{player_name} prediction resulted in NaN")
            print(f"Prediction not generated for {player_name}")
            return None

        # Get decision which includes HPO-adjusted prediction
        decision = orch.decider.decide(
            player=player_name,
            pred_points=pred_points,
            over_line=over_line,
            under_line=under_line,
            prior_rows=prior_rows,
        )

        # Use the HPO-adjusted prediction for final output
        final_predicted_points = round(decision.predicted_points)

        # Determine bet based on adjusted prediction vs line (natural logic)
        if final_predicted_points > over_line:
            bet = "OVER"
        elif final_predicted_points < over_line:
            bet = "UNDER"
        else:
            # Exactly on the line - use original decision engine recommendation as tiebreaker
            bet = decision.bet

        # Get performance note based on recent games
        performance_note = orch.point_model.get_performance_note(boxes)

        # Calculate bet amount based on decision confidence
        print(f"DEBUG: Decision confidence: {decision.confidence:.3f}")

        # More conservative thresholds
        if decision.confidence >= 0.90:
            bet_amount = 5
        elif decision.confidence >= 0.83:
            bet_amount = 4
        elif decision.confidence >= 0.76:
            bet_amount = 3
        elif decision.confidence >= 0.69:
            bet_amount = 2
        elif decision.confidence >= 0.62:
            bet_amount = 1
        else:
            bet_amount = None

        print(f"DEBUG: Calculated bet amount: {bet_amount}")
        print(f"Prediction successful for {player_name}: {final_predicted_points} pts, bet: {bet}")

        return {
            "predicted_points": final_predicted_points,  # Uses original model prediction
            "bet": bet,  # Uses decision engine recommendation
            "over_line": over_line,
            "under_line": under_line,
            "note": performance_note,
            "amount": bet_amount
        }

    except Exception as e:
        print(f"Error processing {player.get('name', 'Unknown')}: {e}")
        print(f"Prediction not generated for {player.get('name', 'Unknown')}")
        return None
# -------------------------------------------------------------------------------------------------
# MAIN / CLI
# -------------------------------------------------------------------------------------------------

def build_engine() -> Engine:
    DATABASE_URL = os.getenv("DATABASE_URL")
    engine = create_engine(DATABASE_URL)
    # quick connectivity test
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        logger.error(f"DB connection failed: {e}")
        raise
    return engine


def parse_args() -> argparse.Namespace:
    """
    Centralized CLI parsing. Adds knobs for historical override strength.
    """
    parser = argparse.ArgumentParser(description="WNBA Self-Learning Model — local analysis (no inserts)")
    parser.add_argument("--props", type=str, default="props.json", help="Path to props.json")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["predict", "simulate"],
        default="predict",
        help="predict: iterate props and print decisions; simulate: quick check against last game",
    )
    parser.add_argument("--player", type=str, default=None, help="Player name for simulate mode")
    parser.add_argument("--over", type=float, default=None, help="Over line for simulate mode")
    parser.add_argument("--under", type=float, default=None, help="Under line for simulate mode")
    parser.add_argument("--width", type=int, default=96, help="Console width")

    # New: Historical Preference Override knobs
    parser.add_argument("--min-history", type=int, default=int(os.getenv("HPO_MIN_HISTORY", "5")),
                        help="Min number of historical bets required to enable historical override.")
    parser.add_argument("--dominance", type=float, default=float(os.getenv("HPO_DOMINANCE", "0.70")),
                        help="Dominance rate (0.5..0.95) for declaring a side overwhelmingly right/wrong.")
    parser.add_argument("--strict-learn", action="store_true",
                        help="If set, HPO decisions carry slightly higher confidence.")

    return parser.parse_args()


def main():
    args = parse_args()

    engine = build_engine()
    orch = Orchestrator(
        engine=engine,
        min_history=args.min_history,
        dominance=args.dominance,
        strict_learn=args.strict_learn,
    )
    orch.report.width = args.width

    if args.mode == "predict":
        orch.run_from_props(args.props)
    else:
        if not args.player or args.over is None or args.under is None:
            raise SystemExit("simulate mode requires --player --over --under")
        sim = Simulator(orch)
        sim.simulate_recent_game(args.player, args.over, args.under)


if __name__ == "__main__":
    main()
