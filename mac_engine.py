from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    from kneed import KneeLocator
except ImportError:
    KneeLocator = None


SIMILARITY_FEATURES = [
    "RelSpeed",
    "InducedVertBreak",
    "HorzBreak",
    "SpinRate",
    "RelHeight",
    "RelSide",
    "ArmSlot",
]

CLUSTER_FEATURES = [
    "RelSpeed",
    "InducedVertBreak",
    "HorzBreak",
    "SpinRate",
    "SpinAxis",
]

ALLOWED_SOURCE_PITCH_TYPES = {
    "4-Seam Fastball": {"4-Seam Fastball", "Cutter"},
    "Sinker": {"Sinker", "2-Seam Fastball"},
    "Cutter": {"Cutter", "4-Seam Fastball"},
    "Slider": {"Slider", "Sweeper", "Slurve"},
    "Sweeper": {"Sweeper", "Slider", "Slurve"},
    "Curveball": {"Curveball", "Knuckle Curve"},
    "Knuckle Curve": {"Knuckle Curve", "Curveball"},
    "Changeup": {"Changeup", "Splitter", "Split-Finger", "Forkball"},
    "Splitter": {"Splitter", "Split-Finger", "Forkball", "Changeup"},
    "Split-Finger": {"Split-Finger", "Splitter", "Forkball", "Changeup"},
    "Forkball": {"Forkball", "Splitter", "Split-Finger", "Changeup"},
}

STATCAST_REQUIRED_COLUMNS = [
    "pitch_type",
    "pitch_name",
    "game_date",
    "release_speed",
    "release_pos_x",
    "release_pos_z",
    "player_name",
    "batter",
    "pitcher",
    "events",
    "description",
    "stand",
    "p_throws",
    "home_team",
    "away_team",
    "inning",
    "inning_topbot",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "launch_speed",
    "launch_angle",
    "release_spin_rate",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "spin_axis",
    "arm_angle",
]

RUN_VALUE_MAP = {
    "Out": -0.30,
    "FieldersChoice": -0.10,
    "Sacrifice": -0.05,
    "Error": 0.45,
    "Single": 0.90,
    "Double": 1.30,
    "Triple": 1.60,
    "HomeRun": 2.00,
}

WHIFF_CALLS = {
    "StrikeSwinging",
    "SwingingStrike",
    "SwingingStrikeBlocked",
}

SWING_CALLS = {
    "InPlay",
    "FoulBall",
    "FoulBallFieldable",
    "FoulBallNotFieldable",
    "StrikeSwinging",
    "SwingingStrike",
    "SwingingStrikeBlocked",
}

IN_PLAY_CALLS = {
    "InPlay",
    "FoulBall",
    "FoulBallFieldable",
    "FoulBallNotFieldable",
}

WOBA_MAP = {
    "Single": 0.90,
    "Double": 1.24,
    "Triple": 1.56,
    "HomeRun": 1.95,
    "Error": 0.70,
}


class MacAnalysis:
    def __init__(
        self,
        *,
        raw_data: pd.DataFrame,
        pitcher_clusters: pd.DataFrame,
        pitcher_cluster_points: pd.DataFrame,
        matchup_summary: pd.DataFrame,
        cluster_matchups: pd.DataFrame,
        similar_pitches: pd.DataFrame,
        distance_samples: pd.DataFrame,
        skipped_files: list[dict[str, str]],
    ) -> None:
        self.raw_data = raw_data
        self.pitcher_clusters = pitcher_clusters
        self.pitcher_cluster_points = pitcher_cluster_points
        self.matchup_summary = matchup_summary
        self.cluster_matchups = cluster_matchups
        self.similar_pitches = similar_pitches
        self.distance_samples = distance_samples
        self.skipped_files = skipped_files


def discover_data_files(data_root: str | Path) -> list[Path]:
    root = Path(data_root).expanduser()
    if not root.exists():
        return []
    if root.is_file() and root.suffix.lower() in {".csv", ".parquet"}:
        return [root]
    paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".parquet"}
    ]
    return sorted(paths)


def _safe_read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            header = pd.read_csv(
                path,
                nrows=0,
                encoding=encoding,
                engine="python",
                on_bad_lines="skip",
            )
            available = header.columns.tolist()
            if not available:
                continue
            usecols = [col for col in STATCAST_REQUIRED_COLUMNS if col in available]
            if not usecols:
                continue
            frame = pd.read_csv(
                path,
                usecols=usecols,
                encoding=encoding,
                engine="python",
                on_bad_lines="skip",
            )
            frame["source_file"] = path.name
            return frame
        except UnicodeDecodeError:
            continue
        except ValueError:
            continue
    raise ValueError(f"Could not read supported columns from {path}")


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    available = pq.read_schema(path).names
    frame = pd.read_parquet(path, columns=[col for col in STATCAST_REQUIRED_COLUMNS if col in available])
    frame["source_file"] = path.name
    return frame


def load_dataset(data_root: str | Path) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    files = discover_data_files(data_root)
    frames: list[pd.DataFrame] = []
    skipped: list[dict[str, str]] = []

    for path in files:
        try:
            if path.suffix.lower() == ".parquet":
                frames.append(_safe_read_parquet(path))
            else:
                frames.append(_safe_read_csv(path))
        except Exception as exc:  # noqa: BLE001
            skipped.append({"file": path.name, "reason": str(exc)})

    if not frames:
        return pd.DataFrame(), skipped

    df = pd.concat(frames, ignore_index=True)
    return prepare_dataset(df), skipped


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["Pitcher"] = data["player_name"].astype("string").str.strip()
    data["Batter"] = data["batter"].astype("Int64").astype("string").str.strip()
    data["PitcherThrows"] = data["p_throws"].astype("string").str.strip().str.title()
    data["BatterSide"] = data["stand"].astype("string").str.strip().str.title()
    data["PitcherTeam"] = np.where(
        data["inning_topbot"].astype("string").str.lower() == "top",
        data["home_team"],
        data["away_team"],
    )
    data["BatterTeam"] = np.where(
        data["inning_topbot"].astype("string").str.lower() == "top",
        data["away_team"],
        data["home_team"],
    )
    data["PAofInning"] = pd.to_numeric(data["at_bat_number"], errors="coerce")
    data["Inning"] = pd.to_numeric(data["inning"], errors="coerce")
    data["Top/Bottom"] = data["inning_topbot"].astype("string").str.title()
    data["GameID"] = data["game_pk"].astype("Int64").astype("string")
    data["RelSpeed"] = pd.to_numeric(data["release_speed"], errors="coerce")
    data["InducedVertBreak"] = pd.to_numeric(data["pfx_z"], errors="coerce") * 12.0
    data["HorzBreak"] = pd.to_numeric(data["pfx_x"], errors="coerce") * 12.0
    data["SpinRate"] = pd.to_numeric(data["release_spin_rate"], errors="coerce")
    data["RelHeight"] = pd.to_numeric(data["release_pos_z"], errors="coerce")
    data["RelSide"] = pd.to_numeric(data["release_pos_x"], errors="coerce")
    data["SpinAxis"] = pd.to_numeric(data["spin_axis"], errors="coerce")
    data["PlateLocSide"] = pd.to_numeric(data["plate_x"], errors="coerce")
    data["PlateLocHeight"] = pd.to_numeric(data["plate_z"], errors="coerce")
    data["ExitSpeed"] = pd.to_numeric(data["launch_speed"], errors="coerce")
    data["Angle"] = pd.to_numeric(data["launch_angle"], errors="coerce")
    data["Date"] = data["game_date"].astype("string")
    data["TaggedPitchType"] = data["pitch_name"].astype("string").str.strip()
    data["AutoPitchType"] = data["pitch_type"].astype("string").str.strip()
    data["PitchCall"] = data["description"].map(_map_statcast_pitch_call).fillna("Other")
    data["PlayResult"] = data["events"].map(_map_statcast_play_result).fillna("Other")

    numeric_columns = [
        "RelSpeed",
        "InducedVertBreak",
        "HorzBreak",
        "SpinRate",
        "RelHeight",
        "RelSide",
        "SpinAxis",
        "PlateLocSide",
        "PlateLocHeight",
        "ExitSpeed",
        "Angle",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    x = data["RelSide"].abs().astype(float)
    z = data["RelHeight"].astype(float)
    data["ArmSlot"] = (-77.375769 - 31.621519 * x + 35.693581 * z + 3.579529 * x * z + 1.713441 * x**2 - 2.300103 * z**2).clip(lower=0.0, upper=90.0)

    data["pitch_type"] = (
        data["TaggedPitchType"]
        .where(data["TaggedPitchType"].notna() & (data["TaggedPitchType"] != "Undefined"))
        .fillna(data["AutoPitchType"])
        .fillna("Unknown")
        .astype("string")
        .str.strip()
    )
    data["pitch_group"] = data["pitch_type"].map(_collapse_pitch_group).fillna("Other")

    data["is_whiff"] = data["PitchCall"].isin(WHIFF_CALLS).astype(float)
    data["is_swing"] = data["PitchCall"].isin(SWING_CALLS).astype(float)
    data["is_in_play"] = data["PitchCall"].isin(IN_PLAY_CALLS).astype(float)
    data["hard_hit"] = (data["ExitSpeed"] >= 95).astype(float)
    data["is_hit"] = data["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"]).astype(float)
    data["estimated_woba"] = data["PlayResult"].map(WOBA_MAP).fillna(0.0)
    data["run_value"] = data["PlayResult"].map(RUN_VALUE_MAP).fillna(0.0)

    pa_parts = [
        data["GameID"].astype("string").fillna(""),
        data["Top/Bottom"].astype("string").fillna(""),
        data["Inning"].astype("Int64").astype("string").fillna(""),
        data["PAofInning"].astype("Int64").astype("string").fillna(""),
        data["Batter"].astype("string").fillna(""),
    ]
    data["pa_key"] = pa_parts[0]
    for part in pa_parts[1:]:
        data["pa_key"] = data["pa_key"] + "|" + part

    data = data.dropna(subset=["Pitcher", "Batter"])
    data = data.dropna(subset=["RelSpeed", "InducedVertBreak", "HorzBreak"])
    data = data[data["RelSpeed"] >= 60].reset_index(drop=True)

    return data


def run_mac(
    data: pd.DataFrame,
    pitcher_name: str,
    hitters: Iterable[str],
    *,
    similarity_threshold: float = 0.6,
    min_similar_pitches: int = 0,
    max_clusters: int = 8,
) -> MacAnalysis:
    hitters = [h for h in hitters if h]
    if data.empty:
        raise ValueError("No usable data was loaded.")
    if not pitcher_name:
        raise ValueError("A pitcher must be selected.")
    if not hitters:
        raise ValueError("Select at least one hitter.")

    pitcher_df = data[data["Pitcher"] == pitcher_name].copy()
    if pitcher_df.empty:
        raise ValueError(f"No pitches found for pitcher '{pitcher_name}'.")

    pitcher_hand = (
        pitcher_df["PitcherThrows"].dropna().iloc[0]
        if pitcher_df["PitcherThrows"].notna().any()
        else None
    )
    if pitcher_hand is not None:
        data = data[data["PitcherThrows"] == pitcher_hand].copy()
        pitcher_df = data[data["Pitcher"] == pitcher_name].copy()

    data = data.dropna(subset=SIMILARITY_FEATURES + ["Pitcher", "Batter"]).copy()
    pitcher_df = pitcher_df.dropna(subset=SIMILARITY_FEATURES + ["Pitcher", "Batter"]).copy()
    if pitcher_df.empty:
        raise ValueError("The selected pitcher does not have enough pitch data.")

    scaler = build_feature_scaler(data, SIMILARITY_FEATURES)
    pitcher_clusters, pitcher_cluster_points, cluster_model, cluster_scaler, cluster_to_type = build_pitcher_clusters(
        pitcher_df,
        max_clusters=max_clusters,
    )

    reference_values = pitcher_df[SIMILARITY_FEATURES].div(scaler[SIMILARITY_FEATURES], axis=1).to_numpy(dtype=float)
    hitter_values = data[SIMILARITY_FEATURES].div(scaler[SIMILARITY_FEATURES], axis=1).to_numpy(dtype=float)
    deltas = hitter_values[:, None, :] - reference_values[None, :, :]
    data["MinDistToPitcher"] = np.sqrt((deltas ** 2).sum(axis=2)).min(axis=1)

    cluster_ready = data.dropna(subset=CLUSTER_FEATURES).copy()
    cluster_ready_scaled = cluster_scaler.transform(cluster_ready[CLUSTER_FEATURES])
    cluster_ready["PitchCluster"] = cluster_model.predict(cluster_ready_scaled)
    cluster_ready["PitchGroup"] = cluster_ready["PitchCluster"].map(cluster_to_type).fillna("Unknown")
    data = data.merge(
        cluster_ready[["Date", "Batter", "Pitcher", "PitchGroup"]],
        on=["Date", "Batter", "Pitcher"],
        how="left",
    )

    cluster_rows: list[dict[str, object]] = []
    similar_pitch_rows: list[pd.DataFrame] = []
    distance_rows: list[pd.DataFrame] = []

    for hitter in hitters:
        hitter_history = data[data["Batter"] == hitter].copy()
        for cluster in pitcher_clusters.to_dict("records"):
            group = cluster["pitch_group"]
            group_all = hitter_history[hitter_history["PitchGroup"] == group].copy()
            group_pitches = group_all[group_all["MinDistToPitcher"] <= similarity_threshold].copy()

            coverage_pct = float(len(group_pitches) / len(group_all)) if len(group_all) > 0 else 0.0
            cluster_rows.append(
                {
                    "cluster_id": cluster["cluster_id"],
                    "pitch_group": group,
                    "dominant_pitch_type": cluster["dominant_pitch_type"],
                    "usage_pct": cluster["usage_pct"],
                    "similar_pitch_count": int(len(group_pitches)),
                    "distance_method": "threshold" if len(group_pitches) else "none",
                    "expected_run_value": float(group_pitches["run_value"].mean()) if not group_pitches.empty else np.nan,
                    "expected_whiff_rate": float(group_pitches["is_whiff"].mean()) if not group_pitches.empty else np.nan,
                    "expected_hard_hit_rate": float(group_pitches["hard_hit"].mean()) if not group_pitches.empty else np.nan,
                    "expected_exit_speed": float(group_pitches["ExitSpeed"].mean()) if (not group_pitches.empty and group_pitches["ExitSpeed"].notna().any()) else np.nan,
                    "coverage_pct": coverage_pct,
                    "Batter": hitter,
                    "Pitcher": pitcher_name,
                }
            )

            distance_frame = group_all[[c for c in ["Date", "MinDistToPitcher"] if c in group_all.columns]].copy()
            if not distance_frame.empty:
                distance_frame = distance_frame.rename(columns={"MinDistToPitcher": "distance"})
                distance_frame["qualifies"] = distance_frame["distance"] <= similarity_threshold
                distance_frame["target_cluster"] = cluster["cluster_id"]
                distance_frame["target_pitch_group"] = group
                distance_frame["target_pitch_type"] = cluster["dominant_pitch_type"]
                distance_frame["Batter"] = hitter
                distance_frame["Pitcher"] = pitcher_name
                distance_rows.append(distance_frame)

            if not group_pitches.empty:
                sample_columns = [
                    "Date", "Batter", "Pitcher", "PitcherThrows", "pitch_type", "pitch_group",
                    "RelSpeed", "InducedVertBreak", "HorzBreak", "SpinRate", "RelHeight", "RelSide",
                    "ArmSlot", "PlateLocSide", "PlateLocHeight", "PitchCall", "PlayResult", "ExitSpeed",
                    "Angle", "run_value", "estimated_woba", "is_whiff", "is_swing", "is_hit",
                    "is_in_play", "hard_hit", "MinDistToPitcher", "source_file"
                ]
                sample_columns = [column for column in sample_columns if column in group_pitches.columns]
                sample_frame = group_pitches[sample_columns].copy().rename(columns={"MinDistToPitcher": "distance"})
                sample_frame["target_cluster"] = cluster["cluster_id"]
                sample_frame["target_pitch_group"] = group
                sample_frame["target_pitch_type"] = cluster["dominant_pitch_type"]
                sample_frame["Batter"] = hitter
                sample_frame["Pitcher"] = pitcher_name
                similar_pitch_rows.append(sample_frame)

    cluster_matchups = pd.DataFrame(cluster_rows)
    matchup_summary = summarize_matchups(cluster_matchups)
    similar_pitches = pd.concat(similar_pitch_rows, ignore_index=True) if similar_pitch_rows else pd.DataFrame()
    distance_samples = pd.concat(distance_rows, ignore_index=True) if distance_rows else pd.DataFrame()

    return MacAnalysis(
        raw_data=data,
        pitcher_clusters=pitcher_clusters,
        pitcher_cluster_points=pitcher_cluster_points,
        matchup_summary=matchup_summary,
        cluster_matchups=cluster_matchups,
        similar_pitches=similar_pitches,
        distance_samples=distance_samples,
        skipped_files=[],
    )


def _map_mlb_pitch_group(pitch_type: str) -> str:
    label = str(pitch_type or "").strip()
    mapping = {
        "Four-Seam": "Fastball",
        "4-Seam Fastball": "Fastball",
        "Fastball": "Fastball",
        "FourSeamFastBall": "Fastball",
        "TwoSeamFastBall": "Sinker",
        "Sinker": "Sinker",
        "Slider": "Slider",
        "Cutter": "Cutter",
        "Curveball": "Curveball",
        "Slurve": "Curveball",
        "Knuckle Curve": "Curveball",
        "Sweeper": "Sweeper",
        "Slow Curve": "Curveball",
        "Eephus": "Curveball",
        "Splitter": "Splitter",
        "Split-Finger": "Splitter",
        "Forkball": "Splitter",
        "ChangeUp": "Changeup",
        "Changeup": "Changeup",
        "Knuckleball": "Knuckleball",
        "Screwball": "Screwball",
    }
    return mapping.get(label, label or "Unknown")



def build_feature_scaler(data: pd.DataFrame, features: list[str]) -> pd.Series:
    scale = data[features].std(ddof=0).replace(0, np.nan).fillna(1.0)
    return scale


def build_pitcher_clusters(
    pitcher_df: pd.DataFrame,
    *,
    max_clusters: int,
) -> tuple[pd.DataFrame, pd.DataFrame, GaussianMixture, StandardScaler, dict[int, str]]:
    working = pitcher_df.dropna(subset=CLUSTER_FEATURES + ["pitch_type"]).copy()
    if working.empty:
        raise ValueError("The selected pitcher does not have enough pitch data.")

    cluster_scaler = StandardScaler()
    x = cluster_scaler.fit_transform(working[CLUSTER_FEATURES])
    max_components = min(max_clusters, 9, len(working))
    min_components = 4 if max_components >= 4 else 2

    if len(working) < 24 or max_components < min_components:
        best_model = GaussianMixture(n_components=1, covariance_type="full", random_state=42)
        best_model.fit(x)
        labels = np.zeros(len(working), dtype=int)
    else:
        bic_scores: list[float] = []
        model_by_k: dict[int, GaussianMixture] = {}
        ks = list(range(min_components, max_components + 1))
        for components in ks:
            model = GaussianMixture(n_components=components, covariance_type="full", random_state=42)
            model.fit(x)
            bic_scores.append(float(model.bic(x)))
            model_by_k[components] = model

        if KneeLocator is not None:
            knee = KneeLocator(ks, bic_scores, curve="convex", direction="decreasing")
            optimal_k = int(knee.elbow) if knee.elbow is not None else ks[int(np.argmin(bic_scores))]
        else:
            optimal_k = ks[int(np.argmin(bic_scores))]
        best_model = model_by_k[optimal_k]
        labels = best_model.predict(x)

    working["raw_cluster_id"] = labels.astype(int)
    cluster_to_type: dict[int, str] = {}
    for cluster_id, cluster_df in working.groupby("raw_cluster_id", sort=True):
        pitch_type_counts = cluster_df["pitch_type"].value_counts()
        most_common_type = pitch_type_counts.index[0] if not pitch_type_counts.empty else "Unknown"
        cluster_to_type[int(cluster_id)] = _map_mlb_pitch_group(most_common_type)

    working["dominant_pitch_type"] = working["raw_cluster_id"].map(cluster_to_type)
    working["pitch_group"] = working["dominant_pitch_type"]

    total_pitches = len(working)
    summaries = []
    for label, cluster_df in working.groupby("dominant_pitch_type", sort=False):
        summaries.append(
            {
                "pitch_count": int(len(cluster_df)),
                "usage_pct": float(len(cluster_df) / total_pitches),
                "pitch_group": label,
                "dominant_pitch_type": label,
                "speed": float(cluster_df["RelSpeed"].mean()),
                "ivb": float(cluster_df["InducedVertBreak"].mean()),
                "hb": float(cluster_df["HorzBreak"].mean()),
                "spin_rate": float(cluster_df["SpinRate"].mean()),
                "rel_height": float(cluster_df["RelHeight"].mean()),
                "rel_side": float(cluster_df["RelSide"].mean()),
                "arm_slot": float(cluster_df["ArmSlot"].mean()),
                "whiff_rate": float(cluster_df["is_whiff"].mean()),
                "hard_hit_rate": float(cluster_df["hard_hit"].mean()),
                "run_value": float(cluster_df["run_value"].mean()),
            }
        )

    summary_df = pd.DataFrame(summaries).sort_values(["usage_pct", "pitch_count"], ascending=[False, False]).reset_index(drop=True)
    summary_df["cluster_id"] = np.arange(len(summary_df))
    merged_id_map = dict(zip(summary_df["dominant_pitch_type"], summary_df["cluster_id"]))
    point_df = working[[
        "pitch_type", "pitch_group", "dominant_pitch_type", "HorzBreak", "InducedVertBreak",
        "RelSpeed", "SpinRate", "RelHeight", "RelSide", "ArmSlot"
    ]].copy()
    point_df["cluster_id"] = point_df["dominant_pitch_type"].map(merged_id_map).astype(int)
    return summary_df, point_df, best_model, cluster_scaler, cluster_to_type


def score_hitter_against_cluster(
    hitter_history: pd.DataFrame,
    *,
    cluster: dict[str, object],
    reference_pitches: pd.DataFrame,
    scaler: pd.Series,
    similarity_threshold: float,
    min_similar_pitches: int,
) -> dict[str, object]:
    if hitter_history.empty or reference_pitches.empty:
        return {
            "cluster_id": cluster["cluster_id"],
            "pitch_group": cluster["pitch_group"],
            "dominant_pitch_type": cluster["dominant_pitch_type"],
            "usage_pct": cluster["usage_pct"],
            "similar_pitch_count": 0,
            "distance_method": "none",
            "expected_run_value": np.nan,
            "expected_whiff_rate": np.nan,
            "expected_hard_hit_rate": np.nan,
            "expected_exit_speed": np.nan,
            "coverage_pct": 0.0,
            "_similar_pitches": pd.DataFrame(),
            "_distance_samples": pd.DataFrame(),
        }

    allowed_pitch_types = _allowed_source_pitch_types(str(cluster["dominant_pitch_type"]))
    working = hitter_history[hitter_history["pitch_type"].isin(allowed_pitch_types)].copy()
    working = working.dropna(subset=SIMILARITY_FEATURES)
    if working.empty:
        return {
            "cluster_id": cluster["cluster_id"],
            "pitch_group": cluster["pitch_group"],
            "dominant_pitch_type": cluster["dominant_pitch_type"],
            "usage_pct": cluster["usage_pct"],
            "similar_pitch_count": 0,
            "distance_method": "none",
            "expected_run_value": np.nan,
            "expected_whiff_rate": np.nan,
            "expected_hard_hit_rate": np.nan,
            "expected_exit_speed": np.nan,
            "coverage_pct": 0.0,
            "_similar_pitches": pd.DataFrame(),
            "_distance_samples": pd.DataFrame(),
        }

    reference = reference_pitches.dropna(subset=SIMILARITY_FEATURES).copy()
    if reference.empty:
        return {
            "cluster_id": cluster["cluster_id"],
            "pitch_group": cluster["pitch_group"],
            "dominant_pitch_type": cluster["dominant_pitch_type"],
            "usage_pct": cluster["usage_pct"],
            "similar_pitch_count": 0,
            "distance_method": "none",
            "expected_run_value": np.nan,
            "expected_whiff_rate": np.nan,
            "expected_hard_hit_rate": np.nan,
            "expected_exit_speed": np.nan,
            "coverage_pct": 0.0,
            "_similar_pitches": pd.DataFrame(),
            "_distance_samples": pd.DataFrame(),
        }

    hitter_values = working[SIMILARITY_FEATURES].to_numpy(dtype=float)
    reference_values = reference[SIMILARITY_FEATURES].to_numpy(dtype=float)
    scale_values = scaler[SIMILARITY_FEATURES].to_numpy(dtype=float)
    diffs = (hitter_values[:, None, :] - reference_values[None, :, :]) / scale_values[None, None, :]
    distances = np.sqrt((diffs**2).sum(axis=2))
    working["distance"] = distances.min(axis=1)

    similar = working[working["distance"] <= similarity_threshold].copy()
    distance_method = "threshold"

    if min_similar_pitches > 0 and len(similar) < min_similar_pitches:
        similar = working.nsmallest(min(min_similar_pitches, len(working)), "distance").copy()
        distance_method = "nearest"

    coverage_pct = (
        len(similar) / len(working)
        if len(working) > 0
        else 0.0
    )

    sample_columns = [
        "Date",
        "Batter",
        "Pitcher",
        "PitcherThrows",
        "pitch_type",
        "pitch_group",
        "RelSpeed",
        "InducedVertBreak",
        "HorzBreak",
        "SpinRate",
        "RelHeight",
        "RelSide",
        "ArmSlot",
        "PlateLocSide",
        "PlateLocHeight",
        "PitchCall",
        "PlayResult",
        "ExitSpeed",
        "Angle",
        "run_value",
        "estimated_woba",
        "is_whiff",
        "is_swing",
        "is_hit",
        "is_in_play",
        "hard_hit",
        "distance",
        "source_file",
    ]
    sample_columns = [column for column in sample_columns if column in similar.columns]
    distance_columns = [column for column in ["Date", "distance"] if column in working.columns]
    distance_frame = working[distance_columns].copy()
    distance_frame["qualifies"] = working["distance"] <= similarity_threshold

    return {
        "cluster_id": cluster["cluster_id"],
        "pitch_group": cluster["pitch_group"],
        "dominant_pitch_type": cluster["dominant_pitch_type"],
        "usage_pct": cluster["usage_pct"],
        "similar_pitch_count": int(len(similar)),
        "distance_method": distance_method,
        "expected_run_value": float(similar["run_value"].mean()),
        "expected_whiff_rate": float(similar["is_whiff"].mean()),
        "expected_hard_hit_rate": float(similar["hard_hit"].mean()),
        "expected_exit_speed": float(similar["ExitSpeed"].mean())
        if similar["ExitSpeed"].notna().any()
        else np.nan,
        "coverage_pct": float(coverage_pct),
        "_similar_pitches": similar[sample_columns],
        "_distance_samples": distance_frame,
    }


def summarize_matchups(cluster_matchups: pd.DataFrame) -> pd.DataFrame:
    if cluster_matchups.empty:
        return pd.DataFrame()

    rows = []
    for batter, batter_df in cluster_matchups.groupby("Batter"):
        usage = batter_df["usage_pct"].fillna(0.0)
        rv = batter_df["expected_run_value"]
        whiff = batter_df["expected_whiff_rate"]
        hard_hit = batter_df["expected_hard_hit_rate"]
        ev = batter_df["expected_exit_speed"]

        weight_sum = usage[rv.notna()].sum()
        if weight_sum == 0:
            mac_score = np.nan
        else:
            mac_score = np.average(rv.dropna(), weights=usage[rv.notna()])

        whiff_sum = usage[whiff.notna()].sum()
        expected_whiff = (
            np.average(whiff.dropna(), weights=usage[whiff.notna()])
            if whiff_sum
            else np.nan
        )

        hard_hit_sum = usage[hard_hit.notna()].sum()
        expected_hard_hit = (
            np.average(hard_hit.dropna(), weights=usage[hard_hit.notna()])
            if hard_hit_sum
            else np.nan
        )

        ev_sum = usage[ev.notna()].sum()
        expected_ev = (
            np.average(ev.dropna(), weights=usage[ev.notna()])
            if ev_sum
            else np.nan
        )

        rows.append(
            {
                "Batter": batter,
                "MAC_Score": mac_score,
                "Expected_Whiff_Rate": expected_whiff,
                "Expected_Hard_Hit_Rate": expected_hard_hit,
                "Expected_EV": expected_ev,
                "Pitch_Clusters_Covered": int(batter_df["expected_run_value"].notna().sum()),
                "Total_Similar_Pitches": int(batter_df["similar_pitch_count"].sum()),
                "Weighted_Coverage": float(
                    np.average(
                        batter_df["coverage_pct"].fillna(0.0),
                        weights=batter_df["usage_pct"].fillna(0.0),
                    )
                ),
            }
        )

    summary = pd.DataFrame(rows).sort_values("MAC_Score", ascending=True).reset_index(drop=True)
    summary["Pitcher_Friendly"] = summary["MAC_Score"].apply(_score_band)
    return summary


def _collapse_pitch_group(pitch_type: str) -> str:
    if pitch_type is None:
        return "Other"

    label = str(pitch_type).strip().lower()
    if "fastball" in label or label in {"sinker", "cutter"}:
        return "Fastball"
    if label in {"slider", "curveball", "sweeper", "slurve", "knuckle curve"}:
        return "Breaking Ball"
    if label in {"changeup", "splitter", "forkball", "split-finger"}:
        return "Offspeed"
    return "Other"


def _allowed_source_pitch_types(target_pitch_type: str) -> set[str]:
    target = str(target_pitch_type or "").strip()
    return ALLOWED_SOURCE_PITCH_TYPES.get(target, {target})


def _map_statcast_pitch_call(description: object) -> str:
    text = str(description).strip().lower()
    mapping = {
        "swinging_strike": "SwingingStrike",
        "swinging_strike_blocked": "SwingingStrikeBlocked",
        "foul": "FoulBall",
        "foul_tip": "FoulBall",
        "foul_bunt": "FoulBall",
        "hit_into_play": "InPlay",
        "hit_into_play_no_out": "InPlay",
        "hit_into_play_score": "InPlay",
        "blocked_ball": "Ball",
        "ball": "Ball",
        "called_strike": "CalledStrike",
        "missed_bunt": "SwingingStrike",
        "pitchout": "Ball",
        "foul_pitchout": "FoulBall",
    }
    return mapping.get(text, "Other")


def _map_statcast_play_result(events: object) -> str:
    text = str(events).strip().lower()
    if text in {"single"}:
        return "Single"
    if text in {"double"}:
        return "Double"
    if text in {"triple"}:
        return "Triple"
    if text in {"home_run"}:
        return "HomeRun"
    if text in {"field_error"}:
        return "Error"
    if text in {"sac_fly", "sac_bunt", "sac_fly_double_play"}:
        return "Sacrifice"
    if text in {"fielders_choice"}:
        return "FieldersChoice"
    if text in {
        "field_out",
        "force_out",
        "double_play",
        "triple_play",
        "grounded_into_double_play",
        "fielders_choice_out",
        "strikeout",
        "strikeout_double_play",
        "other_out",
        "pickoff_1b",
        "pickoff_2b",
        "pickoff_3b",
        "pickoff_caught_stealing_2b",
        "pickoff_caught_stealing_3b",
        "pickoff_caught_stealing_home",
        "caught_stealing_2b",
        "caught_stealing_3b",
        "caught_stealing_home",
    }:
        return "Out"
    return "Other"


def _score_band(value: float) -> str:
    if pd.isna(value):
        return "No data"
    if value <= -0.05:
        return "Strong"
    if value <= 0.05:
        return "Solid"
    if value <= 0.20:
        return "Neutral"
    return "Danger"
