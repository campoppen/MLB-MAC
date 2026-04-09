from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from mac_engine import load_dataset, run_mac


APP_DIR = __import__("pathlib").Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = str(APP_DIR / "data")
SIMILARITY_THRESHOLD = 0.6
MIN_COMP_PITCHES = 0
MAX_PITCH_CLUSTERS = 8
DEFAULT_PITCHER = "Flaherty, Jack"
DEFAULT_OPPONENT_TEAM = "NYY"
DEFAULT_HITTER_NAMES = [
    "Gleyber Torres",
    "Juan Soto",
    "Aaron Judge",
    "Giancarlo Stanton",
    "Jazz Chisholm Jr",
    "Anthony Rizzo",
    "Anthony Volpe",
    "Austin Wells",
    "Alex Verdugo",
]
DEFAULT_HITTER_ID_HINTS = {
    "NYY": ["650402", "665742", "592450", "519317", "665862", "519203", "683011", "669224", "657077"],
}
PLAYER_ID_NAME_OVERRIDES = {}
PITCH_TYPE_BASE_COLORS = {
    "4-Seam Fastball": "#d94841",
    "Fastball": "#d94841",
    "Sinker": "#f28e2b",
    "Cutter": "#ff9d76",
    "Slider": "#2f6fed",
    "Sweeper": "#5e3de6",
    "Curveball": "#1565c0",
    "Knuckle Curve": "#4c78a8",
    "Changeup": "#5a9e4b",
    "Splitter": "#2ca02c",
    "Forkball": "#3aa657",
}
PITCH_TYPE_FALLBACK_COLORS = [
    "#d94841",
    "#f28e2b",
    "#ff9d76",
    "#2f6fed",
    "#5e3de6",
    "#1565c0",
    "#4c78a8",
    "#5a9e4b",
    "#2ca02c",
    "#3aa657",
    "#b07aa1",
    "#9c755f",
]
STANDARD_STAT_COLUMNS = [
    "RV/100",
    "AVG",
    "Whiff%",
    "SwStr%",
    "HH%",
    "ExitVelo",
    "Launch",
    "GB%",
    "wOBA",
    "Pitches",
    "Hit Into Play",
]
STANDARD_PITCH_TYPE_COLUMNS = [
    "Pitch Type",
    *STANDARD_STAT_COLUMNS[:-1],
    "Usage Weight",
    "Hit Into Play",
]


def pitch_type_sort_key(value: str) -> tuple[str, str]:
    text = str(value or "").strip()
    return (text.lower(), text)


def normalize_comparison_pitch_type(pitch_type: str) -> str:
    label = str(pitch_type or "").strip().lower()
    if label in {"curveball", "knuckle curve"}:
        return "Curveball"
    if label in {"changeup", "splitter", "forkball", "split-finger"}:
        return "Changeup" if label == "changeup" else "Splitter"
    return str(pitch_type or "").strip()


def build_pitch_type_color_scale(pitch_types: list[str]) -> alt.Scale:
    ordered = []
    seen = set()
    for pitch_type in pitch_types:
        if pitch_type not in seen:
            ordered.append(pitch_type)
            seen.add(pitch_type)

    ranges: list[str] = []
    fallback_index = 0
    for pitch_type in ordered:
        if pitch_type == "Overall":
            ranges.append("#ffffff")
        elif pitch_type in PITCH_TYPE_BASE_COLORS:
            ranges.append(PITCH_TYPE_BASE_COLORS[pitch_type])
        else:
            ranges.append(PITCH_TYPE_FALLBACK_COLORS[fallback_index % len(PITCH_TYPE_FALLBACK_COLORS)])
            fallback_index += 1

    return alt.Scale(domain=ordered, range=ranges)


def format_person_name(name: str) -> str:
    if pd.isna(name):
        return ""
    text = str(name).strip()
    if "," not in text:
        return text
    last, first = [part.strip() for part in text.split(",", 1)]
    return f"{first} {last}".strip()


def format_slug_name(slug: str) -> str:
    parts = [part for part in slug.strip().split("-") if part]
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return " ".join(part.capitalize() for part in parts)


@st.cache_data(show_spinner=False)
def lookup_mlb_player_name(player_id: str) -> str:
    text = str(player_id).strip()
    if not text.isdigit():
        return text
    if text in PLAYER_ID_NAME_OVERRIDES:
        return PLAYER_ID_NAME_OVERRIDES[text]

    url = f"https://www.mlb.com/player/{text}"
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
        },
    )
    try:
        with urlopen(request, timeout=8) as response:
            final_url = response.geturl()
    except Exception:
        return text

    path_parts = [part for part in urlparse(final_url).path.split("/") if part]
    if len(path_parts) >= 2 and path_parts[0] == "player":
        return format_slug_name(path_parts[1])
    return text


def display_name(value: str) -> str:
    text = str(value).strip()
    if text.isdigit():
        return lookup_mlb_player_name(text)
    return format_person_name(text)


def normalize_person_name(name: str) -> str:
    text = str(name or "").strip().lower()
    replacements = {
        ".": "",
        ",": "",
        "-": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return " ".join(text.split())


def last_name_sort_key(name: str) -> tuple[str, str]:
    if pd.isna(name):
        return ("", "")
    text = display_name(name)
    if "," in text:
        last, first = [part.strip() for part in text.split(",", 1)]
        return (last.lower(), first.lower())
    parts = text.split()
    if not parts:
        return ("", "")
    return (parts[-1].lower(), " ".join(parts[:-1]).lower())


st.set_page_config(page_title="MAC Clone", layout="wide")

st.title("MLB MAC")


@st.cache_data(show_spinner=False)
def cached_load_dataset(data_root: str) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    return load_dataset(data_root)


st.sidebar.caption("Data refresh")
reload_requested = st.sidebar.button("Reload data", width="stretch")

if reload_requested:
    cached_load_dataset.clear()

data, skipped_files = cached_load_dataset(DEFAULT_DATA_ROOT)

if data.empty:
    st.error("No usable CSV data was found. Check the folder path and file schema.")
    st.stop()

st.sidebar.metric("Usable pitches", f"{len(data):,}")
st.sidebar.metric("Pitchers", f"{data['Pitcher'].nunique():,}")
st.sidebar.metric("Batters", f"{data['Batter'].nunique():,}")
st.sidebar.metric("Skipped files", f"{len(skipped_files):,}")

pitcher_team_map = (
    data.dropna(subset=["Pitcher", "PitcherTeam"])
    .groupby("Pitcher")["PitcherTeam"]
    .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0])
    .to_dict()
)

pitchers = sorted(data["Pitcher"].dropna().unique().tolist(), key=last_name_sort_key)
pitcher_display_map = {pitcher: format_person_name(pitcher) for pitcher in pitchers}
default_pitcher_index = pitchers.index(DEFAULT_PITCHER) if DEFAULT_PITCHER in pitchers else 0
selected_pitcher = st.selectbox(
    "Current Pitcher",
    pitchers,
    index=default_pitcher_index,
    format_func=lambda value: pitcher_display_map.get(value, value),
)
selected_pitcher_team = pitcher_team_map.get(selected_pitcher)
opponent_teams = sorted(
    team
    for team in data["BatterTeam"].dropna().unique().tolist()
    if team and team != selected_pitcher_team
)
default_opponent_index = opponent_teams.index(DEFAULT_OPPONENT_TEAM) if DEFAULT_OPPONENT_TEAM in opponent_teams else 0
selected_opponent_team = st.selectbox("Opposing Team", opponent_teams, index=default_opponent_index)


def derive_lineup(source_data: pd.DataFrame, opponent_team: str) -> list[str]:
    opponent_df = source_data[source_data["BatterTeam"] == opponent_team].copy()
    if opponent_df.empty:
        return []

    if "pa_key" not in opponent_df.columns:
        pa_parts = [
            opponent_df.get("GameID", pd.Series("", index=opponent_df.index)).astype("string").fillna(""),
            opponent_df.get("Top/Bottom", pd.Series("", index=opponent_df.index)).astype("string").fillna(""),
            pd.to_numeric(
                opponent_df.get("Inning", pd.Series(pd.NA, index=opponent_df.index)),
                errors="coerce",
            ).astype("Int64").astype("string").fillna(""),
            pd.to_numeric(
                opponent_df.get("PAofInning", pd.Series(pd.NA, index=opponent_df.index)),
                errors="coerce",
            ).astype("Int64").astype("string").fillna(""),
            opponent_df["Batter"].astype("string").fillna(""),
        ]
        opponent_df["pa_key"] = pa_parts[0]
        for part in pa_parts[1:]:
            opponent_df["pa_key"] = opponent_df["pa_key"] + "|" + part

    pa_summary = (
        opponent_df.dropna(subset=["Batter"])
        .groupby("Batter", as_index=False)
        .agg(
            PA=("pa_key", "nunique"),
            PitchCount=("Batter", "size"),
        )
        .assign(sort_key=lambda frame: frame["Batter"].map(last_name_sort_key))
        .sort_values(["PA", "PitchCount", "sort_key"], ascending=[False, False, True])
        .drop(columns="sort_key")
    )
    return pa_summary["Batter"].head(9).tolist()


def build_summary_plot(
    cluster_matchups: pd.DataFrame,
    summary: pd.DataFrame,
    similar_pitches: pd.DataFrame,
) -> alt.Chart:
    detail = cluster_matchups.copy()
    detail["weighted_rv"] = detail["expected_run_value"] * detail["usage_pct"]
    detail["PitchType"] = detail["dominant_pitch_type"]
    detail["BatterDisplay"] = detail["Batter"].map(display_name)
    detail = (
        detail.groupby(["Batter", "BatterDisplay", "PitchType"], as_index=False)
        .agg(
            weighted_rv=("weighted_rv", "sum"),
            usage_sum=("usage_pct", "sum"),
            similar_pitch_count=("similar_pitch_count", "sum"),
        )
    )
    detail = detail[detail["similar_pitch_count"] > 0].copy()
    detail["RV100"] = np.where(
        detail["usage_sum"] > 0,
        (detail["weighted_rv"] / detail["usage_sum"]) * 100,
        np.nan,
    )
    detail["PointSize"] = 65.0

    overall = summary.copy()
    overall["RV100"] = overall["MAC_Score"] * 100
    overall["PitchType"] = "Overall"
    overall["PointSize"] = 125.0
    overall["BatterDisplay"] = overall["Batter"].map(display_name)
    batter_order = (
        summary["Batter"]
        .drop_duplicates()
        .tolist()
    )
    batter_order = [display_name(name) for name in sorted(batter_order, key=last_name_sort_key)]

    detail_df = detail[["BatterDisplay", "PitchType", "RV100", "PointSize"]].copy()
    overall_df = overall[["BatterDisplay", "PitchType", "RV100", "PointSize"]].copy()

    detail_stats = build_all_hitters_pitch_type_summary(
        similar_pitches,
        cluster_matchups,
        summary["Batter"].dropna().tolist(),
    )
    if not detail_stats.empty:
        detail_df = detail_df.merge(
            detail_stats,
            left_on=["BatterDisplay", "PitchType"],
            right_on=["Hitter", "Pitch Type"],
            how="left",
        )

    overall_stats = build_lineup_stat_summary(similar_pitches, summary["Batter"].dropna().tolist())
    if not overall_stats.empty:
        overall_df = overall_df.merge(
            overall_stats,
            left_on="BatterDisplay",
            right_on="Hitter",
            how="left",
        )

    band_df = pd.DataFrame(
        {
            "label": ["25th percentile", "75th percentile"],
            "value": [
                summary["MAC_Score"].quantile(0.25) * 100,
                summary["MAC_Score"].quantile(0.75) * 100,
            ],
        }
    )

    color_scale = build_pitch_type_color_scale(detail_df["PitchType"].tolist() + ["Overall"])

    points = (
        alt.Chart(detail_df)
        .mark_circle(opacity=0.9)
        .encode(
            x=alt.X(
                "BatterDisplay:N",
                sort=batter_order,
                title=None,
                axis=alt.Axis(labelAngle=-45, labelAlign="right", labelLimit=180),
            ),
            y=alt.Y("RV100:Q", title="Better for Pitchers  <---- RV/100 ---->  Better for Hitters"),
            color=alt.Color("PitchType:N", scale=color_scale, title=None),
            size=alt.Size("PointSize:Q", legend=None),
            tooltip=[
                alt.Tooltip("BatterDisplay:N", title="Hitter"),
                alt.Tooltip("PitchType:N", title="Pitch type"),
                alt.Tooltip("RV100:Q", title="RV/100", format=".2f"),
                alt.Tooltip("AVG:Q", title="AVG", format=".3f"),
                alt.Tooltip("Whiff%:Q", title="Whiff%", format=".1%"),
                alt.Tooltip("SwStr%:Q", title="SwStr%", format=".1%"),
                alt.Tooltip("HH%:Q", title="HH%", format=".1%"),
                alt.Tooltip("ExitVelo:Q", title="ExitVelo", format=".1f"),
                alt.Tooltip("Launch:Q", title="Launch", format=".1f"),
                alt.Tooltip("GB%:Q", title="GB%", format=".1%"),
                alt.Tooltip("wOBA:Q", title="wOBA", format=".3f"),
                alt.Tooltip("Pitches:Q", title="Pitches", format=".0f"),
                alt.Tooltip("Hit Into Play:Q", title="Hit Into Play", format=".0f"),
            ],
        )
    )

    overall_points = (
        alt.Chart(overall_df)
        .mark_circle(opacity=1.0, stroke="#111111", strokeWidth=1.6)
        .encode(
            x=alt.X(
                "BatterDisplay:N",
                sort=batter_order,
                title=None,
                axis=alt.Axis(labelAngle=-45, labelAlign="right", labelLimit=180),
            ),
            y=alt.Y("RV100:Q", title="Better for Pitchers  <---- RV/100 ---->  Better for Hitters"),
            size=alt.Size("PointSize:Q", legend=None),
            color=alt.value("#ffffff"),
            tooltip=[
                alt.Tooltip("BatterDisplay:N", title="Hitter"),
                alt.Tooltip("RV100:Q", title="Overall RV/100", format=".2f"),
                alt.Tooltip("AVG:Q", title="AVG", format=".3f"),
                alt.Tooltip("Whiff%:Q", title="Whiff%", format=".1%"),
                alt.Tooltip("SwStr%:Q", title="SwStr%", format=".1%"),
                alt.Tooltip("HH%:Q", title="HH%", format=".1%"),
                alt.Tooltip("ExitVelo:Q", title="ExitVelo", format=".1f"),
                alt.Tooltip("Launch:Q", title="Launch", format=".1f"),
                alt.Tooltip("GB%:Q", title="GB%", format=".1%"),
                alt.Tooltip("wOBA:Q", title="wOBA", format=".3f"),
                alt.Tooltip("Pitches:Q", title="Pitches", format=".0f"),
                alt.Tooltip("Hit Into Play:Q", title="Hit Into Play", format=".0f"),
            ],
        )
    )

    rules = (
        alt.Chart(band_df)
        .mark_rule(strokeDash=[6, 6], color="#b8bcc2")
        .encode(y="value:Q")
    )

    return (rules + overall_points + points).properties(height=430)


def build_rankings_table(rankings_df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def color_value(value: float) -> str:
        if pd.isna(value):
            return "background-color: #5a5f69; color: #f3f4f6;"
        if value <= -5:
            return "background-color: #0b7d28; color: white;"
        if value <= -1:
            return "background-color: #8fdb8d; color: #111111;"
        if value < 1:
            return "background-color: #f7f3c6; color: #111111;"
        if value < 5:
            return "background-color: #f3a0a0; color: #111111;"
        return "background-color: #9e0000; color: white;"

    hitter_columns = [column for column in rankings_df.columns if column != "Pitcher"]
    return rankings_df.style.format({column: "{:.2f}" for column in hitter_columns}).map(
        color_value, subset=hitter_columns
    )


def build_distance_distribution_plot(
    distance_samples: pd.DataFrame,
    *,
    similarity_threshold: float,
) -> alt.Chart:
    plot_df = distance_samples.copy()
    plot_df["Status"] = np.where(plot_df["qualifies"], "Qualifies", "Does Not Qualify")

    bars = (
        alt.Chart(plot_df)
        .mark_bar(opacity=0.85)
        .encode(
            x=alt.X("distance:Q", bin=alt.Bin(maxbins=40), title="Euclidean Distance"),
            y=alt.Y("count():Q", title="Pitch Count"),
            color=alt.Color(
                "Status:N",
                scale=alt.Scale(
                    domain=["Qualifies", "Does Not Qualify"],
                    range=["#5a9e4b", "#4c78a8"],
                ),
                title=None,
            ),
            tooltip=[
                alt.Tooltip("count():Q", title="Pitches"),
                alt.Tooltip("Status:N", title="Status"),
            ],
        )
    )

    threshold_df = pd.DataFrame({"threshold": [similarity_threshold]})
    threshold_rule = (
        alt.Chart(threshold_df)
        .mark_rule(color="#d94841", strokeDash=[6, 4], strokeWidth=2)
        .encode(x="threshold:Q")
    )

    return (bars + threshold_rule).properties(height=320)


def build_cluster_plot(cluster_points: pd.DataFrame) -> alt.Chart:
    plot_df = cluster_points.copy()
    plot_df["ClusterLabel"] = plot_df["dominant_pitch_type"].astype(str)
    cluster_order = (
        plot_df[["cluster_id", "ClusterLabel"]]
        .drop_duplicates()
        .sort_values("cluster_id")["ClusterLabel"]
        .tolist()
    )
    return (
        alt.Chart(plot_df)
        .mark_circle(size=70, opacity=0.75)
        .encode(
            x=alt.X("HorzBreak:Q", title="Horizontal Break"),
            y=alt.Y("InducedVertBreak:Q", title="Induced Vertical Break"),
            color=alt.Color("ClusterLabel:N", sort=cluster_order, title="Pitch Type"),
            tooltip=[
                alt.Tooltip("ClusterLabel:N", title="Pitch Type"),
                alt.Tooltip("pitch_type:N", title="Tagged pitch"),
                alt.Tooltip("HorzBreak:Q", title="HB", format=".1f"),
                alt.Tooltip("InducedVertBreak:Q", title="IVB", format=".1f"),
            ],
        )
        .properties(height=360)
    )


def build_similar_pitch_movement_plot(similar_pitches: pd.DataFrame) -> alt.Chart:
    plot_df = similar_pitches.copy()
    plot_df["BatterDisplay"] = plot_df["Batter"].map(display_name)
    plot_df["DisplayHB"] = -plot_df["HorzBreak"]

    color_scale = build_pitch_type_color_scale(
        plot_df["target_pitch_type"].dropna().astype(str).tolist()
    )

    return (
        alt.Chart(plot_df)
        .mark_circle(size=65, opacity=0.7)
        .encode(
            x=alt.X(
                "DisplayHB:Q",
                title="Horizontal Break",
                scale=alt.Scale(domain=[-25, 25]),
            ),
            y=alt.Y(
                "InducedVertBreak:Q",
                title="Induced Vertical Break",
                scale=alt.Scale(domain=[-25, 25]),
            ),
            color=alt.Color("target_pitch_type:N", title="Pitch Type", scale=color_scale),
            tooltip=[
                alt.Tooltip("BatterDisplay:N", title="Hitter"),
                alt.Tooltip("target_pitch_type:N", title="Target Type"),
                alt.Tooltip("pitch_type:N", title="Similar Pitch"),
                alt.Tooltip("SpinRate:Q", title="Spin Rate", format=".0f"),
                alt.Tooltip("DisplayHB:Q", title="HB", format=".1f"),
                alt.Tooltip("InducedVertBreak:Q", title="IVB", format=".1f"),
                alt.Tooltip("distance:Q", title="Distance", format=".3f"),
            ],
        )
        .properties(width=600, height=420)
    )


def build_current_pitcher_detail(hitter_cluster_view: pd.DataFrame) -> pd.DataFrame:
    if hitter_cluster_view.empty:
        return pd.DataFrame()

    grouped = (
        hitter_cluster_view.groupby(["dominant_pitch_type"], as_index=False)
        .apply(
            lambda frame: pd.Series(
                {
                    "usage_pct": frame["usage_pct"].sum(),
                    "expected_run_value": np.average(
                        frame["expected_run_value"].dropna(),
                        weights=frame.loc[frame["expected_run_value"].notna(), "usage_pct"],
                    )
                    if frame["expected_run_value"].notna().any()
                    else np.nan,
                    "expected_whiff_rate": np.average(
                        frame["expected_whiff_rate"].dropna(),
                        weights=frame.loc[frame["expected_whiff_rate"].notna(), "usage_pct"],
                    )
                    if frame["expected_whiff_rate"].notna().any()
                    else np.nan,
                    "expected_hard_hit_rate": np.average(
                        frame["expected_hard_hit_rate"].dropna(),
                        weights=frame.loc[frame["expected_hard_hit_rate"].notna(), "usage_pct"],
                    )
                    if frame["expected_hard_hit_rate"].notna().any()
                    else np.nan,
                    "expected_exit_speed": np.average(
                        frame["expected_exit_speed"].dropna(),
                        weights=frame.loc[frame["expected_exit_speed"].notna(), "usage_pct"],
                    )
                    if frame["expected_exit_speed"].notna().any()
                    else np.nan,
                }
            )
        )
        .reset_index(drop=True)
    )
    return grouped.sort_values(["usage_pct", "dominant_pitch_type"], ascending=[False, True]).reset_index(drop=True)


def build_hitter_stat_summary(hitter_samples: pd.DataFrame) -> pd.DataFrame:
    if hitter_samples.empty:
        return pd.DataFrame()

    summary_frames = []
    type_order = ["Overall"] + sorted(
        hitter_samples["target_pitch_type"].dropna().astype(str).unique().tolist(),
        key=pitch_type_sort_key,
    )

    for pitch_type in type_order:
        if pitch_type == "Overall":
            sample = hitter_samples.copy()
        else:
            sample = hitter_samples[hitter_samples["target_pitch_type"] == pitch_type].copy()
        if sample.empty:
            continue

        swing_col = sample["is_swing"] if "is_swing" in sample.columns else pd.Series(0, index=sample.index)
        in_play_col = sample["is_in_play"] if "is_in_play" in sample.columns else pd.Series(0, index=sample.index)
        is_hit_col = sample["is_hit"] if "is_hit" in sample.columns else pd.Series(0, index=sample.index)
        whiff_col = sample["is_whiff"] if "is_whiff" in sample.columns else pd.Series(0, index=sample.index)
        hard_hit_col = sample["hard_hit"] if "hard_hit" in sample.columns else pd.Series(np.nan, index=sample.index)
        woba_col = sample["estimated_woba"] if "estimated_woba" in sample.columns else pd.Series(np.nan, index=sample.index)
        angle_col = sample["Angle"] if "Angle" in sample.columns else pd.Series(np.nan, index=sample.index)
        exit_velo_col = sample["ExitSpeed"] if "ExitSpeed" in sample.columns else pd.Series(np.nan, index=sample.index)

        swings = swing_col.sum()
        in_play = in_play_col.sum()

        row = {
            "Pitch Type": pitch_type,
            "RV/100": sample["run_value"].mean() * 100 if "run_value" in sample.columns else np.nan,
            "wOBA": woba_col.mean() if "estimated_woba" in sample.columns else np.nan,
            "AVG": is_hit_col.sum() / in_play if in_play else np.nan,
            "Whiff%": whiff_col.sum() / swings if swings else np.nan,
            "SwStr%": whiff_col.sum() / len(sample) if len(sample) else np.nan,
            "HH%": hard_hit_col[in_play_col == 1].mean()
            if "hard_hit" in sample.columns and in_play
            else np.nan,
            "GB%": (
                ((angle_col < 10) & angle_col.notna()).sum() / in_play
                if "Angle" in sample.columns and in_play
                else np.nan
            ),
            "ExitVelo": exit_velo_col.mean() if "ExitSpeed" in sample.columns else np.nan,
            "Launch": angle_col.mean() if "Angle" in sample.columns else np.nan,
            "Pitches": int(len(sample)),
            "Hit Into Play": int(in_play),
        }
        summary_frames.append(pd.DataFrame([row]))

    if not summary_frames:
        return pd.DataFrame()

    return pd.concat(summary_frames, ignore_index=True)


def build_lineup_stat_summary(similar_pitches: pd.DataFrame, hitters: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    has_batter_column = not similar_pitches.empty and "Batter" in similar_pitches.columns

    for hitter in hitters:
        if has_batter_column:
            sample = similar_pitches[similar_pitches["Batter"] == hitter].copy()
        else:
            sample = pd.DataFrame()
        if sample.empty:
            rows.append(
                {
                    "Hitter": format_person_name(hitter),
                    "Hitter": display_name(hitter),
                    "RV/100": np.nan,
                    "AVG": np.nan,
                    "Whiff%": np.nan,
                    "SwStr%": np.nan,
                    "ExitVelo": np.nan,
                    "Launch": np.nan,
                    "HH%": np.nan,
                    "GB%": np.nan,
                    "Pitches": 0,
                    "Hit Into Play": 0,
                    "wOBA": np.nan,
                }
            )
            continue

        swing_col = sample["is_swing"] if "is_swing" in sample.columns else pd.Series(0, index=sample.index)
        in_play_col = sample["is_in_play"] if "is_in_play" in sample.columns else pd.Series(0, index=sample.index)
        is_hit_col = sample["is_hit"] if "is_hit" in sample.columns else pd.Series(0, index=sample.index)
        whiff_col = sample["is_whiff"] if "is_whiff" in sample.columns else pd.Series(0, index=sample.index)
        hard_hit_col = sample["hard_hit"] if "hard_hit" in sample.columns else pd.Series(np.nan, index=sample.index)
        woba_col = sample["estimated_woba"] if "estimated_woba" in sample.columns else pd.Series(np.nan, index=sample.index)
        angle_col = sample["Angle"] if "Angle" in sample.columns else pd.Series(np.nan, index=sample.index)
        exit_velo_col = sample["ExitSpeed"] if "ExitSpeed" in sample.columns else pd.Series(np.nan, index=sample.index)

        swings = swing_col.sum()
        in_play = in_play_col.sum()

        rows.append(
            {
                    "RV/100": sample["run_value"].mean() * 100 if "run_value" in sample.columns else np.nan,
                    "Hitter": display_name(hitter),
                    "AVG": is_hit_col.sum() / in_play if in_play else np.nan,
                "Whiff%": whiff_col.sum() / swings if swings else np.nan,
                "SwStr%": whiff_col.sum() / len(sample) if len(sample) else np.nan,
                "ExitVelo": exit_velo_col.mean() if "ExitSpeed" in sample.columns else np.nan,
                "Launch": angle_col.mean() if "Angle" in sample.columns else np.nan,
                "HH%": hard_hit_col[in_play_col == 1].mean()
                if "hard_hit" in sample.columns and in_play
                else np.nan,
                "GB%": (
                    ((angle_col < 10) & angle_col.notna()).sum() / in_play
                    if "Angle" in sample.columns and in_play
                    else np.nan
                ),
                "Pitches": int(len(sample)),
                "Hit Into Play": int(in_play),
                "wOBA": woba_col.mean() if "estimated_woba" in sample.columns else np.nan,
            }
        )

    return pd.DataFrame(rows)


def build_pitch_type_stat_summary(hitter_samples: pd.DataFrame, hitter_cluster_view: pd.DataFrame) -> pd.DataFrame:
    if hitter_samples.empty:
        return pd.DataFrame()

    usage_lookup = (
        hitter_cluster_view.groupby("dominant_pitch_type", as_index=False)["usage_pct"]
        .sum()
        .rename(columns={"dominant_pitch_type": "Pitch Type", "usage_pct": "Usage Weight"})
    )

    rows: list[dict[str, object]] = []
    pitch_types = sorted(
        hitter_samples["target_pitch_type"].dropna().astype(str).unique().tolist(),
        key=pitch_type_sort_key,
    )
    for pitch_type in pitch_types:
        sample = hitter_samples[hitter_samples["target_pitch_type"] == pitch_type].copy()
        if sample.empty:
            continue

        swing_col = sample["is_swing"] if "is_swing" in sample.columns else pd.Series(0, index=sample.index)
        in_play_col = sample["is_in_play"] if "is_in_play" in sample.columns else pd.Series(0, index=sample.index)
        is_hit_col = sample["is_hit"] if "is_hit" in sample.columns else pd.Series(0, index=sample.index)
        whiff_col = sample["is_whiff"] if "is_whiff" in sample.columns else pd.Series(0, index=sample.index)
        hard_hit_col = sample["hard_hit"] if "hard_hit" in sample.columns else pd.Series(np.nan, index=sample.index)
        woba_col = sample["estimated_woba"] if "estimated_woba" in sample.columns else pd.Series(np.nan, index=sample.index)
        angle_col = sample["Angle"] if "Angle" in sample.columns else pd.Series(np.nan, index=sample.index)
        exit_velo_col = sample["ExitSpeed"] if "ExitSpeed" in sample.columns else pd.Series(np.nan, index=sample.index)

        swings = swing_col.sum()
        in_play = in_play_col.sum()
        usage_weight = usage_lookup.loc[usage_lookup["Pitch Type"] == pitch_type, "Usage Weight"]

        rows.append(
            {
                "Pitch Type": pitch_type,
                "AVG": is_hit_col.sum() / in_play if in_play else np.nan,
                "RV/100": sample["run_value"].mean() * 100 if "run_value" in sample.columns else np.nan,
                "Whiff%": whiff_col.sum() / swings if swings else np.nan,
                "SwStr%": whiff_col.sum() / len(sample) if len(sample) else np.nan,
                "HH%": hard_hit_col[in_play_col == 1].mean()
                if "hard_hit" in sample.columns and in_play
                else np.nan,
                "ExitVelo": exit_velo_col.mean() if "ExitSpeed" in sample.columns else np.nan,
                "Launch": angle_col.mean() if "Angle" in sample.columns else np.nan,
                "GB%": (
                    ((angle_col < 10) & angle_col.notna()).sum() / in_play
                    if "Angle" in sample.columns and in_play
                    else np.nan
                ),
                "Usage Weight": float(usage_weight.iloc[0]) if not usage_weight.empty else np.nan,
                "Pitches": int(len(sample)),
                "Hit Into Play": int(in_play),
                "wOBA": woba_col.mean() if "estimated_woba" in sample.columns else np.nan,
            }
        )

    return pd.DataFrame(rows)


def build_all_hitters_pitch_type_summary(
    similar_pitches: pd.DataFrame,
    cluster_matchups: pd.DataFrame,
    hitters: list[str],
) -> pd.DataFrame:
    if similar_pitches.empty or "Batter" not in similar_pitches.columns:
        return pd.DataFrame()

    usage_lookup = (
        cluster_matchups.groupby(["Batter", "dominant_pitch_type"], as_index=False)["usage_pct"]
        .sum()
        .rename(columns={"Batter": "batter_name", "dominant_pitch_type": "Pitch Type", "usage_pct": "Usage Weight"})
    )

    rows: list[dict[str, object]] = []
    for hitter in hitters:
        hitter_sample = similar_pitches[similar_pitches["Batter"] == hitter].copy()
        if hitter_sample.empty:
            continue

        pitch_types = sorted(
            hitter_sample["target_pitch_type"].dropna().astype(str).unique().tolist(),
            key=pitch_type_sort_key,
        )
        for pitch_type in pitch_types:
            sample = hitter_sample[hitter_sample["target_pitch_type"] == pitch_type].copy()
            if sample.empty:
                continue

            swing_col = sample["is_swing"] if "is_swing" in sample.columns else pd.Series(0, index=sample.index)
            in_play_col = sample["is_in_play"] if "is_in_play" in sample.columns else pd.Series(0, index=sample.index)
            is_hit_col = sample["is_hit"] if "is_hit" in sample.columns else pd.Series(0, index=sample.index)
            whiff_col = sample["is_whiff"] if "is_whiff" in sample.columns else pd.Series(0, index=sample.index)
            hard_hit_col = sample["hard_hit"] if "hard_hit" in sample.columns else pd.Series(np.nan, index=sample.index)
            woba_col = sample["estimated_woba"] if "estimated_woba" in sample.columns else pd.Series(np.nan, index=sample.index)
            angle_col = sample["Angle"] if "Angle" in sample.columns else pd.Series(np.nan, index=sample.index)
            exit_velo_col = sample["ExitSpeed"] if "ExitSpeed" in sample.columns else pd.Series(np.nan, index=sample.index)

            swings = swing_col.sum()
            in_play = in_play_col.sum()
            usage_weight = usage_lookup.loc[
                (usage_lookup["batter_name"] == hitter) & (usage_lookup["Pitch Type"] == pitch_type),
                "Usage Weight",
            ]

            rows.append(
                {
                    "Hitter": display_name(hitter),
                    "Pitch Type": pitch_type,
                    "AVG": is_hit_col.sum() / in_play if in_play else np.nan,
                    "RV/100": sample["run_value"].mean() * 100 if "run_value" in sample.columns else np.nan,
                    "Whiff%": whiff_col.sum() / swings if swings else np.nan,
                    "SwStr%": whiff_col.sum() / len(sample) if len(sample) else np.nan,
                    "HH%": hard_hit_col[in_play_col == 1].mean()
                    if "hard_hit" in sample.columns and in_play
                    else np.nan,
                    "ExitVelo": exit_velo_col.mean() if "ExitSpeed" in sample.columns else np.nan,
                    "Launch": angle_col.mean() if "Angle" in sample.columns else np.nan,
                    "GB%": (
                        ((angle_col < 10) & angle_col.notna()).sum() / in_play
                        if "Angle" in sample.columns and in_play
                        else np.nan
                    ),
                    "Usage Weight": float(usage_weight.iloc[0]) if not usage_weight.empty else np.nan,
                    "Pitches": int(len(sample)),
                    "Hit Into Play": int(in_play),
                    "wOBA": woba_col.mean() if "estimated_woba" in sample.columns else np.nan,
                }
            )

    return pd.DataFrame(rows)


def build_all_hitters_comparison_group_summary(
    similar_pitches: pd.DataFrame,
    cluster_matchups: pd.DataFrame,
    hitters: list[str],
) -> pd.DataFrame:
    if similar_pitches.empty or "Batter" not in similar_pitches.columns:
        return pd.DataFrame()

    usage_lookup = cluster_matchups.copy()
    usage_lookup["Comparison Pitch Type"] = usage_lookup["dominant_pitch_type"].map(normalize_comparison_pitch_type)
    usage_lookup = (
        usage_lookup.groupby(["Batter", "Comparison Pitch Type"], as_index=False)["usage_pct"]
        .sum()
        .rename(columns={"Batter": "batter_name", "usage_pct": "Usage Weight"})
    )

    rows: list[dict[str, object]] = []
    for hitter in hitters:
        hitter_sample = similar_pitches[similar_pitches["Batter"] == hitter].copy()
        if hitter_sample.empty:
            continue

        hitter_sample["Comparison Pitch Type"] = hitter_sample["target_pitch_type"].map(normalize_comparison_pitch_type)
        comparison_pitch_types = hitter_sample["Comparison Pitch Type"].dropna().astype(str).unique().tolist()
        comparison_pitch_types = sorted(comparison_pitch_types, key=pitch_type_sort_key)

        for pitch_type in comparison_pitch_types:
            sample = hitter_sample[hitter_sample["Comparison Pitch Type"] == pitch_type].copy()
            if sample.empty:
                continue

            swing_col = sample["is_swing"] if "is_swing" in sample.columns else pd.Series(0, index=sample.index)
            in_play_col = sample["is_in_play"] if "is_in_play" in sample.columns else pd.Series(0, index=sample.index)
            is_hit_col = sample["is_hit"] if "is_hit" in sample.columns else pd.Series(0, index=sample.index)
            whiff_col = sample["is_whiff"] if "is_whiff" in sample.columns else pd.Series(0, index=sample.index)
            hard_hit_col = sample["hard_hit"] if "hard_hit" in sample.columns else pd.Series(np.nan, index=sample.index)
            woba_col = sample["estimated_woba"] if "estimated_woba" in sample.columns else pd.Series(np.nan, index=sample.index)
            angle_col = sample["Angle"] if "Angle" in sample.columns else pd.Series(np.nan, index=sample.index)
            exit_velo_col = sample["ExitSpeed"] if "ExitSpeed" in sample.columns else pd.Series(np.nan, index=sample.index)

            swings = swing_col.sum()
            in_play = in_play_col.sum()
            usage_weight = usage_lookup.loc[
                (usage_lookup["batter_name"] == hitter) & (usage_lookup["Comparison Pitch Type"] == pitch_type),
                "Usage Weight",
            ]

            rows.append(
                {
                    "Hitter": display_name(hitter),
                    "Pitch Type": pitch_type,
                    "AVG": is_hit_col.sum() / in_play if in_play else np.nan,
                    "RV/100": sample["run_value"].mean() * 100 if "run_value" in sample.columns else np.nan,
                    "Whiff%": whiff_col.sum() / swings if swings else np.nan,
                    "SwStr%": whiff_col.sum() / len(sample) if len(sample) else np.nan,
                    "HH%": hard_hit_col[in_play_col == 1].mean()
                    if "hard_hit" in sample.columns and in_play
                    else np.nan,
                    "ExitVelo": exit_velo_col.mean() if "ExitSpeed" in sample.columns else np.nan,
                    "Launch": angle_col.mean() if "Angle" in sample.columns else np.nan,
                    "GB%": (
                        ((angle_col < 10) & angle_col.notna()).sum() / in_play
                        if "Angle" in sample.columns and in_play
                        else np.nan
                    ),
                    "UsageWeight": float(usage_weight.iloc[0]) if not usage_weight.empty else np.nan,
                    "Pitches": int(len(sample)),
                    "hit_into_play": int(in_play),
                    "wOBA": woba_col.mean() if "estimated_woba" in sample.columns else np.nan,
                }
            )

    return pd.DataFrame(rows)

default_hitters = derive_lineup(data, selected_opponent_team)
if not default_hitters:
    st.error("Could not build an opponent lineup from the current data.")
    st.stop()

team_hitter_options = (
    data.loc[data["BatterTeam"] == selected_opponent_team, "Batter"]
    .dropna()
    .drop_duplicates()
    .tolist()
)
team_hitter_options = sorted(
    team_hitter_options,
    key=last_name_sort_key,
)
hitter_display_map = {hitter: format_person_name(hitter) for hitter in team_hitter_options}
hitter_display_map = {hitter: display_name(hitter) for hitter in team_hitter_options}
default_hitter_lookup = {
    normalize_person_name(display_name(hitter)): hitter
    for hitter in team_hitter_options
}
configured_default_hitters = [
    default_hitter_lookup[normalize_person_name(name)]
    for name in DEFAULT_HITTER_NAMES
    if normalize_person_name(name) in default_hitter_lookup
]
if selected_opponent_team in DEFAULT_HITTER_ID_HINTS:
    hinted_hitters = [
        hitter
        for hitter in DEFAULT_HITTER_ID_HINTS[selected_opponent_team]
        if hitter in team_hitter_options and hitter not in configured_default_hitters
    ]
    configured_default_hitters.extend(hinted_hitters)
if configured_default_hitters:
    default_hitters = configured_default_hitters[:9]
st.caption(
    "Hitters auto-fill with the top 9 by plate appearances for the selected opposing team. "
    "You can add or remove hitters from that team before running the analysis."
)
selected_hitters = st.multiselect(
    "Hitters",
    options=team_hitter_options,
    default=default_hitters,
    format_func=lambda value: hitter_display_map.get(value, value),
    help="Only hitters from the selected opposing team appear here.",
)

if not selected_hitters:
    st.error("Select at least one hitter from the opposing team.")
    st.stop()

run_clicked = st.button("Run Complete MAC Analysis", type="primary")
current_run_config = {
    "selected_pitcher": selected_pitcher,
    "selected_opponent_team": selected_opponent_team,
    "selected_hitters": tuple(selected_hitters),
}

if run_clicked:
    analysis = run_mac(
        data,
        pitcher_name=selected_pitcher,
        hitters=selected_hitters,
        similarity_threshold=SIMILARITY_THRESHOLD,
        min_similar_pitches=MIN_COMP_PITCHES,
        max_clusters=MAX_PITCH_CLUSTERS,
    )

    st.session_state["mac_last_run"] = {
        "config": current_run_config,
        "analysis": analysis,
    }

last_run = st.session_state.get("mac_last_run")
if last_run is None:
    st.info("Set the pitcher, opposing team, and hitters, then click Run Complete MAC Analysis.")
    st.stop()

last_config = last_run["config"]
analysis = last_run["analysis"]
analysis.skipped_files = skipped_files
selected_pitcher = last_config["selected_pitcher"]
selected_opponent_team = last_config["selected_opponent_team"]
selected_hitters = list(last_config["selected_hitters"])

if current_run_config != last_config:
    st.caption("Selections changed. Click Run Complete MAC Analysis to refresh the analysis with the new inputs.")

metric_columns = st.columns(3)
metric_columns[0].metric("Current Pitcher", format_person_name(selected_pitcher))
metric_columns[1].metric("Opposing Team", selected_opponent_team)
metric_columns[2].metric("Lineup size", f"{len(selected_hitters)}")

st.subheader("Current Pitcher Pitch Mix")
pitch_mix = analysis.pitcher_clusters.copy()
pitch_mix_display = pitch_mix.rename(
    columns={
        "pitch_count": "Pitches",
        "usage_pct": "Usage",
        "dominant_pitch_type": "Pitch Type",
        "speed": "Velo",
        "ivb": "IVB",
        "hb": "HB",
        "whiff_rate": "Whiff",
        "hard_hit_rate": "HardHit",
        "run_value": "RV",
    }
)
pitch_mix_columns = [
    column
    for column in [
        "Pitch Type",
        "Usage",
        "Pitches",
        "Velo",
        "IVB",
        "HB",
        "Whiff",
        "HardHit",
        "RV",
    ]
    if column in pitch_mix_display.columns
]
st.dataframe(
    pitch_mix_display[pitch_mix_columns].style.format(
        {
            "Usage": "{:.1%}",
            "Velo": "{:.1f}",
            "IVB": "{:.1f}",
            "HB": "{:.1f}",
            "Whiff": "{:.1%}",
            "HardHit": "{:.1%}",
            "RV": "{:.3f}",
        }
    ),
    width="stretch",
    hide_index=True,
)

usage_chart = (
    alt.Chart(pitch_mix)
    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
    .encode(
        x=alt.X(
            "dominant_pitch_type:N",
            sort=alt.SortField(field="usage_pct", order="descending"),
            title="Pitch type",
            axis=alt.Axis(labelAngle=-45, labelAlign="right", labelLimit=160),
        ),
        y=alt.Y("usage_pct:Q", title="Usage"),
        color=alt.Color(
            "dominant_pitch_type:N",
            title="Pitch type",
            scale=build_pitch_type_color_scale(
                pitch_mix["dominant_pitch_type"].dropna().astype(str).tolist()
            ),
        ),
        tooltip=[
            alt.Tooltip("dominant_pitch_type:N", title="Pitch type"),
            alt.Tooltip("usage_pct:Q", title="Usage", format=".1%"),
            alt.Tooltip("speed:Q", title="Velo", format=".1f"),
            alt.Tooltip("ivb:Q", title="IVB", format=".1f"),
            alt.Tooltip("hb:Q", title="HB", format=".1f"),
        ],
    )
)
st.altair_chart(usage_chart, use_container_width=True)

distance_samples = getattr(analysis, "distance_samples", pd.DataFrame())
if not distance_samples.empty and "distance" in distance_samples.columns:
    st.subheader("Euclidean Distance Distribution")
    qualifying_count = int(distance_samples["qualifies"].sum()) if "qualifies" in distance_samples.columns else 0
    total_count = int(len(distance_samples))
    metric_cols = st.columns(3)
    metric_cols[0].metric("Total Pitch Comparisons", f"{total_count:,}")
    metric_cols[1].metric("Within Threshold", f"{qualifying_count:,}")
    metric_cols[2].metric(
        "Qualifying Rate",
        f"{(qualifying_count / total_count):.1%}" if total_count else "0.0%",
    )
    st.altair_chart(
        build_distance_distribution_plot(
            distance_samples,
            similarity_threshold=SIMILARITY_THRESHOLD,
        ),
        use_container_width=True,
    )

if not analysis.similar_pitches.empty:
    st.subheader("Similar Pitch Movement")
    st.altair_chart(
        build_similar_pitch_movement_plot(analysis.similar_pitches),
        use_container_width=False,
    )

st.subheader(f"Expected Matchup RV/100 + Hitter Summary: {format_person_name(selected_pitcher)}")
leaderboard = analysis.matchup_summary.copy()
leaderboard["BatterDisplay"] = leaderboard["Batter"].map(display_name)
leaderboard_display = leaderboard.rename(
    columns={
        "BatterDisplay": "Hitter",
        "MAC_Score": "MAC",
        "Expected_Whiff_Rate": "Whiff",
        "Expected_Hard_Hit_Rate": "HardHit",
        "Expected_EV": "EV",
        "Pitch_Clusters_Covered": "Clusters",
        "Total_Similar_Pitches": "CompPitches",
        "Weighted_Coverage": "Coverage",
        "Pitcher_Friendly": "Pitcher Edge",
    }
)
plot = build_summary_plot(analysis.cluster_matchups, leaderboard, analysis.similar_pitches)
st.altair_chart(plot, use_container_width=True)

leaderboard_for_display = leaderboard_display.drop(columns=["Batter"]).copy()
ordered_columns = [
    "Hitter",
    "Pitcher Edge",
]
leaderboard_for_display = leaderboard_for_display[ordered_columns]
leaderboard_for_display["RV/100"] = leaderboard["MAC_Score"] * 100

lineup_summary = build_lineup_stat_summary(analysis.similar_pitches, selected_hitters)
lineup_summary = lineup_summary.drop(columns=["RV/100"], errors="ignore")
combined_summary = leaderboard_for_display.merge(lineup_summary, on="Hitter", how="outer")
combined_summary = combined_summary[
    [
        "Hitter",
        "Pitcher Edge",
        "RV/100",
        "AVG",
        "Whiff%",
        "SwStr%",
        "HH%",
        "ExitVelo",
        "Launch",
        "GB%",
        "wOBA",
        "Pitches",
        "Hit Into Play",
    ]
]
st.subheader("Lineup Summary")
st.dataframe(
    combined_summary.style.format(
        {
            "AVG": "{:.3f}",
            "RV/100": "{:.2f}",
            "Whiff%": "{:.1%}",
            "SwStr%": "{:.1%}",
            "ExitVelo": "{:.1f}",
            "Launch": "{:.1f}",
            "HH%": "{:.1%}",
            "GB%": "{:.1%}",
            "wOBA": "{:.3f}",
        }
    ),
    width="stretch",
    hide_index=True,
)

all_hitters_pitch_type_summary = build_all_hitters_pitch_type_summary(
    analysis.similar_pitches,
    analysis.cluster_matchups,
    selected_hitters,
)
if not all_hitters_pitch_type_summary.empty:
    lineup_by_pitch_type_columns = [
        column
        for column in [
            "Hitter",
            "Pitch Type",
            "RV/100",
            "AVG",
            "Whiff%",
            "SwStr%",
            "HH%",
            "ExitVelo",
            "Launch",
            "GB%",
            "wOBA",
            "Pitches",
            "Hit Into Play",
        ]
        if column in all_hitters_pitch_type_summary.columns
    ]
    st.subheader("Lineup By Pitch Type")
    st.dataframe(
        all_hitters_pitch_type_summary[lineup_by_pitch_type_columns].style.format(
            {
                "AVG": "{:.3f}",
                "RV/100": "{:.2f}",
                "Whiff%": "{:.1%}",
                "SwStr%": "{:.1%}",
                "HH%": "{:.1%}",
                "ExitVelo": "{:.1f}",
                "Launch": "{:.1f}",
                "GB%": "{:.1%}",
                "wOBA": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

comparison_group_summary = build_all_hitters_comparison_group_summary(
    analysis.similar_pitches,
    analysis.cluster_matchups,
    selected_hitters,
)
if not comparison_group_summary.empty:
    comparison_columns = [
        column
        for column in [
            "Hitter",
            "Pitch Type",
            "AVG",
            "RV/100",
            "Whiff%",
            "SwStr%",
            "HH%",
            "ExitVelo",
            "Launch",
            "GB%",
            "UsageWeight",
            "Pitches",
            "hit_into_play",
            "wOBA",
        ]
        if column in comparison_group_summary.columns
    ]
    st.subheader("Original-Style Comparison View")
    st.dataframe(
        comparison_group_summary[comparison_columns].style.format(
            {
                "AVG": "{:.3f}",
                "RV/100": "{:.2f}",
                "Whiff%": "{:.1%}",
                "SwStr%": "{:.1%}",
                "HH%": "{:.1%}",
                "ExitVelo": "{:.1f}",
                "Launch": "{:.1f}",
                "GB%": "{:.1%}",
                "UsageWeight": "{:.2f}",
                "wOBA": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

st.subheader("Detailed Hitter View")
selected_hitter = st.selectbox(
    "Choose Hitter",
    options=leaderboard["Batter"].tolist(),
    format_func=lambda value: hitter_display_map.get(value, display_name(value)),
    key="detailed_hitter_view",
)

hitter_cluster_view = (
    analysis.cluster_matchups.loc[analysis.cluster_matchups["Batter"] == selected_hitter]
    .copy()
)
hitter_cluster_view = hitter_cluster_view.sort_values(
    ["usage_pct", "dominant_pitch_type"], ascending=[False, True]
)
hitter_cluster_display_source = build_current_pitcher_detail(hitter_cluster_view)

st.subheader(f"Current Pitcher Detail: {display_name(selected_hitter)}")
hitter_cluster_display = hitter_cluster_display_source.rename(
    columns={
        "dominant_pitch_type": "Pitch Type",
        "usage_pct": "Usage",
        "expected_run_value": "ExpectedRV",
        "expected_whiff_rate": "Whiff",
        "expected_hard_hit_rate": "HardHit",
        "expected_exit_speed": "EV",
    }
)
pitcher_detail_columns = [
    column
    for column in ["Pitch Type", "Usage", "ExpectedRV", "Whiff", "HardHit", "EV"]
    if column in hitter_cluster_display.columns
]
st.dataframe(
    hitter_cluster_display[pitcher_detail_columns].style.format(
        {
            "Usage": "{:.1%}",
            "ExpectedRV": "{:.2f}",
            "Whiff": "{:.1%}",
            "HardHit": "{:.1%}",
            "EV": "{:.1f}",
        }
    ),
    width="stretch",
    hide_index=True,
)

hitter_samples = pd.DataFrame()
if not analysis.similar_pitches.empty and "Batter" in analysis.similar_pitches.columns:
    hitter_samples = analysis.similar_pitches.loc[
        analysis.similar_pitches["Batter"] == selected_hitter
    ].copy()

if skipped_files:
    with st.expander("Skipped Files / Schema Issues"):
        st.dataframe(pd.DataFrame(skipped_files), width="stretch")
