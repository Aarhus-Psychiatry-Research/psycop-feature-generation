"""Generates a df with feature descriptions for the predictors in the source
df."""
from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from psycop_feature_generation.data_checks.utils import save_df_to_pretty_html_table
from psycop_feature_generation.loaders.flattened.local_feature_loaders import (
    load_split,
)
from timeseriesflattener.feature_spec_objects import (
    PredictorSpec,
    StaticSpec,
    TemporalSpec,
    TextPredictorSpec,
)
from wasabi import Printer

UNICODE_HIST = {
    0: " ",
    1 / 8: "▁",
    1 / 4: "▂",
    3 / 8: "▃",
    1 / 2: "▄",
    5 / 8: "▅",
    3 / 4: "▆",
    7 / 8: "▇",
    1: "█",
}

HIST_BINS = 8


def get_value_proportion(series, value):
    """Get proportion of series that is equal to the value argument."""
    if np.isnan(value):
        return round(series.isna().mean(), 2)
    return round(series.eq(value).mean(), 2)


def _find_nearest(array, value):
    """Find the nearest numerical match to value in an array.

    Args:
        array (np.ndarray): An array of numbers to match with.
        value (float): Single value to find an entry in array that is close.

    Returns:
        np.array: The entry in array that is closest to value.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def create_unicode_hist(series: pd.Series) -> pd.Series:
    """Return a histogram rendered in block unicode. Given a pandas series of
    numerical values, returns a series with one entry, the original series
    name, and a histogram made up of unicode characters.

    Args:
        series (pd.Series): Numeric column of data frame for analysis

    Returns:
        pd.Series: Index of series name and entry with unicode histogram as
        a string, eg '▃▅█'

    All credit goes to the python package skimpy.
    """
    # Remove any NaNs
    series = series.dropna()

    if series.dtype == "bool":
        series = series.astype("int")

    hist, _ = np.histogram(series, density=True, bins=HIST_BINS)
    hist = hist / hist.max()

    # Now do value counts
    key_vector = np.array(list(UNICODE_HIST.keys()), dtype="float")

    ucode_to_print = "".join(
        [UNICODE_HIST[_find_nearest(key_vector, val)] for val in hist],
    )

    return pd.Series(ucode_to_print)


def generate_temporal_feature_description(
    series: pd.Series,
    predictor_spec: TemporalSpec,
    feature_name: str | None = None,
):
    """Generate a row with feature description for a temporal predictor."""
    if feature_name is not None:
        feature_name = feature_name
    else:
        feature_name = predictor_spec.feature_name

    d = {
        "Predictor df": feature_name,
        "Lookbehind days": predictor_spec.interval_days,
        "Resolve multiple": predictor_spec.resolve_multiple_fn.__name__,  # type: ignore
        "N unique": series.nunique(),
        "Fallback strategy": str(predictor_spec.fallback),
        "Proportion missing": series.isna().mean(),
        "Mean": round(series.mean(), 2),
        "Histogram": create_unicode_hist(series),
        "Proportion using fallback": get_value_proportion(
            series,
            predictor_spec.fallback,
        ),
    }

    for percentile in (0.01, 0.25, 0.5, 0.75, 0.99):
        # Get the value representing the percentile
        d[f"{percentile * 100}-percentile"] = round(series.quantile(percentile), 1)

    return d


def generate_static_feature_description(series: pd.Series, predictor_spec: StaticSpec):
    """Generate a row with feature description for a static predictor."""
    return {
        "Predictor df": predictor_spec.feature_name,
        "Lookbehind days": "N/A",
        "Resolve multiple": "N/A",
        "N unique": series.nunique(),
        "Fallback strategy": "N/A",
        "Proportion missing": series.isna().mean(),
        "Mean": round(series.mean(), 2),
        "Histogram": create_unicode_hist(series),
        "Proportion using fallback": "N/A",
    }


def generate_feature_description_row(
    series: pd.Series,
    predictor_spec: StaticSpec | TemporalSpec,
    feature_name: str | None = None,
) -> dict:
    """Generate a row with feature description.

    Args:
        series (pd.Series): Series with data to describe.
        predictor_spec (PredictorSpec): Predictor specification.
        feature_name (str, optional): Name of the feature. Defaults to None.

    Returns:
        dict: dictionary with feature description.
    """

    if isinstance(predictor_spec, StaticSpec):
        return generate_static_feature_description(series, predictor_spec)
    if isinstance(predictor_spec, TemporalSpec):
        return generate_temporal_feature_description(
            series,
            predictor_spec,
            feature_name=feature_name,
        )
    raise ValueError(f"Unknown predictor spec type: {type(predictor_spec)}")


def generate_feature_description_df(
    df: pd.DataFrame,
    predictor_specs: list[PredictorSpec | StaticSpec | TemporalSpec],
) -> pd.DataFrame:
    """Generate a data frame with feature descriptions.

    Args:
        df (pd.DataFrame): Data frame with data to describe.
        predictor_specs (Union[PredictorSpec, StaticSpec, TemporalSpec]): Predictor specifications.

    Returns:
        pd.DataFrame: Data frame with feature descriptions.
    """

    rows = []

    for spec in predictor_specs:
        column_name = spec.get_col_str()

        if isinstance(spec, TextPredictorSpec):
            last_part = column_name.split(f"{spec.prefix}_{spec.feature_name}")[1]
            first_part = column_name.split(last_part)[0]
            string_match = f"{first_part}[\\dA-Za-z\\-]+{last_part}"

            column_names = [
                re.match(string_match, column)[0]  # type: ignore
                for column in df.columns
                if re.match(string_match, column) is not None
            ]

            for column_name in column_names:
                rows.append(
                    generate_feature_description_row(
                        series=df[column_name],
                        predictor_spec=spec,
                        feature_name=column_name,
                    ),
                )

        else:
            rows.append(
                generate_feature_description_row(
                    series=df[column_name],
                    predictor_spec=spec,
                ),
            )

    # Convert to dataframe
    feature_description_df = pd.DataFrame(rows)

    # Sort feature_description_df by Predictor df to make outputs easier to read
    feature_description_df = feature_description_df.sort_values(by="Predictor df")

    return feature_description_df


def save_feature_descriptive_stats_from_dir(
    feature_set_dir: Path,
    feature_specs: list[TemporalSpec | StaticSpec],
    file_suffix: str,
    splits: Sequence[str] = ("train",),
    out_dir: Path | None = None,
):
    """Write a html table and csv with descriptive stats for features in the directory.

    Args:
        feature_set_dir (Path): Path to directory with data frames.
        feature_specs (list[PredictorSpec]): List of feature specifications.
        file_suffix (str): Suffix of the data frames to load. Must be either ".csv" or ".parquet".
        splits (tuple[str]): tuple of splits to include in the description. Defaults to ("train").
        out_dir (Path): Path to directory where to save the feature description. Defaults to None.
    """
    msg = Printer(timestamp=True)

    if out_dir is None:
        out_dir = feature_set_dir / "feature_set_descriptive_stats"

    out_dir.mkdir(exist_ok=True, parents=True)

    for split in splits:
        msg.info(f"{split}: Creating descriptive stats for feature set")

        dataset = load_split(
            feature_set_dir=feature_set_dir,
            split=split,
            file_suffix=file_suffix,
        )

        msg.info(f"{split}: Generating descriptive stats dataframe")

        feature_descriptive_stats = generate_feature_description_df(
            df=dataset,
            predictor_specs=feature_specs,
        )

        msg.info(f"{split}: Writing descriptive stats dataframe to disk")

        feature_descriptive_stats.to_csv(
            out_dir / f"{split}_feature_descriptive_stats.csv",
            index=False,
        )
        # Writing html table as well
        save_df_to_pretty_html_table(
            df=feature_descriptive_stats,
            path=out_dir / f"{split}_feature_descriptive_stats.html",
            title="Feature descriptive stats",
        )
