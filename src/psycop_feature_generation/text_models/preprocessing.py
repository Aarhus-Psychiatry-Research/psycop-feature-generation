import re
from typing import Literal, Optional

import pandas as pd
from psycop_feature_generation.loaders.raw.load_text import (
    get_valid_text_sfi_names,
    load_text_split,
)
from psycop_feature_generation.text_models.utils import stop_words
from psycop_ml_utils.sql.writer import write_df_to_sql


def text_preprocessing(
    df: pd.DataFrame,
    text_column_name: str = "value",
) -> pd.DataFrame:
    """Preprocess texts by lower casing, removing stopwords and symbols.

    Args:
        df (pd.DataFrame): Dataframe with a column containing text to clean.
        text_column_name (str): Name of column containing text. Defaults to "value".

    Returns:
        pd.DataFrame: _description_
    """
    regex_symbol_removal_and_stop_words = re.compile(
        r"[^ÆØÅæøåA-Za-z0-9 ]+|\b%s\b" % r"\b|\b".join(map(re.escape, stop_words)),
    )

    df[text_column_name] = (
        df[text_column_name]
        .str.lower()
        .replace(regex_symbol_removal_and_stop_words, value="", regex=True)
    )

    return df


def text_preprocessing_pipeline(
    split_name: list[Literal["train", "val"]] = ["train", "val"],
    n_rows: Optional[int] = None,
) -> None:
    """Pipeline for preprocessing all sfis from given splits. Filtering of which sfis to include in features happens in the loader."""

    # Load text from splits
    df = load_text_split(
        text_sfi_names=get_valid_text_sfi_names(),
        split_name=split_name,
        include_sfi_name=True,
        n_rows=n_rows,
    )

    # preprocess
    df = text_preprocessing(df)

    # save to sql
    split_name = "_".join(split_name)  # type: ignore

    write_df_to_sql(
        df,
        table_name=f"psycop_{split_name}_all_sfis_all_years_preprocess",
        if_exists="replace",
        rows_per_chunk=5000,
    )
