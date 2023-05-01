import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from psycop_feature_generation.loaders.raw.load_text import (
    get_valid_text_sfi_names,
    load_text_split,
)
from psycop_feature_generation.text_models.utils import stop_words
from psycop_feature_generation.utils import write_df_to_file

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
    # define regex for stop words by joining them with word boundary
    regex_stop_words = r"\b%s" % r"\b|\b".join(stop_words)

    # define regex that removes symbols (by keeping everything else)
    regex_symbol_removal = r"[^ÆØÅæøåA-Za-z0-9 ]+"

    # combine
    regex_symbol_removal_and_stop_words = re.compile(
        f"{regex_stop_words}|{regex_symbol_removal}",
    )

    # lower case and remove stop words and symbols
    df[text_column_name] = (
        df[text_column_name]
        .str.lower()
        .replace(regex_symbol_removal_and_stop_words, value="", regex=True)  # type: ignore
    )

    return df


def text_preprocessing_pipeline(
    split_names: Sequence[Literal["train", "val"]] = ["train", "val"],
    n_rows: Optional[int] = None,
) -> None:
    """Pipeline for preprocessing all sfis from given splits. Filtering of which sfis to include in features happens in the loader."""

    # Load text from splits
    df = load_text_split(
        text_sfi_names=get_valid_text_sfi_names(),
        split_name=split_names,
        include_sfi_name=True,
        n_rows=n_rows,
    )

    # preprocess
    df = text_preprocessing(df)

    # save to parquet
    split_names = "_".join(split_names)  # type: ignore

    write_df_to_file(
        df=df,
        file_path=Path(
            f"E:/shared_resources/preprocessed_text/psycop_{split_names}_all_sfis_preprocessed.parquet",
        ),
    )
