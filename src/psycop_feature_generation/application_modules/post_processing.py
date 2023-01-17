"""Post processing modules for psycop_feature_generation.

This includes modules for:
1. Removing zero-variance features
2. Removing features with high missingness
3. Removing predictions times outside of a specified age boundary
"""
import logging
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd
from psycop_feature_generation.application_modules.project_setup import ProjectInfo

from timeseriesflattener.feature_spec_objects import BaseModel

log = logging.getLogger(__name__)


class PostProcessingArguments(BaseModel):
    """A class for holding the thresholds for post processing."""

    missingness_threshold: float = 0.9
    age_column_name: str = "pred_age_in_years"
    age_boundary: tuple


class PostProcess:
    """Base class for post processing modules."""

    @staticmethod
    def _identify_zero_variance_features(
        df: pd.DataFrame,
        thresholds: PostProcessingArguments,  # pylint: disable=unused-argument
    ):
        """Identifies zero variance features. Thresholds args is unused, but kept
        for compatibility with other processors."""
        stds = df.std(numeric_only=True)
        features_to_drop = stds[stds == 0].index.tolist()
        log.debug(f"Zero variance features: {features_to_drop}")
        return features_to_drop

    @staticmethod
    def _identify_high_missingness_features(
        df: pd.DataFrame,
        thresholds: PostProcessingArguments,
    ):
        """Identifies features with high missingness."""
        missingness = df.isna().mean()
        features_to_drop = missingness[
            missingness > thresholds.missingness_threshold
        ].index.tolist()
        log.debug(f"High missingness features: {features_to_drop}")
        return features_to_drop
        

    @staticmethod
    def _remove_outside_age_boundary(
        df: pd.DataFrame,
        thresholds: PostProcessingArguments,
    ):
        """Removes rows outside of the age boundary."""
        n_rows_before = df.shape[0]
        df = df.loc[
            (df[thresholds.age_column_name] >= thresholds.age_boundary[0])
            & (df[thresholds.age_column_name] <= thresholds.age_boundary[1]),
            :,
        ]
        n_rows_after = df.shape[0]
        log.info(
            f"Removed {n_rows_before - n_rows_after} ({round(n_rows_after / n_rows_before * 100, 2)}%) rows outside of age boundary."
        )
        log.debug(f"{n_rows_before} rows before, {n_rows_after} rows after.")
        return df

    def __init__(self, project_info: ProjectInfo):
        self.project_info = project_info
        self.features_to_drop = []
        self.processors = {
            "zero_variance": self._identify_zero_variance_features,
            "missingness": self._identify_high_missingness_features,
            "age_boundary": self._remove_outside_age_boundary,
        }

    def _load_predictors_from_split(self, path: Path) -> pd.DataFrame:
        """Loads the predictors from a given path."""
        df = pd.read_parquet(path)
        # only keep predictors
        df = df.loc[:, df.columns.str.startswith(self.project_info.prefix.predictor + "_")]
        return df

    def _load_training_data(self) -> pd.DataFrame:
        """Loads the predictors from the training data."""
        train_path = list(self.project_info.feature_set_path.glob("*train*"))[0]
        return self._load_predictors_from_split(train_path)

    def process(
        self,
        thresholds: PostProcessingArguments,
        processors: Iterable[
            Literal["zero_variance", "missingness", "age_boundary"]
        ] = ("zero_variance", "missingness", "age_boundary"),
    ) -> None:
        """Postprocesses the dataframes with the given processors. Features to
        drop are estimated from the training data and afterwards applied to all
        splits. Processed splits are saved to the same location as the
        original.

        Args:
            thresholds (PostProcessingArguments, optional): The thresholds for the processors.
            processors (Literal["zero_variance", "missingness", "age_boundary"], optional):
                The processors to apply. Defaults to ("zero_variance", "missingness", "age_boundary").
        """
        log.info("–––––––– Post processing ––––––––")
        log.info(f"Applying processors: {processors} with thresholds: {thresholds}")
        train_df = self._load_training_data()

        # identify features to remove based on train set
        self.identify_features_to_drop(
            processors=processors,
            thresholds=thresholds,
            train_df=train_df,
        )

        log.info(
            f"Dropping {len(self.features_to_drop)} features: {self.features_to_drop}"
            + " from all splits",
        )

        # remove features from all splits
        self.process_splits(processors, thresholds)

    def identify_features_to_drop(
        self,
        processors: Iterable[
            Literal["zero_variance", "missingness", "age_boundary"]
        ],
        thresholds: PostProcessingArguments,
        train_df: pd.DataFrame,
    ) -> None:
        """Identifies features to drop based on the on the specified processors
        and thresholds. Each processors adds column names to the
        features_to_drop attribute.

        Args:
            processors (Literal["zero_variance", "missingness", "age_boundary"]):
                The processors to apply.
            thresholds (PostProcessingArguments): The thresholds for the processors.
            train_df (pd.DataFrame): The training data.
        """
        for processor in processors:
            if processor == "age_boundary":
                continue
            self.features_to_drop.extend(self.processors[processor](train_df, thresholds))

    def process_splits(
        self,
        processors: Iterable[
            Literal["zero_variance", "missingness", "age_boundary"]
        ],
        thresholds: PostProcessingArguments,
    ) -> None:
        """Processes the splits by removing the features_to_drop from each
        split. If "age_boundary" is in processors, the age boundary is also
        applied. Splits are saved to the same path as the original splits.

        Args:
            processors (Literal["zero_variance", "missingness", "age_boundary"]):
                The processors applied to the document. Only used for "age_boundary".
            thresholds (PostProcessingArguments): The thresholds for the age boundary.
        """
        for split in self.project_info.feature_set_path.glob("*.parquet"):
            log.debug(f"Processing split {split}")
            split_df = self._load_predictors_from_split(split)
            split_df = split_df.drop(columns=self.features_to_drop)
            if "age_boundary" in processors:
                split_df = self.processors["age_boundary"](split_df, thresholds)
            split_df.to_parquet(split)


def post_process_splits(
    project_info: ProjectInfo,
    thresholds: PostProcessingArguments,
    processors: Iterable[
        Literal["zero_variance", "missingness", "age_boundary"]
    ],
) -> None:
    """Apply postprocessing to all splits.

    Args:
        project_info (ProjectInfo): The project info.
        thresholds (PostProcessingArguments): The thresholds for the processors.
        processors (Iterable[Literal["zero_variance", "missingness", "age_boundary"]]): The processors to apply.
    """
    post_processor = PostProcess(project_info)
    post_processor.process(processors=processors, thresholds=thresholds)
