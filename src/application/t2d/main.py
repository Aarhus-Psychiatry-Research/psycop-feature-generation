"""Main example on how to generate features.

Uses T2D-features. WIP, will be migrated to psycop-t2d when reaching
maturity.
"""

import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Union

import wandb

import psycop_feature_generation.loaders.raw  # noqa
from application.t2d.modules.flatten_dataset import create_flattened_dataset
from application.t2d.modules.flattened_dataset_description import (
    save_feature_set_description_to_disk,
)
from application.t2d.modules.save_dataset_to_disk import split_and_save_dataset_to_disk
from application.t2d.modules.specify_features import get_spec_set
from psycop_feature_generation.loaders.raw.load_demographic import birthdays
from psycop_feature_generation.loaders.raw.load_visits import (
    physical_visits_to_psychiatry,
)
from psycop_feature_generation.timeseriesflattener.feature_spec_objects import (
    BaseModel,
    PredictorSpec,
)
from psycop_feature_generation.utils import FEATURE_SETS_PATH, RELATIVE_PROJECT_ROOT


def init_wandb(
    wandb_project_name: str,
    predictor_specs: Sequence[PredictorSpec],
    save_dir: Union[Path, str],
) -> None:
    """Initialise wandb logging. Allows to use wandb to track progress, send
    Slack notifications if failing, and track logs.

    Args:
        wandb_project_name (str): Name of wandb project.
        predictor_specs (Iterable[dict[str, Any]]): List of predictor specs.
        save_dir (Union[Path, str]): Path to save dir.
    """

    feature_settings = {
        "save_path": save_dir,
        "predictor_list": [spec.__dict__ for spec in predictor_specs],
    }

    # on Overtaci, the wandb tmp directory is not automatically created,
    # so we create it here.
    # create debug-cli.one folders in /tmp and project dir
    if sys.platform == "win32":
        (Path(tempfile.gettempdir()) / "debug-cli.onerm").mkdir(
            exist_ok=True,
            parents=True,
        )
        (RELATIVE_PROJECT_ROOT / "wandb" / "debug-cli.onerm").mkdir(
            exist_ok=True, parents=True
        )

    wandb.init(project=wandb_project_name, config=feature_settings)


def create_feature_set_path(
    proj_path: Path,
    feature_set_id: str,
) -> Path:
    """Create save directory.

    Args:
        proj_path (Path): Path to project.
        feature_set_id (str): Feature set id.

    Returns:
        Path: Path to sub directory.
    """

    # Split and save to disk
    # Create directory to store all files related to this run
    save_dir = proj_path / "feature_sets" / feature_set_id

    if not save_dir.exists():
        save_dir.mkdir()

    return save_dir


def get_project_info(
    n_predictors: int,
    project_name: str,
) -> ProjectInfo:
    """Setup for main.

    Args:
        n_predictors (int): Number of predictors.
        feature_sets_path (Path): Path to feature sets.
        project_name (str): Name of project.
    Returns:
        tuple[Path, str]: Tuple of project path, and feature_set_id
    """
    proj_path = SHARED_RESOURCES_PATH / project_name

    current_user = Path().home().name
    feature_set_id = f"psycop_{project_name}_{current_user}_{n_predictors}_features_{time.strftime('%Y_%m_%d_%H_%M')}"

    save_dir = create_feature_set_path(
        feature_set_id=feature_set_id,
        proj_path=proj_path,
    )

    return ProjectInfo(
        project_path=proj_path,
        feature_set_path=save_dir,
        feature_set_id=feature_set_id,
        project_name=project_name,
    )


class ProjectInfo(BaseModel):
    project_name: str
    project_path: Path
    feature_set_path: Path
    feature_set_id: str

    def __init__():
        super().__init__()

        # Iterate over each attribute. If the attribute is a Path, create it if it does not exist.
        for attr in self.__dict__:
            if isinstance(attr, Path):
                attr.mkdir(exist_ok=True)


def main(
    proj_name: str,
    feature_sets_path: Path,
):
    """Main function for loading, generating and evaluating a flattened
    dataset.

    Args:
        proj_name (str): Name of project.
        feature_sets_path (Path): Path to where feature sets should be stored.
    """
    feature_specs = get_spec_set()

    project_info = get_project_info(
        n_predictors=len(feature_specs.temporal_predictors),
        proj_name=proj_name,
    )

    init_wandb(
        wandb_project_name=project_info.project_name,
        predictor_specs=feature_specs.temporal_predictors,
        save_dir=project_info.feature_set_path,  # Save-dir as argument because we want to log the path
    )

    flattened_df = create_flattened_dataset(
        prediction_times=physical_visits_to_psychiatry(),
        spec_set=feature_specs,
        proj_path=proj_path,
        birthdays=birthdays(),
    )

    split_and_save_dataset_to_disk(
        flattened_df=flattened_df,
        out_dir=save_dir,
        file_prefix=feature_set_id,
        file_suffix="parquet",
    )

    save_feature_set_description_to_disk(
        predictor_specs=feature_specs.temporal_predictors
        + feature_specs.static_predictors,
        flattened_dataset_file_dir=save_dir,
        out_dir=save_dir,
        file_suffix="parquet",
    )

    wandb.log_artifact("poetry.lock", name="poetry_lock_file", type="poetry_lock")


if __name__ == "__main__":
    main(
        feature_sets_path=FEATURE_SETS_PATH,
        proj_name="t2d",
    )
