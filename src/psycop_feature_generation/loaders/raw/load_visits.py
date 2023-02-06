"""Loaders for visits to psychiatry."""

import logging
from typing import Optional, Literal

import pandas as pd

from psycop_feature_generation.loaders.raw.sql_load import sql_load
from psycop_feature_generation.utils import data_loaders
from timeseriesflattener.feature_spec_objects import BaseModel

log = logging.getLogger(__name__)


class RawValueSourceSchema(BaseModel):
    """A class for structuring raw values from source schemas.

    Fields:
        view (str): The name of the view for the schema.
        datetime_col (str): The name of the datetime column in the view. This should be defined as the end time of the visit, in order to prevent data leakage.
        value_col (Optional[str]): The name of the column used to calculate the value for the loader. To calculate length of visit, this needs to be the name of the column indicating the starting time of the visit. Defaults to None.
        location_col (Optional[str]): The name of the column indicating which shak unit was responsible for the visit (e.g. 6600). Defaults to None.
        where (str): A where clause that defines additional subsetting for views.
    """

    view: str
    datetime_col: str
    value_col: Optional[str] = None
    location_col: Optional[str] = None
    where: str


@data_loaders.register("physical_visits")
def physical_visits(
    shak_code: Optional[int] = None,
    shak_sql_operator: Optional[str] = "=",
    where_clause: Optional[str] = None,
    where_separator: Optional[str] = "AND",
    n_rows: Optional[int] = None,
    return_value_as_visit_length_days: Optional[bool] = False,
    visit_type: Optional[
        Literal["admissions", "ambulatory_visits", "emergency_visits"]
    ] = None,
) -> pd.DataFrame:
    """Load pshysical visits to both somatic and psychiatry.

    Args:
        shak_code (Optional[int], optional): SHAK code indicating where to keep/not keep visits from (e.g. 6600). Combines with
            shak_sql_operator, e.g. "!= 6600". Defaults to None, in which case all admissions are kept.
        shak_sql_operator (Optional[str], optional): Operator to use with shak_code. Defaults to "=".
        where_clause (Optional[str], optional): Extra where-clauses to add to the SQL call. E.g. dw_ek_borger = 1. Defaults to None. # noqa: DAR102
        where_separator (Optional[str], optional): Separator between where-clauses. Defaults to "AND".
        n_rows (Optional[int], optional): Number of rows to return. Defaults to None.
        return_value_as_visit_length_days (Optional[bool], optional): Whether to return length of visit in days as the value for the loader. Defaults to False which results in value=1 for all visits.
        visit_type (Optional[Literal["admissions", "ambulatory_visits", "emergency_visits"]], optional): Whether to subset visits by visit type. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe with all physical visits to psychiatry. Has columns dw_ek_borger and timestamp.
    """

    # SHAK = 6600 ≈ in psychiatry

    d = {
        "LPR3": RawValueSourceSchema(
            view="[FOR_LPR3kontakter_psyk_somatik_inkl_2021_feb2022]",
            datetime_col="datotid_lpr3kontaktslut",
            value_col="datotid_lpr3kontaktstart",
            location_col="shakkode_lpr3kontaktansvarlig",
            where_clause="AND [Kontakttype] = 'Fysisk fremmøde'",
        ),
        "ambulatory_visits": RawValueSourceSchema(
            view="[FOR_besoeg_psyk_somatik_LPR2_inkl_2021_feb2022]",
            datetime_col="datotid_slut",
            value_col="datotid_start",
            location_col="shakafskode",
            where_clause="AND ambbesoeg = 1",
        ),
        "emergency_visits": RawValueSourceSchema(
            view="[FOR_akutambulantekontakter_psyk_somatik_LPR2_inkl_2021_feb2022]",
            datetime_col="datotid_slut",
            value_col="datotid_start",
            location_col="afsnit_stam",
            where_clause="",
        ),
        "admissions": RawValueSourceSchema(
            view="[FOR_indlaeggelser_psyk_somatik_LPR2_inkl_2021_feb2022]",
            datetime_col="datotid_udskrivning",
            value_col="datotid_indlaeggelse",
            location_col="shakKode_kontaktansvarlig",
            where_clause="",
        ),
    }

    allowed_visit_types = ("admissions", "ambulatory_visits", "emergency_visits")
    if visit_type not in allowed_visit_types:
        raise ValueError(
            f"Invalid visit type. Allowed types of visits are {allowed_visit_types}."
        )

    if visit_type:
        LPR3_types = {
            "admissions": "'Indlæggelse'",
            "ambulatory_visits": "'Ambulant'",
            "emergency_visits": "'Akut ambulant'",
        }
        d = {key: d[key] for key in ["LPR3", visit_type]}
        d["LPR3"].where_clause += f" AND pt_type = {LPR3_types[visit_type]}"

    dfs = []

    for schema in d.values():
        cols = f"{schema.datetime_col}, dw_ek_borger"

        if return_value_as_visit_length_days:
            cols += f", {schema.value_col} AS value_col"

        sql = f"SELECT {cols} FROM [fct].{schema.view} WHERE {schema.datetime_col} IS NOT NULL {schema.where_clause}"

        if shak_code is not None:
            sql += f" AND {schema.location_col} != 'Ukendt'"
            sql += f" AND left({schema.location_col}, {len(str(shak_code))}) {shak_sql_operator} {str(shak_code)}"

        if where_clause is not None:
            sql += f" {where_separator} {where_clause}"

        df = sql_load(sql, database="USR_PS_FORSK", chunksize=None, n_rows=n_rows)
        df.rename(columns={schema.datetime_col: "timestamp"}, inplace=True)

        dfs.append(df)

    # Concat the list of dfs
    output_df = pd.concat(dfs)

    # 0,8% of visits are duplicates. Unsure if overlap between sources or errors in source data. Removing.
    output_df = output_df.drop_duplicates(
        subset=["timestamp", "dw_ek_borger"],
        keep="first",
    )

    # Change value column to length of admission in days
    if return_value_as_visit_length_days:
        output_df["value"] = (
            output_df["timestamp"] - pd.to_datetime(output_df["value_col"])
        ).dt.total_seconds() / 86400
        output_df = output_df.drop(columns="value_col")
    else:
        output_df["value"] = 1

    log.info("Loaded physical visits")

    return output_df.reset_index(drop=True)


@data_loaders.register("physical_visits")
def physical_visits(n_rows: Optional[int] = None) -> pd.DataFrame:
    """Load physical visits to all units."""
    return physical_visits(n_rows=n_rows)


@data_loaders.register("physical_visits_to_psychiatry")
def physical_visits_to_psychiatry(
    n_rows: Optional[int] = None,
    timestamps_only: bool = True,
) -> pd.DataFrame:
    """Load physical visits to psychiatry."""
    df = physical_visits(shak_code=6600, shak_sql_operator="=", n_rows=n_rows)

    if timestamps_only:
        df = df.drop("value", axis=1)

    return df


@data_loaders.register("physical_visits_to_somatic")
def physical_visits_to_somatic(n_rows: Optional[int] = None) -> pd.DataFrame:
    """Load physical visits to somatic."""
    return physical_visits(shak_code=6600, shak_sql_operator="!=", n_rows=n_rows)


@data_loaders.register("admissions")
def admissions(
    n_rows: Optional[int] = None,
    return_value_as_visit_length_days: Optional[bool] = False,
    shak_code: Optional[int] = None,
    shak_sql_operator: Optional[str] = None,
) -> pd.DataFrame:
    """Load admissions."""
    return physical_visits(
        visit_type="admissions",
        return_value_as_visit_length_days=return_value_as_visit_length_days,
        n_rows=n_rows,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
    )


@data_loaders.register("ambulatory_visits")
def ambulatory_visits(
    n_rows: Optional[int] = None,
    return_value_as_visit_length_days: Optional[bool] = False,
    shak_code: Optional[int] = None,
    shak_sql_operator: Optional[str] = None,
) -> pd.DataFrame:
    """Load ambulatory visits."""
    return physical_visits(
        visit_type="ambulatory_visits",
        return_value_as_visit_length_days=return_value_as_visit_length_days,
        n_rows=n_rows,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
    )


@data_loaders.register("emergency_visits")
def emergency_visits(
    n_rows: Optional[int] = None,
    return_value_as_visit_length_days: Optional[bool] = False,
    shak_code: Optional[int] = None,
    shak_sql_operator: Optional[str] = None,
) -> pd.DataFrame:
    """Load emergency visits."""
    return physical_visits(
        visit_type="emergency_visits",
        return_value_as_visit_length_days=return_value_as_visit_length_days,
        n_rows=n_rows,
        shak_code=shak_code,
        shak_sql_operator=shak_sql_operator,
    )
