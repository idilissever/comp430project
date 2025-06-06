from pathlib import Path
from dgh.dgh import DGH
from dgh.anjana.anjana_dgh import build_dgh_from_csv


def get_csv_adult_dghs() -> list[DGH]:
    """Return a dictionary of DGHs for the adult dataset using CSV-based hierarchy files."""

    spec_dir = Path("../../anjana_anonymizer/hierarchies")
    dghs: list[DGH] = [
        build_dgh_from_csv(spec_dir / "age.csv", "age"),
        # build_dgh_from_csv(spec_dir / "country.csv", "native_country"),
        build_dgh_from_csv(spec_dir / "education.csv", "education"),
        build_dgh_from_csv(spec_dir / "marital.csv", "marital_status"),
        build_dgh_from_csv(spec_dir / "occupation.csv", "occupation"),
        # build_dgh_from_csv(spec_dir / "race.csv", "race"),
        # build_dgh_from_csv(spec_dir / "salary.csv", "salary_class"),
        build_dgh_from_csv(spec_dir / "sex.csv", "sex"),
        build_dgh_from_csv(spec_dir / "workclass.csv", "workclass"),
    ]
    return dghs
