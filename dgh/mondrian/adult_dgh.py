from pathlib import Path

from dgh.dgh import DGH
from dgh.mondrian.mondrian_basic_dgh import build_dgh_from_file


def get_adult_dghs() -> dict[str, DGH]:
    """Return a dictionary of DGHs for the adult dataset."""

    # ---- construct all six hierarchies ----
    spec_dir = Path("../../data")
    dghs: dict[str, DGH] = {
        "marital_status": build_dgh_from_file(
            spec_dir / "adult_marital_status.txt", "marital_status"
        ),
        "native_country": build_dgh_from_file(
            spec_dir / "adult_native_country.txt", "native_country"
        ),
        "occupation": build_dgh_from_file(
            spec_dir / "adult_occupation.txt", "occupation"
        ),
        "race": build_dgh_from_file(spec_dir / "adult_race.txt", "race"),
        "relationship": build_dgh_from_file(
            spec_dir / "adult_relationship.txt", "relationship"
        ),
        "workclass": build_dgh_from_file(spec_dir / "adult_workclass.txt", "workclass"),
    }
    return dghs
