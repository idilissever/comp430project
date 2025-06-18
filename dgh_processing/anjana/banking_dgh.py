from pathlib import Path
from dgh_processing.dgh import DGH
from dgh_processing.anjana.anjana_dgh import build_dgh_from_csv


def get_csv_banking_dghs() -> list[DGH]:
	"""Return a dictionary of DGHs for the banking dataset using CSV-based hierarchy files."""

	spec_dir = Path("../anjana_anonymizer/banking_hierarchies")
	dghs: list[DGH] = [
		build_dgh_from_csv(spec_dir / "age.csv", "age"),
		build_dgh_from_csv(spec_dir / "balance.csv", "balance"),
		build_dgh_from_csv(spec_dir / "education.csv", "education"),
		build_dgh_from_csv(spec_dir / "job.csv", "job"),
		build_dgh_from_csv(spec_dir / "marital.csv", "marital"),
		build_dgh_from_csv(spec_dir / "month.csv", "month"),
		build_dgh_from_csv(spec_dir / "previous.csv", "previous"),
	]
	return dghs
