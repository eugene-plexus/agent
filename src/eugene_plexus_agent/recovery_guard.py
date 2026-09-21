"""A restored identity must not enroll or advertise before explicit activation."""

from pathlib import Path


def refuse_quarantined(config_file: Path) -> None:
    directory = config_file.resolve().parent
    if any((path / ".recovery-quarantine").exists() for path in (directory, directory.parent)):
        raise RuntimeError(
            "This is a quarantined recovery copy. Stop and fence the original install, "
            "then use recovery.py activate before starting this identity."
        )
