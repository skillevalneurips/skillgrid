"""Dependency compatibility checks for optional integrations."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version


class CompatibilityError(RuntimeError):
    """Raised when an unsupported dependency combination is detected."""


def installed_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def check_transformers_trl_compat(strict: bool = True) -> list[str]:
    """Validate installed transformers/trl versions.

    Rules:
      - If ``trl`` is installed, ``transformers`` must also be installed.
      - ``trl`` requires ``transformers>=4.46`` (processing_class support).
      - Warn when ``transformers>=5`` because API breakages are likely.
    """
    warnings: list[str] = []
    tf_v = installed_version("transformers")
    trl_v = installed_version("trl")

    if trl_v is not None and tf_v is None:
        msg = "Detected trl but transformers is missing."
        if strict:
            raise CompatibilityError(msg)
        warnings.append(msg)
        return warnings

    if tf_v is not None:
        tf_version = Version(tf_v)
        if tf_version >= Version("5.0.0"):
            warnings.append(
                f"transformers=={tf_v} detected; this project is validated on <5.0.0."
            )

    if trl_v is not None and tf_v is not None:
        tf_version = Version(tf_v)
        trl_version = Version(trl_v)
        if tf_version < Version("4.46.0"):
            msg = (
                f"Incompatible versions: trl=={trl_v} requires transformers>=4.46.0, "
                f"but found transformers=={tf_v}."
            )
            if strict:
                raise CompatibilityError(msg)
            warnings.append(msg)
        if trl_version >= Version("1.0.0"):
            warnings.append(
                f"trl=={trl_v} detected; this project is validated on trl<1.0.0."
            )
    return warnings
