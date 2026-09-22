"""Read shared state the way Postel would: every field optional, nulls inert."""

from __future__ import annotations

import types
import typing
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


def _allows_none(annotation: Any) -> bool:
    if annotation is None or annotation is type(None) or annotation is Any:
        return True
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        return any(_allows_none(arg) for arg in typing.get_args(annotation))
    return False


class TolerantModel(BaseModel):
    """Every model that reads shared state derives from this.

    Every such model parses a file that another version of gpuc -- an older
    build still installed in a second session, a newer one from `uv tool
    upgrade` -- may have written. Two rules make that safe in both directions:
    an unknown key is ignored (a newer writer may add fields), and an explicit
    `null` for a field that is not nullable is dropped so the field's default
    applies (a newer writer may make a field optional). Without the second
    rule, one `"retention_days": null` in the shared registry made every subcommand
    of the other session -- including `status` and `logs` -- fail validation.
    """

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _null_means_default(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        document: dict[Any, Any] = data
        drop = [
            key
            for key, value in document.items()
            if value is None
            and isinstance(key, str)
            and key in cls.model_fields
            and not _allows_none(cls.model_fields[key].annotation)
        ]
        if not drop:
            return document
        return {key: value for key, value in document.items() if key not in drop}
