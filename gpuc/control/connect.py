"""Register a host, and change one: the two ways onto a host's own config.

A host owns its `config.json` -- its cards, its mirror, its env, its timers --
so `gpuc host add` is a *connect*: read that file, and if it is there, adopt
it. A second control machine meeting a host the first one set up is therefore
the ordinary path and not a special one, and nothing about the machine that
bootstrapped a host first matters afterwards. Only a host that has no config at
all is configured from the flags that registered it.

`gpuc host set` writes through to the same file. There is no local copy to set,
so it does not work offline, which is the point.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from gpuc.control.config import (
    ConfigError,
    HostEntry,
    Settings,
    config_changes,
    transport_for,
    utc_now,
)
from gpuc.control.remote import read_remote_config, resolve_home, write_remote_config
from gpuc.control.transport import Transport
from gpuc.host import jobs
from gpuc.host.jobs import HostConfig


class ConnectError(ConfigError):
    """The host could not be asked, or answered something we will not act on."""


@dataclass
class Connection:
    """One completed connect: the entry to register, and what it cost the host."""

    entry: HostEntry
    transport: Transport
    home: str
    adopted: bool = False
    """True when the host already had a config and we took it as it was."""
    changes: list[str] = field(default_factory=list)
    """Per-field lines for what this connect wrote to the host's own config."""


def connect_host(
    address: HostEntry,
    settings: Settings | None = None,
    *,
    fields: Mapping[str, Any] | None = None,
    env_updates: Mapping[str, str | None] | None = None,
    transport: Transport | None = None,
    force: bool = False,
    gpu_hint: str = "",
) -> Connection:
    """Read the host's config (or write its first one) and return its entry.

    `fields` is the config the flags asked for, and only the keys that were
    given: on a host that already has a config each one is an explicit
    override, written through to the host and reported field by field; on a
    host that has none they are its initial config, and `gpus` must be among
    them because nothing else can know which of the cards in the box are ours.
    """
    transport = transport or transport_for(address, settings)
    home = resolve_home(transport, address)
    existing = _read_config(transport, home, address.name)
    patch = dict(fields or {})
    entry = address
    if existing:
        _refuse_overlapping_gpus(address, existing, patch, force)
        # The host's own name wins: it is what `config.json` says, what the
        # host reports as its own, and what its `s3_prefix` was derived from.
        # Read off the document rather than the parsed config, whose default
        # name is `local` -- which would quietly rename somebody's ssh box.
        name = str(existing.get("host") or "") or address.name
        entry = address.model_copy(update={"name": name})
    else:
        if "gpus" not in patch:
            raise ConnectError(
                f"host {address.name} has no config of its own yet ({home}/config.json does not "
                f"exist), so this is the machine that decides what it is: pass --gpus with the "
                f"cards gpuc may use there (--gpus '' for none).{gpu_hint}"
            )
        patch.setdefault("host", address.name)
        patch.setdefault("created_at", utc_now())
        provider = address.provider()
        if provider is not None:
            patch.setdefault("provider", provider)
    return _apply(entry, transport, home, existing, patch, env_updates, adopted=bool(existing))


def push_config(
    entry: HostEntry,
    settings: Settings | None = None,
    *,
    fields: Mapping[str, Any] | None = None,
    env_updates: Mapping[str, str | None] | None = None,
    transport: Transport | None = None,
) -> Connection:
    """`gpuc host set`: change the host's own config, and cache what it now holds."""
    transport = transport or transport_for(entry, settings)
    home = resolve_home(transport, entry)
    existing = _read_config(transport, home, entry.name)
    return _apply(
        entry, transport, home, existing, dict(fields or {}), env_updates, adopted=bool(existing)
    )


def _apply(
    entry: HostEntry,
    transport: Transport,
    home: str,
    existing: dict[str, Any],
    patch: dict[str, Any],
    env_updates: Mapping[str, str | None] | None,
    *,
    adopted: bool,
) -> Connection:
    patch = _with_env(existing, patch, env_updates)
    changes = config_changes(existing, patch)
    # A patch that would leave the file exactly as it is does not write it: a
    # `gpuc host set` repeating what a host already says is not a reason to
    # touch the file a dispatcher is reading.
    merged = jobs.merged_config(existing, patch) if patch else existing
    document = (
        write_remote_config(transport, home, patch, python=entry.python, env=entry.env)
        if merged != existing
        else existing
    )
    return Connection(
        entry=entry.with_config(document),
        transport=transport,
        home=home,
        adopted=adopted,
        changes=changes,
    )


def _read_config(transport: Transport, home: str, name: str) -> dict[str, Any]:
    document = read_remote_config(transport, home)
    if document is None:
        raise ConnectError(
            f"host {name} answered ssh but {home}/config.json could not be read.\n"
            f"Check that {home} is readable by this user, or point the host somewhere else "
            f"with --gpuc-home."
        )
    return document


def _with_env(
    existing: Mapping[str, Any],
    patch: dict[str, Any],
    env_updates: Mapping[str, str | None] | None,
) -> dict[str, Any]:
    """Resolve the `env` this patch asks for against the one the host has.

    The host's `env` is replaced wholesale, because "set it to exactly this" is
    the only rule that can also express "set it to nothing". Both the flag that
    names one variable (`--cache-dir`) and the one key the host owns itself
    (`UV_CACHE_DIR`) therefore have to be resolved against what is there, which
    is why it happens here and not where the flags are parsed.
    """
    theirs = HostConfig.from_dict(existing).env
    if not env_updates and "env" not in patch:
        return patch
    env = dict(patch.get("env", theirs))
    # `--env` replaces what somebody set by hand; it is not where the uv cache
    # bootstrap derived from the host's own filesystem lives, and dropping that
    # silently costs every job on the host a full copy of every wheel. Only
    # `--cache-dir` (below) moves or clears it.
    if "UV_CACHE_DIR" in theirs:
        env.setdefault("UV_CACHE_DIR", theirs["UV_CACHE_DIR"])
    for key, value in (env_updates or {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return {**patch, "env": env}


def _refuse_overlapping_gpus(
    address: HostEntry, existing: Mapping[str, Any], patch: Mapping[str, Any], force: bool
) -> None:
    """Refuse a `--gpus` that claims some, but not all, of the host's cards.

    Every other flag on an adopted host is an override somebody can undo by
    typing it again. This one is not: two machines that each believe they own
    an overlapping share of one box hand the same card to two jobs, and the
    first anyone hears of it is a run that died out of memory. A disjoint list
    is a deliberate reassignment and goes through.

    Both lists are read through the cards the probe just saw, because `--gpus
    2,3` and `--gpus GPU-a,GPU-b` can name the same two cards -- and an index
    is the spelling somebody copies off the probe's own output.
    """
    wanted = patch.get("gpus")
    if not isinstance(wanted, list) or force:
        return
    cards = _by_uuid(address)
    mine = {cards.get(str(item), str(item)) for item in wanted}
    theirs = HostConfig.from_dict(existing).gpus
    # Named as the *host* spells them, which is how the sentence below reads.
    shared = [item for item in theirs if cards.get(item, item) in mine]
    if not shared or {cards.get(item, item) for item in theirs} == mine:
        return
    named = sorted(str(item) for item in wanted)
    raise ConnectError(
        f"host {address.name} is already configured with GPUs {', '.join(theirs)}, and "
        f"--gpus {','.join(named)} claims {', '.join(shared)} of them and not the rest.\n"
        f"That is the one difference that can hand one card to two jobs, so it is refused "
        f"rather than warned about: drop --gpus to adopt what the host has, name a disjoint "
        f"set to reassign it, or pass --force if you are sure."
    )


def _by_uuid(address: HostEntry) -> dict[str, str]:
    """`index -> uuid` for the cards this host's probe saw, plus uuid -> itself.

    Cards the probe did not see are left as they were typed: an unresolvable
    entry is a typo or a card this container was not given, which the health
    check refuses at bootstrap, not something to guess at here.
    """
    cards = {uuid: uuid for uuid in address.gpu_info}
    for uuid, info in address.gpu_info.items():
        if info.index is not None:
            cards[str(info.index)] = uuid
    return cards
