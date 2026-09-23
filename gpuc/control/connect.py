"""Register a host, and change one: the two ways onto a host's own config.

A host owns its `config.json` -- its cards, its mirror, its env, its timers --
so `gpuc host add` is a *connect*: read that file, and if it is there, adopt
it. A second control machine meeting a host the first one set up is therefore
the ordinary path and not a special one, and nothing about the machine that
bootstrapped a host first matters afterwards. Only a host that has no config at
all is configured from the flags that registered it (`config.first_config`),
and by default it owns every card nvidia-smi reports there.

`gpuc host set` writes through to the same file. There is no local copy to set,
so it does not work offline, which is the point.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from gpuc.control.config import (
    ConfigError,
    HostEntry,
    Settings,
    config_changes,
    first_config,
    load_settings,
    transport_for,
)
from gpuc.control.remote import HostConfigRead, read_config, resolve_home, write_config
from gpuc.control.transport import Transport
from gpuc.host import jobs
from gpuc.host.jobs import HostConfig


class ConnectError(ConfigError):
    """The host could not be asked, or answered something we will not act on."""


@dataclass
class Connection:
    """One completed connect: the entry to register, and what it cost the host."""

    entry: HostEntry
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
    before_write: Callable[[HostEntry], None] = lambda entry: None,
) -> Connection:
    """Read the host's config (or write its first one) and return its entry.

    `fields` is the config the flags asked for, and only the keys that were
    given: on a host that already has a config each one is an explicit
    override, written through to the host and reported field by field; on a
    host that has none they are overrides of `first_config`, whose default
    `gpus` is every card the probe saw (`address.gpu_info`), less any it was
    asked to share.

    `before_write` is judged once the host's own name is known and before
    anything is written to it, so a caller that refuses the result refuses it
    without having changed the host first.
    """
    settings = settings if settings is not None else load_settings()
    transport = transport or transport_for(address, settings)
    home = resolve_home(transport, address)
    existing = _read(transport, home, address.name)
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
        # Whatever else has to be true of the name we just took -- it is the
        # caller's rule, and it is judged here, before the flags below reach
        # somebody's host.
        before_write(entry)
    else:
        if "gpus" not in patch and "shared_gpus" not in patch and not address.gpu_info:
            # An empty default is a driver that is still coming up, or an
            # nvidia-smi that is missing, as often as a box with no cards.
            # Writing "owns nothing" as the host's first config would stick.
            raise ConnectError(
                f"{address.name} reports no GPUs (nvidia-smi is missing there, or found no "
                f"cards), so there is nothing for it to own by default. Pass --gpus '' to "
                f"register it with none, or fix nvidia-smi on it and add it again."
            )
        patch = first_config(address, settings, patch)
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
    existing = _read(transport, home, entry.name)
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
    _refuse_shared_overlap(entry, existing, patch)
    changes = config_changes(existing, patch)
    # A patch that changes nothing the host is not already doing does not write
    # it: `gpuc host set` repeating what a host says is no reason to replace a
    # file a dispatcher is reading, and it is what the report says happened.
    # A host with no config at all is the exception -- there is the whole file.
    document = (
        write_config(transport, home, patch, host=entry.name)
        if patch and (changes or not existing)
        else existing
    )
    return Connection(
        entry=entry.with_config(document),
        home=home,
        adopted=adopted,
        changes=changes,
    )


def _read(transport: Transport, home: str, name: str) -> dict[str, Any]:
    """The host's config, or `{}` for a host that has none; never a guess."""
    return refuse_unreadable(read_config(transport, home), name, home).document or {}


def refuse_unreadable(
    read: HostConfigRead, name: str, home: str, error: type[Exception] = ConnectError
) -> HostConfigRead:
    """A config that is there and cannot be read is a file the host is running
    on, so nothing -- connect, `host set` or bootstrap -- writes over it.
    `error` is the caller's own failure class, so bootstrap's refusal is a
    bootstrap failure and connect's a connect failure."""
    if read.unreadable:
        raise error(
            f"host {name} answered, but {home}/config.json could not be read ({read.unreadable}) "
            f"-- and what a host is is that file, so nothing here will replace it.\n"
            f"Check that it is readable by this user and holds JSON; delete it to set that "
            f"host up again, or point somewhere else with --gpuc-home."
        )
    return read


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
    # `--env` replaces what somebody set by hand; the keys bootstrap derived
    # from the host's own filesystem (`jobs.MANAGED_ENV`) are carried over,
    # and only an explicit update moves or clears one of those.
    env = jobs.sticky_env(theirs, dict(patch.get("env", theirs)))
    for key, value in (env_updates or {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    # `--env K=` with nothing after the sign removes K, sticky or not: the one
    # way to say "no HF_HOME at all" on a host bootstrap gave one.
    return {**patch, "env": {k: v for k, v in env.items() if v != ""}}


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
        f"--gpus {','.join(named)} shares {', '.join(shared)} with that list without "
        f"matching it.\n"
        f"That is the one difference that can hand one card to two jobs, so it is refused "
        f"rather than warned about: drop --gpus to adopt what the host has, name a disjoint "
        f"set to reassign it, or pass --force if you are sure."
    )


def _refuse_shared_overlap(
    entry: HostEntry, existing: Mapping[str, Any], patch: Mapping[str, Any]
) -> None:
    """Refuse a config that has one card both owned and shared.

    The two lists say opposite things about a card -- hand this out, and borrow
    this only while nobody else is on it -- so a card in both is never what
    anybody meant, and the dispatcher has to resolve it somehow (it keeps the
    owned claim). Judged on the *result*, patch over what the host holds, so
    `--shared-gpus 3` on a host that already owns 3 is caught as readily as
    both flags in one command.

    The host's own health check refuses this too, at bootstrap. This is the
    copy that fires where it was typed: on every write, so a hand-edited
    config in that state is refused by the next `host set` about anything,
    and adopting one unchanged is not a write.
    """
    if not patch:
        return
    merged = {**(existing if isinstance(existing, dict) else {}), **patch}
    config = HostConfig.from_dict(merged)
    cards = _by_uuid(entry)
    owned = {cards.get(item, item) for item in config.gpus}
    both = [item for item in config.shared_gpus if cards.get(item, item) in owned]
    if not both:
        return
    raise ConnectError(
        f"host {entry.name} would have {', '.join(both)} in both --gpus and --shared-gpus, "
        f"and a card is either ours to hand out or somebody else's to borrow.\n"
        f"Owned: {', '.join(config.gpus) or 'none'}. Shared: {', '.join(config.shared_gpus)}."
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
