"""`gpuc host add`, `host set` and `host bootstrap`: what they do, for the CLI and the dashboard.

Each returns what its `--json` form prints and whatever the text form needs;
rendering stays in the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gpuc.control import rented
from gpuc.control.actions import (
    Answer,
    CliError,
    Interrupted,
    UsageError,
    connection_document,
    make_provider,
    provider_for,
)
from gpuc.control.bootstrap import (
    BootstrapError,
    BootstrapResult,
    HealthOptions,
    bootstrap_host,
)
from gpuc.control.config import (
    ConfigError,
    HostEntry,
    LocalStateUnreadable,
    Reporter,
    Settings,
    forget_host,
    load_settings,
    open_registry,
    registry_transaction,
    update_cache,
)
from gpuc.control.connect import Connection, connect_host, push_config
from gpuc.control.jsonout import warn
from gpuc.control.probe import ProbeReport, probe_host
from gpuc.control.remote import Gone, RemoteError, rental_state
from gpuc.control.transport import TransportError


@dataclass
class HostChange:
    """What `host add` or `host set` did: the `--json` document and the text lines.

    `warnings` are the ones the text form says on stderr rather than beside the
    result; `document` carries them either way.
    """

    document: dict[str, Any]
    lines: list[str]
    warnings: list[str] = field(default_factory=list)

    def render(self) -> str:
        return "\n".join(self.lines)


def _pod_address(address: HostEntry, pod_id: str, settings: Settings) -> HostEntry:
    """Where a rented pod is now, asked of the provider that is billing for it.

    Adopting a pod another machine created is the ordinary path, not a special
    one: the pod owns its config and carries its own record of what it was
    bought as (`rented`), so all this has to find is the door.
    """
    provider = make_provider(settings)
    pod = provider.get(pod_id)
    if pod is None or provider.is_gone(pod):
        raise CliError(
            f"pod {pod_id} is {'gone' if pod is None else pod.status} on this account, so "
            f"there is nothing to add. `gpuc pods` lists the pods it can see."
        )
    reached = rented.address_for(address.name, pod, provider.name)
    if reached is None:
        raise CliError(
            f"pod {pod_id} ({pod.name}) is {pod.status} and has no direct SSH endpoint yet, so "
            f"it cannot be asked what it is. Try again once `gpuc pods` shows it RUNNING."
        )
    return address.model_copy(
        update={"ssh": reached.ssh, "port": reached.port, "rental": reached.rental}
    )


def add_host(
    name: str,
    *,
    ssh: str | None = None,
    port: int = 22,
    gpuc_home: str | None = None,
    persistent_root: str | None = None,
    pod_id: str | None = None,
    fields: dict[str, Any],
    env_updates: dict[str, str | None],
    force: bool = False,
) -> HostChange:
    """Register a host by asking it what it is.

    The registry holds the address; the host holds its config. So this probes,
    and a host that already has a `config.json` is adopted as it stands --
    which is what makes a second machine driving a box somebody else set up the
    ordinary path. Flags are explicit overrides of it, and say so.
    """
    settings = load_settings()
    if pod_id and ssh:
        raise UsageError(
            f"--pod {pod_id} finds the host's ssh endpoint from the provider, so it cannot be "
            f"given --ssh {ssh} as well."
        )
    address = HostEntry(
        name=name,
        ssh=ssh,
        port=port,
        gpuc_home=gpuc_home,
        persistent_root=persistent_root,
    )
    if pod_id:
        address = _pod_address(address, pod_id, settings)
    report = probe_host(address, settings)
    address = address.with_cache(
        gpu_info=report.gpu_info,
        driver_version=report.driver_version,
        python=report.host_python,
    )
    connection = connect_host(
        address,
        settings,
        fields=fields,
        env_updates=env_updates,
        force=force,
        before_write=lambda adopted: _refuse_a_taken_name(
            open_registry().registry.hosts.get(adopted.name), adopted, name
        ),
    )
    entry = connection.entry
    with registry_transaction() as registry:
        current = registry.hosts.get(entry.name)
        _refuse_a_taken_name(current, entry, name)
        if current is not None:
            # Re-registering a host this machine knows: its own bootstrap and
            # the interpreter that bootstrap chose are still true, and worth
            # more than what a probe can see.
            entry = entry.with_cache(python=current.python).model_copy(
                update={"bootstrapped_at": current.bootstrapped_at}
            )
        registry.put(entry)
    warnings: list[str] = []
    if not connection.adopted and not entry.config.gpus:
        warnings.append(_owns_nothing_warning(entry, fields, report))
    if pod_id and not connection.adopted:
        # A pod nobody has set up has no dispatcher, so nothing will ever idle
        # it out: it bills until bootstrap gives it one or a person ends it.
        warnings.append(
            f"nothing has bootstrapped this pod, so nothing on it will ever terminate it: "
            f"`gpuc host bootstrap {entry.name}` gives it a dispatcher that does"
        )
    lines = [_added_line(entry, connection, name)]
    lines += [f"  {warning}" for warning in warnings]
    return HostChange(connection_document(entry, connection, warnings=warnings), lines)


def _owns_nothing_warning(entry: HostEntry, fields: dict[str, Any], report: ProbeReport) -> str:
    """A first config that owns no card is legal and useless; say which it was."""
    if "gpus" in fields:
        why = "--gpus '' asked for none"
    elif not report.has_nvidia_smi:
        why = "it has no nvidia-smi"
    elif entry.config.shared_gpus:
        why = "every card it has is shared"
    else:
        why = "nvidia-smi found no cards on it"
    return (
        f"it owns no GPUs ({why}), so nothing can be submitted to it: "
        f"`gpuc host set {entry.name} --gpus <list>` assigns some"
    )


def _refuse_a_taken_name(current: HostEntry | None, entry: HostEntry, asked_for: str) -> None:
    """Never let an adopted name replace a different host registered under it.

    A host's `config.json` says what it calls itself, and taking that name is
    what keeps two machines agreeing about one box. But a box somebody set up
    as `local` on their own machine is called `local` here too, and registering
    it would otherwise overwrite *this* machine's `local` -- silently, since
    the address is the only thing that differs.

    Judged before the host is written to as well as before the registry is, so
    a refusal does not leave the flags applied to somebody's host.
    """
    if entry.name == asked_for or current is None:
        return
    if (current.ssh, current.port, current.remote_home) == (
        entry.ssh,
        entry.port,
        entry.remote_home,
    ):
        return
    raise CliError(
        f"host {asked_for} calls itself {entry.name!r}, and a different host is already "
        f"registered here under that name ({current.ssh or 'this machine'}).\n"
        f"Registering it would replace that one. If they are the same box, remove the entry "
        f"here first (`gpuc host remove {entry.name}`); if they are not, give one of them a "
        f"name of its own by editing `host` in its ~/.gpuc/config.json."
    )


def _added_line(entry: HostEntry, connection: Connection, asked_for: str) -> str:
    lines = [
        f"added host {entry.name} [{entry.kind}] "
        f"{entry.ssh or 'this machine'} with {len(entry.config.gpus)} GPU(s)"
    ]
    if connection.adopted:
        lines.append(f"adopted the config on the host ({connection.home}/config.json)")
        if entry.name != asked_for:
            lines.append(
                f"the host calls itself {entry.name!r}, not {asked_for!r}, so that is the name "
                f"it is registered under here"
            )
        # What the flags changed about somebody's host. On a host that had no
        # config every field "changed", and the line above already said so.
        lines += [f"  host <- {change}" for change in connection.changes]
    else:
        lines.append(f"wrote its first config to {connection.home}/config.json")
    home = _home_line(entry)
    if home:
        lines.append(home)
    lines.append(f"next: gpuc host bootstrap {entry.name}")
    return "\n".join(lines)


def _home_line(entry: HostEntry) -> str:
    if entry.root is None:
        return ""
    return (
        f"persistent root {entry.root}, so gpuc home (queue, specs, state, logs, "
        f"workdirs) is {entry.remote_home}"
    )


def set_host(
    name: str,
    *,
    address: dict[str, object],
    fields: dict[str, Any],
    env_updates: dict[str, str | None],
) -> HostChange:
    """Change one host: its address here, its config on the host itself.

    The config half writes through to the host's `config.json`, because that is
    the only copy of it. It therefore needs the host to answer -- there is
    nothing to set offline -- and what it changed is reported field by field.
    """
    entry = open_registry().require(name)
    lines = [f"host {entry.name}:"]
    # The config first, and through the address the host still has: a
    # `--persistent-root` in the same command moves gpuc home, and writing the
    # config to where the host is not would leave the real one behind.
    config: dict[str, Any] | None = None
    if fields or env_updates:
        connection = push_config(entry, load_settings(), fields=fields, env_updates=env_updates)
        entry, config = connection.entry, connection.entry.cache.config
        lines += [f"  host <- {change}" for change in connection.changes] or [
            "  host already holds that config; nothing changed"
        ]
    else:
        # The address alone changed, so the host was not asked: the document
        # still says which config it holds, from the cache, dated as such.
        connection = Connection(entry=entry, home=entry.remote_home, adopted=True)
    entry = entry.model_copy(update=address)
    lines += [f"  here <- {key}={value!r}" for key, value in sorted(address.items())]
    warnings: list[str] = []
    with registry_transaction() as registry:
        current = registry.hosts.get(entry.name)
        if current is None:
            warnings.append(f"host {entry.name} was removed while this ran; nothing was registered")
        else:
            updated = current.model_copy(update=address)
            registry.put(updated if config is None else updated.with_config(config))
    home = _home_line(entry)
    if home:
        lines.append(home)
        lines.append(f"the host moves there on: gpuc host bootstrap {entry.name}")
    document = connection_document(entry, connection, warnings=warnings)
    return HostChange({**document, "address": address}, lines, warnings)


def bootstrap_and_record(
    entry: HostEntry, settings: Settings, health: HealthOptions, report: Reporter = print
) -> BootstrapResult:
    """Bootstrap one host, persist what it told us about itself, and say so."""
    updated, result = bootstrap_host(entry, settings, health_options=health, report=report)
    update_cache(
        updated.name,
        config=updated.cache.config,
        python=updated.python,
        uv=updated.uv,
        gpu_info=updated.gpu_info,
        driver_version=updated.driver_version,
        bootstrapped_at=updated.bootstrapped_at,
    )
    report(result.render())
    return result


@dataclass
class BootstrapTally:
    """The last word of a `--all` run: what worked, what did not, what was never read.

    Counted rather than claimed, because the run this ends can be long enough
    that nobody reads the middle of it: a host this build could not parse out
    of the registry was never bootstrapped either, and saying "all of them"
    over the top of that warning is how one gets missed for a month. One
    entry per registered host, in the order they were taken, so the `--json`
    form is the tally as data rather than as a sentence.
    """

    hosts: list[HostEntry]
    unreadable: list[str]
    """Registry entries this build could not read, by name: never attempted."""
    errors: list[str]
    outcomes: list[dict[str, Any]] = field(default_factory=list)
    interrupted: bool = False

    def record(
        self,
        entry: HostEntry,
        outcome: str,
        result: BootstrapResult | None = None,
        *,
        error: str | None = None,
    ) -> None:
        detail = result.document() if result else BootstrapResult.no_document()
        detail.pop("host", None)
        self.outcomes.append(
            {
                "name": entry.name,
                "outcome": outcome,
                "error": error,
                "ephemeral": entry.rental is not None,
                **detail,
            }
        )

    @property
    def done(self) -> list[str]:
        return [o["name"] for o in self.outcomes if o["outcome"] == "bootstrapped"]

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [o for o in self.outcomes if o["outcome"] == "failed"]

    @property
    def gone(self) -> list[str]:
        """Rentals the provider says no longer exist, forgotten as they were found."""
        return [o["name"] for o in self.outcomes if o["outcome"] == "gone"]

    def render(self) -> str:
        lines = [f"{len(self.done)}/{len(self.hosts)} host(s) bootstrapped"]
        if self.gone:
            lines.append(f"forgotten, their pods are gone: {', '.join(self.gone)}")
        if self.failed:
            lines.append(f"failed: {', '.join(o['name'] for o in self.failed)}")
            if any(o["ephemeral"] for o in self.failed):
                # Not one of the forgotten: the provider still has this pod, so
                # somebody has to decide whether to fix it or drop it.
                lines.append(
                    "a rental the provider still has is forgotten by `gpuc host remove <name>`"
                )
        if self.unreadable:
            lines.append(
                f"{len(self.unreadable)} host(s) in the registry could not be read (warnings above)"
            )
        return "\n".join(lines)

    def document(self) -> dict[str, Any]:
        """`gpuc host bootstrap --all --json`: every registered host and what became of it."""
        attempted = {o["name"] for o in self.outcomes}
        for entry in self.hosts:
            if entry.name not in attempted:
                self.record(entry, "not_attempted")
        return {
            "hosts": self.outcomes,
            "total": len(self.hosts),
            "bootstrapped": self.done,
            "failed": [o["name"] for o in self.failed],
            "gone": self.gone,
            "unreadable": list(self.unreadable),
            "interrupted": self.interrupted,
            "errors": list(self.errors),
        }

    def answer(self, text: str | None) -> Answer:
        """Exit 1 if any host failed, or was never attempted because this build
        could not read its entry: nobody may read a wall of output as "all
        upgraded" over the top of one that did not get done, whichever way."""
        failures = [o["error"] or o["name"] for o in self.failed]
        failures += [f"registry entry {name} could not be read" for name in self.unreadable]
        return Answer(self.document(), text, failures=failures)


def bootstrap_every_host(
    settings: Settings, health: HealthOptions, *, report: Reporter
) -> BootstrapTally:
    """`gpuc host bootstrap --all`: the upgrade loop, one command.

    A host that fails does not stop the others -- the hosts that are still
    there are the reason the flag exists. Each failure is named again in the
    tally, so nobody reads a wall of output as "all upgraded". A rental the
    provider no longer has is not one of those failures: it ended itself, so
    the entry is forgotten and the run carries on.

    A Ctrl-C ends the run with the tally so far: health alone allows five
    minutes a host, so this is a command somebody does give up on, and what
    it got through is still true.
    """
    read = open_registry()
    if read.unreadable:
        raise LocalStateUnreadable("\n".join(read.errors))
    hosts = list(read.registry.hosts.values())
    tally = BootstrapTally(hosts, sorted(read.skipped), list(read.errors))
    if not hosts:
        return tally
    # Built once, and only when a rental is registered: a host that fails to
    # answer may simply have ended, and only the provider knows which.
    provider = provider_for(hosts, settings, report)
    for index, entry in enumerate(hosts, start=1):
        if index > 1:
            report("")
        report(f"== {entry.name} ({index}/{len(hosts)}) ==")
        try:
            tally.record(
                entry, "bootstrapped", bootstrap_and_record(entry, settings, health, report)
            )
        except KeyboardInterrupt:
            tally.record(entry, "interrupted")
            tally.interrupted = True
            raise Interrupted(
                f"interrupted during {entry.name}\n{tally.render()}", tally.document()
            ) from None
        except LocalStateUnreadable:
            # The registry stopped being readable mid-run, so the next host's
            # write would be a guess: say how far this got, and exit 3. Under
            # --json the error document is the one stdout gets.
            report(f"\n{tally.render()}")
            raise
        except (BootstrapError, ConfigError, RemoteError, TransportError) as exc:
            state = rental_state(entry, provider)
            if isinstance(state, Gone):
                report(f"{state.reason}; forgetting this host")
                tally.record(entry, "gone")
                forget_host(entry.name, entry.pod_id, report)
                continue
            warn(f"host {entry.name}: {exc}")
            tally.record(entry, "failed", error=str(exc))
    return tally
