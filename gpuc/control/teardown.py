"""`gpuc host terminate`: end a rental now, because a person said so.

A pod ends itself when its queue goes idle, and nothing on a client watches it
-- that is the contract, and this command does not change it. It is the other
half: the deliberate, one-step "stop billing for this, now", for the pod whose
dispatcher is dead and which therefore will never idle out, and for the one
that is perfectly healthy and simply no longer wanted.

It goes at the provider, not at the host: a hard terminate is the one call
that works whether or not anything on the pod still answers, and once it has
run there is no host left to own any state.

What it costs is anything the pod had not finished uploading, so the rule is
that **the host has to say it is idle**. A host that says it is busy, a host
that did not answer, and a pod registered nowhere here are all the same
refusal, because they are the same fact: this machine cannot say that ending
the pod now throws nothing away. `--force` is how the user says it anyway, and
skipping the question is also what makes it the fast path against a pod that
could never have answered.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from gpuc.control import status as status_mod
from gpuc.control.config import (
    HostEntry,
    HostNotFound,
    Registry,
    Reporter,
    Settings,
    forget_host,
)
from gpuc.control.providers.base import Pod, Provider, ProviderError


class TerminateError(RuntimeError):
    """The pod was not ended. Reported as the command's own failure, exit 1."""


class TerminateRefused(TerminateError):
    """The pod is doing something; the user is told what, and how to insist."""


class TerminateFailed(TerminateError):
    """The provider would not confirm the pod is gone, so it is still billing."""


@dataclass
class Target:
    """Which pod this is about, and what this machine knows about it."""

    pod_id: str
    pod: Pod | None = None
    entry: HostEntry | None = None

    @property
    def label(self) -> str:
        name = self.entry.name if self.entry else (self.pod.name if self.pod else None)
        return f"{name} ({self.pod_id})" if name else self.pod_id

    @property
    def named(self) -> str:
        """What to spell this pod as in a command we print: what the user typed
        would do, and a pod registered nowhere here has only its id."""
        return self.entry.name if self.entry else self.pod_id


@dataclass
class Termination:
    """What was ended, what it was doing, and what is left here to clean up."""

    target: Target
    checked: bool = False
    """Whether the host itself answered. Everything below is empty when it did
    not, which is not the same as the pod having had nothing to do."""
    unasked: str | None = None
    """Why it did not answer, when it was asked and did not. None under
    `--force`, which does not ask."""
    pod_dead: bool = False
    """The provider itself said the pod is dead or gone, so nothing on it can
    be running. Not the same as a pod it could not describe."""
    running: list[str] = field(default_factory=list)
    queued: list[str] = field(default_factory=list)
    outputs_pending: list[str] = field(default_factory=list)
    terminated: bool = False
    forgotten: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def busy(self) -> bool:
        return bool(self.running or self.queued or self.outputs_pending)

    def document(self) -> dict[str, Any]:
        pod = self.target.pod
        entry = self.target.entry
        return {
            "host": entry.name if entry else None,
            "pod_id": self.target.pod_id,
            "pod_name": pod.name if pod else None,
            # What the provider said *before* the terminate: a gone status
            # here with `terminated` false is a pod that had already ended.
            "pod_status": pod.status if pod else None,
            "cost_usd_hr": pod.cost_usd_hr if pod else None,
            "checked": self.checked,
            "running": list(self.running),
            "queued": list(self.queued),
            "outputs_pending": list(self.outputs_pending),
            "terminated": self.terminated,
            "forgotten": self.forgotten,
            "notes": list(self.notes),
        }


def resolve(target: str, registry: Registry, provider: Provider) -> Target:
    """A registered host name, a pod id, or a pod name -- in that order.

    The registry comes first because that is the name the user sees everywhere
    else, and a pod is only looked for when no host answers to the name: the
    provider read costs an API call, and a name this machine drives is not
    ambiguous. Pods without our prefix are not searched, for the same reason
    nothing else here touches them.
    """
    entry = registry.hosts.get(target)
    if entry is not None:
        if entry.rental is None:
            raise TerminateRefused(
                f"host {entry.name} is a {entry.kind!r} host: there is no rental to end.\n"
                f"`gpuc host remove {entry.name}` forgets it here, and nothing on it changes."
            )
        return Target(pod_id=entry.rental.pod_id, entry=entry)
    pod = _find_pod(target, provider)
    return Target(pod_id=pod.id, pod=pod, entry=_entry_for_pod(pod.id, registry))


def _find_pod(target: str, provider: Provider) -> Pod:
    for pod in provider.list_ours():
        if target in (pod.id, pod.name):
            return pod
    raise HostNotFound(
        f"no host or pod named {target!r}. `gpuc host list` has this machine's hosts and "
        f"`gpuc pods` every pod in the account with our prefix; a pod without the prefix "
        f"is not ours and is only ever ended in the provider's console."
    )


def _entry_for_pod(pod_id: str, registry: Registry) -> HostEntry | None:
    """The registry entry for a pod named by id, so it is forgotten too."""
    return next((e for e in registry.hosts.values() if e.pod_id == pod_id), None)


def inspect(target: Target, settings: Settings, provider: Provider) -> Termination:
    """What the pod is doing, as the host itself reports it.

    One `gpuc status` of that host: the provider read fills in what is being
    billed, and the ssh half fills in what would be thrown away. `checked` is
    the whole point of the return -- an empty queue and a question nobody
    answered must never read alike, and only one of them is safe to act on.
    """
    result = Termination(target=target)
    if target.entry is None:
        result.unasked = (
            f"pod {target.pod_id} is not registered on this machine, so nothing here can "
            f"ask what it is running"
        )
        # Only what the provider actually said: a read that failed leaves
        # `target.pod` None, and None is not a dead pod.
        result.pod_dead = target.pod is not None and provider.is_dead(target.pod)
        return result
    view = status_mod.gather(target.entry, settings, provider=provider)
    target.pod = view.pod or target.pod
    if view.pod_gone or view.pod_dead:
        # No ssh was attempted and none would have answered. What became of the
        # pod is the whole answer, and the caller has it in `target.pod`.
        result.pod_dead = True
        return result
    if not view.reachable:
        result.unasked = (
            f"could not ask {target.entry.name} what it is doing: {view.error or 'no answer'}"
        )
        return result
    result.checked = True
    result.running = [status_mod.job_label(job) for job in view.running]
    result.queued = [status_mod.job_label(job) for job in view.queue]
    result.outputs_pending = [status_mod.job_label(job) for job in view.outputs_at_risk]
    return result


def refusal(result: Termination) -> str:
    """Why this pod was not ended, and the two ways on. Exit 1, never silent.

    Both refusals end in `--force`, because both are the same question: this
    machine cannot say that ending the pod now throws nothing away, and only
    the user can decide to do it anyway.
    """
    target = result.target
    insist = f"  gpuc host terminate {target.named} --force"
    if result.unasked:
        lines = [
            result.unasked,
            "Nothing here knows whether it is running a job or still uploading.",
        ]
        if target.entry is None:
            lines.append(f"  gpuc host add <name> --pod {target.pod_id}   drive it from here first")
        lines.append(f"{insist}   end it now regardless")
        return "\n".join(lines)
    lines = [f"{target.label} is not idle:"]
    lines += [f"  running   {label}" for label in result.running]
    lines += [f"  queued    {label}" for label in result.queued]
    lines += [
        f"  unsynced  {label} (its outputs are not confirmed uploaded)"
        for label in result.outputs_pending
    ]
    lines += [
        "Terminating now kills those jobs and loses anything not already uploaded.",
        f"  gpuc host set {target.named} --idle-min 0   let it finish, then stop by itself",
        f"{insist}   end it now anyway",
    ]
    return "\n".join(lines)


def terminate(
    target: str,
    settings: Settings,
    *,
    registry: Registry,
    provider: Provider,
    force: bool = False,
    report: Reporter = print,
    sleep: Callable[[float], None] = time.sleep,
) -> Termination:
    """Resolve, ask the host unless `--force`, terminate, forget it here.

    **Without `--force` the host has to say it is idle.** Anything short of
    that -- work in flight, a host that did not answer, a pod registered
    nowhere here and so askable by nothing -- is refused, because all three
    mean the same thing: this machine cannot say that ending the pod now
    throws nothing away. An ssh that blipped must not cost somebody six hours
    of training just because a terminate happened to be typed.

    `--force` is therefore the fast path as well as the insistent one: it does
    not ask, which on the pod this command exists for -- one whose dispatcher
    is dead -- saves a minute of timeouts on a question nothing can answer.

    A pod the provider already calls dead is never refused over. There is no
    live container to be running anything, and holding the one command that
    frees the rental behind a flag would be protecting nothing. That takes
    the provider *saying* so: a provider read that failed leaves no pod to
    look at, which `is_dead` would also call dead, and a busy host would be
    ended over a 503.
    """
    resolved = resolve(target, registry, provider)
    if force:
        result = Termination(target=resolved)
    else:
        result = inspect(resolved, settings, provider)
        if not result.pod_dead and (result.busy or not result.checked):
            raise TerminateRefused(refusal(result))
    if resolved.pod is None:
        # What is being billed, for the result to report -- and whether there
        # is anything to end at all. A provider that will not answer raises
        # rather than reading as "gone": forgetting a pod over one bad response
        # is how a terminated entry and a running bill part company.
        resolved.pod = provider.get(resolved.pod_id)
    pod = resolved.pod
    if result.unasked:
        result.notes.append(result.unasked)
        report(f"note: {result.unasked}")

    if pod is None or provider.is_gone(pod):
        gone = "is already terminated" if pod else "does not exist at the provider"
        result.notes.append(f"pod {resolved.pod_id} {gone}; nothing was billing")
    else:
        if provider.is_dead(pod):
            result.notes.append(
                f"the provider says pod {resolved.pod_id} is {pod.status}, so nothing was "
                f"running on it; ending it frees the rental"
            )
        report(f"terminating {resolved.label}")
        try:
            provider.terminate_confirmed(resolved.pod_id, report=report, sleep=sleep)
        except ProviderError as exc:
            raise TerminateFailed(
                f"{resolved.label}: {exc}\n"
                f"It is still billing: `gpuc pods` shows it, and the provider's console "
                f"ends it."
            ) from exc
        report(f"{resolved.label} terminated and confirmed gone")
        result.terminated = True
    result.forgotten = _forget(resolved, report)
    return result


def _forget(target: Target, report: Reporter) -> bool:
    """Drop the registry entry for a pod that no longer exists; did it go?

    Only ever the entry that names *this* pod (`forget_host`'s own rule), and
    only after the provider has confirmed the terminate: a pod still billing
    must stay visible in `gpuc status`. The answer is what `forget_host`
    reports rather than what it was asked to do -- a lock another session is
    holding leaves the entry there, and saying otherwise is how a caller ends
    up believing a POD GONE line is a bug.
    """
    if target.entry is None:
        return False
    return forget_host(target.entry.name, target.pod_id, report)
