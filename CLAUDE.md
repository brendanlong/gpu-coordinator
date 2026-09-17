# gpu-coordinator

`gpuc`: provision GPU hosts (local, ssh, rented) and queue jobs on them.

## Where things are

| | |
| --- | --- |
| [docs/SPEC.md](docs/SPEC.md) | goals and non-goals. Every change is checked against it; where it and anything else disagree, the spec wins. Included below |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | the contract the code keeps: on-host state, dispatcher and runner rules, transport, testing rules, code conventions |
| [docs/setup.md](docs/setup.md), [docs/usage.md](docs/usage.md) | user-facing behaviour: install, hosts, the job spec, every command, failure reasons, `--json` schemas |
| [skills/gpuc/SKILL.md](skills/gpuc/SKILL.md) | the agent guide, shipped in the wheel; `gpuc skill` prints it |
| [README.md](README.md) | the short public overview |
| `gpuc/host/` | runs on hosts, stdlib only; `gpuc/control/` runs on the client and may use dependencies |
| `./check.sh` | lint, typecheck, tests; what CI runs. A bare `pytest` never rents hardware |

## Working here

- Read the spec before changing behaviour. A PR that moves the code away
  from it needs the spec changed in the same PR, deliberately.
- When behaviour changes, update the doc that describes it (usage.md or
  setup.md), ARCHITECTURE.md if the contract moved, and the skill if an agent
  would need to know.
- Open issues track where the code has not yet caught up with the spec.

@docs/SPEC.md
