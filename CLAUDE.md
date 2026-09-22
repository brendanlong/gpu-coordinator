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
| [docs/media/README.md](docs/media/README.md) | the README's recording and screenshots, and how to make them again when the CLI's output or the dashboard changes |
| `gpuc/host/` | runs on hosts, stdlib only; `gpuc/control/` runs on the client and may use dependencies |
| `./check.sh` | lint, typecheck, tests; what CI runs. A bare `pytest` never rents hardware |

## Working here

- Read the spec before changing behaviour. A PR that moves the code away
  from it needs the spec changed in the same PR, deliberately.
- When behaviour changes, update the doc that describes it (usage.md or
  setup.md), ARCHITECTURE.md if the contract moved, and the skill if an agent
  would need to know.
- Open issues track where the code has not yet caught up with the spec.

## Writing the docs

Each doc has an altitude, and a fact belongs at exactly one of them:

- **SPEC.md** -- the requirement, stated once and straightforwardly. That there
  is a command to wait for jobs, not how it polls, and never a requirement plus
  the exceptions it has grown.
- **ARCHITECTURE.md** -- the invariant a second implementation would have to
  hold to, not a walkthrough of the code that holds it.
- **usage.md**, **setup.md** -- what a user does and what the tool does back:
  flags, refusals, exit codes, schemas. Not why it works that way.
- **docstrings** -- the *why*. Every "because", the incident that motivated a
  rule, the option that was rejected.

The test when cutting: **would a reader act differently without this
sentence?** Commentary fails it and a rule passes -- but the two look alike
while you are cutting, so check a candidate against the code, not against the
prose around it. A kill ladder reads like implementation detail right up until
you are the one trapping SIGTERM to checkpoint.

@docs/SPEC.md
