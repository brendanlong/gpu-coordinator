Shapes of the two files two builds of gpuc share.

`*.current.json` are copies of what a build wrote at the time (`hosts.json`
from a real local registry -- including a RunPod host carrying the
`"ttl_hours": null` that started all this, with its endpoint and pod id
redacted -- and `config.json` as bootstrap rendered one). `ttl_hours` has since
been removed altogether, so every fixture now also proves that a key this build
no longer knows is ignored. `*.older.json` are hand-written in the shape a
build from before `ttl_hours` became optional wrote -- no `schema_version`, no
`retention_days`, no `gpu_info`, a concrete `ttl_hours: 24.0`. `*.newer.json` are hand-written as a *future* build might:
unknown keys, and explicit nulls where a field has since become optional.

Every one of them must parse under today's models. That is the regression test
for the incident where one `"ttl_hours": null` in the shared registry made
every subcommand of another session fail validation, and crashed the on-host
dispatcher 20 times in `HostConfig.from_dict`.
