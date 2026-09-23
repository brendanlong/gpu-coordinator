Shapes of the two files two builds of gpuc share.

`*.current.json` are what this build writes (`hosts.json` from a local
registry with a rental's endpoint and pod id redacted, and `config.json` as
bootstrap renders one). `*.newer.json` are hand-written as a *future* build
might write them: unknown keys at every level, and explicit nulls where a field
has since become optional.

Every one of them must parse under today's models, and the current registry
must round-trip unchanged. That is the regression test for the incident where
one `"ttl_hours": null` in the shared registry made every subcommand of
another session fail validation, and crashed the on-host dispatcher 20 times
in `HostConfig.from_dict`. The shapes earlier builds of this repository wrote
are not kept: pre-release, a registry is re-created with `gpuc host add`, and
what tolerance promises is that unknown keys and nulls never fail, not that
every historical layout is resurrected.
