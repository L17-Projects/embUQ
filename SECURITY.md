# Security policy

`MesoUQ` is a public research-to-release software line. Security issues should be reported privately first so the maintainer can assess impact before disclosure.

## Supported versions

Only the currently maintained public release line receives security fixes:

| Version line | Supported |
| --- | --- |
| `main` | Yes |
| latest `v0.1.x` release line | Yes |
| older snapshots / stale release-hardening branches | No |

## Reporting a vulnerability

Please do **not** open a public GitHub issue for a suspected vulnerability.

Preferred reporting order:

1. use GitHub private vulnerability reporting or a private security advisory if that feature is enabled for the repository;
2. otherwise, contact the maintainer directly before any public disclosure;
3. use a normal public issue only for non-sensitive hardening problems that do not expose users or infrastructure.

Include:

- affected commit, tag, or branch
- a concise description of the issue and impact
- reproduction steps or a minimal proof-of-concept
- any environment constraints such as Python, MPI, CUDA, or cluster-specific setup
- whether the issue affects vendored dependencies, public workflow scripts, or the packaged `meso_uq` surface

## Disclosure expectations

The initial goal is triage, reproducibility, scope assessment, and a safe fix path. Coordinated disclosure is preferred over immediate public posting for issues with real security impact.

## Scope notes

- This policy covers the public `MesoUQ` repository contents, including the packaged Python surface, workflow wrappers, and committed CI/operator tooling.
- Issues that originate entirely in third-party software such as the vendored Korali subtree may still be tracked here when they affect the shipped release line, but the upstream project remains the canonical fix owner for the dependency itself.
