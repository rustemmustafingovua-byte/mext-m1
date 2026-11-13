# Security policy

## Reporting a vulnerability

Please **do not open a public issue**. Use GitHub's private reporting:

<https://github.com/rustemmustafingovua-byte/mext-m1/security/advisories/new>

Include: affected version (`mext --version`), macOS version and chip, a minimal
reproduction, and the impact you expect. We aim to acknowledge within 72 hours
and to ship a fix or mitigation within 14 days for confirmed issues.

## Threat model

MEXT is a local, single-user tool. By design:

- The dashboard binds `127.0.0.1` only and speaks plain HTTP on loopback.
- The CLI and the AI engine communicate over Unix sockets (`~/.mext/*.sock`)
  created with mode `0600`.
- No telemetry or network calls are made by the daemon. Optional features that
  reach the network (Hugging Face downloads, tunnels) are explicit and
  user-driven.
- `--trace-open` shells out to `/usr/bin/eslogger` or `/usr/bin/fs_usage` through
  `sudo -n`; it never stores or prompts for a password. Without passwordless
  sudo it degrades to FSEvents plus the `mext open` hook.
- Model weights, indexes and predictor state stay on disk under `~/.mext` and
  the workspace. MEXT does not upload them.

## Out of scope

- Malicious models or GGUF files that exploit MLX/llama.cpp — report those
  upstream.
- Physical access attacks, or other users on the same macOS account.
- Denial of service caused by deliberately setting an absurd `--ram-limit`.

## Supported versions

Only the latest release receives fixes. `0.6.x` is current.
