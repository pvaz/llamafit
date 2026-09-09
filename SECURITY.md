# Security policy

LlamaFit runs commands on your machine (vendor GPU tools, `llama-server`, `llama-bench`),
reads files you point it at, downloads llama.cpp builds and model weights in later phases, and
serves a web dashboard bound to localhost. Each of those is a place where a bug could matter.

## What counts as a security issue

- LlamaFit executing a command, path or URL that did not come from its own code, its bundled
  data, or an explicit user setting.
- The web dashboard or API becoming reachable from other machines without the user passing
  `--host` explicitly, or the API doing anything beyond reading state and starting the
  operations the user asked for.
- A download being written outside the configured directory, or a checksum failure being
  ignored.
- Any handling of the model catalog or Hugging Face metadata that lets remote content change
  what LlamaFit runs locally.

Incorrect recommendations, wrong speed estimates and crashes are bugs, not security issues;
please report those through the normal issue template.

## How to report

Use GitHub's private vulnerability reporting on this repository
(**Security** tab → **Report a vulnerability**). Do not open a public issue. Include the
LlamaFit version, your operating system, the command you ran and what you observed.

You will get an acknowledgement within a week. Fixes ship as a patch release with a
changelog entry that credits the reporter unless they prefer otherwise.

## Supported versions

The latest minor release receives security fixes. Older versions should upgrade.

## Design choices that limit exposure

- No code is ever downloaded and executed: llama.cpp comes from official release archives
  whose checksums are verified before extraction, and LlamaFit only starts binaries from the
  directory it installed to or the one the user configured.
- The catalog is data, validated against a schema; it can name files to download and flags to
  pass to `llama-server`, and flags are rendered from a fixed allow-list of known options.
- The dashboard binds to `127.0.0.1` by default and has no authentication because it is not
  meant to be exposed; passing another host prints a warning.
- Nothing is sent anywhere. The only outbound requests are to GitHub releases and Hugging Face
  for downloads and metadata, and only when a command that needs them is run.
