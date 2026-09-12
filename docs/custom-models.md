# Custom models

You do not need a pull request to make LlamaFit consider a model it does not ship, or to
override an entry you disagree with. Put entries in a custom models file; they are merged with
the bundled catalog every time LlamaFit runs.

## Where the file lives

| OS | Path |
|---|---|
| Windows | `%LOCALAPPDATA%\llamafit\custom_models.yaml` |
| macOS | `~/Library/Application Support/llamafit/custom_models.yaml` |
| Linux | `~/.local/share/llamafit/custom_models.yaml` |

`LLAMAFIT_CUSTOM_MODELS` names any other path, and `LLAMAFIT_HOME` moves the whole data
directory, putting the file at `<LLAMAFIT_HOME>/data/custom_models.yaml`. No command prints the
path in use; the table above and those two variables are the whole rule.

The file does not have to exist. When it does, `llamafit catalog validate` checks it along with
the bundled files, and every command loads it.

## Format

The same YAML schema as the bundled catalog (see [catalog.md](catalog.md)), as a list of
entries:

```yaml
- id: my-org-model-7b
  name: My Model 7B
  vendor: My Org
  family: mymodel
  release_date: 2026-05-01
  license: {spdx: Apache-2.0, url: https://huggingface.co/my-org/my-model-7b/blob/main/LICENSE}
  params: {total_b: 7, active_b: 7}
  architecture: {class: dense, gguf_arch: llama}
  context: {native: 32768}
  capabilities: [coding, tools]
  use_cases: [coding, general]
  quality: {baseline: 70, benchmarks: [{name: HumanEval, score: 72.0, source: https://...}]}
  sources:
    - repo: my-org/my-model-7b-GGUF
      kind: gguf
      trust: community
      quants:
        - {name: Q4_K_M}
```

A quant needs only its `name`, and an extra only its `file`. `llamafit catalog refresh --model
my-org-model-7b` fills in the file names, sizes, checksums, bits per weight and GGUF facts, and
writes them to a `.facts.json` file beside your YAML rather than into it, which is the same
split the bundled catalog uses, described in [catalog.md](catalog.md#where-the-facts-live). A quant name
must not be claimed by two of a model's sources, because that file keys on the name alone.

A file that is not on Hugging Face is described with `kind: local` and the path to the GGUF
file:

```yaml
  sources:
    - kind: local
      path: D:/models/my-model-7b-Q4_K_M.gguf
      quants:
        - {name: Q4_K_M}
```

Such an entry is accepted and validated, but nothing reads the file yet: `refresh` queries
`gguf` sources only, so a local source's size and GGUF facts stay empty until a later phase
reads it from disk. `repo_path`, which narrows a `gguf` source to one directory inside its
repository, is refused on a `local` source, and `path` is refused on a `gguf` one; each kind
takes its own field.

## Precedence

- An entry whose `id` matches a bundled entry **replaces** it completely. To change one field,
  copy the bundled entry (`llamafit catalog show ID --yaml`), edit it, and keep it in the
  custom file.
- New ids are appended.
- The custom file is validated with the same rules; a problem in it is reported and the
  bundled catalog is still used.

## When to send it upstream

If the model is public and your entry has sources, open a pull request with it in
`src/llamafit/data/catalog/`; everyone benefits, and `refresh --check` keeps it current.
