# Qwen3-Coder-Next on this machine

Written by LlamaFit 1.2.3 on 2026-09-10, by `llamafit preset qwen3-coder-next`.

- **Model** Qwen3-Coder-Next, quantised UD-Q4_K_XL
- **File** `D:\Models\Qwen3 Coder Next\Qwen3-Coder-Next-UD-Q4_K_XL.gguf`
- **Placement** moe-offload, planned at 32768 tokens, llama.cpp build 10867
- **Machine** Card: NVIDIA GeForce RTX 4060, 8188 MiB total, 7638 MiB free at plan time.

## Start it

```
start-qwen3-coder-next.cmd
```

Double-clicking it works too. It reads how much card memory is free at that moment, keeps 256 MiB back, and takes the largest context in the table below that still fits.

## Endpoints

| What | Where |
| --- | --- |
| Web interface | http://127.0.0.1:8080/ |
| Chat completions, OpenAI shape | POST http://127.0.0.1:8080/v1/chat/completions |
| Completions | POST http://127.0.0.1:8080/v1/completions |
| Models | http://127.0.0.1:8080/v1/models |
| Health | http://127.0.0.1:8080/health |

The alias to ask for is `qwen3-coder-next`. The server binds loopback only, because llama-server has no authentication of any kind.

## The context it will choose

| Context | The card needs | Free card memory to reach it |
| ---: | ---: | ---: |
| 32768 | 6481 MiB | 6737 MiB |
| 24576 | 6225 MiB | 6481 MiB |
| 16384 | 5969 MiB | 6225 MiB |

The plan behind this table was made when the machine was quieter than it may be when you start it. A configuration that asks for more card memory than is free does not fail on an NVIDIA driver: it starts, pages the overflow into system memory, and runs at a fraction of its speed while the log looks healthy. Stepping down a rung is how that is avoided.

Nothing above the planned 32768 is offered, even when a longer context would fit the card, because system memory and what you asked for went into that choice and the script can measure only the card.

## When this machine changes

| What changed | What the script does |
| --- | --- |
| Something else is holding the card | takes a lower rung, and says which |
| The model file has moved | stops, with the path it looked in, and exits 3 |
| llama.cpp has moved | uses the one on `PATH` if there is one, else exits 3 |
| A different card | starts, and says the layer split was chosen for another |
| llama.cpp has been upgraded and rejects a flag | reports the exit code and says to re-render |

Re-render after any of them with `llamafit preset qwen3-coder-next`. It will not overwrite a file you have edited: each one carries a checksum of itself, and `--force` is what says to discard your changes anyway.

## The command it runs

```
llama-server -m D:\Models\Qwen3 Coder Next\Qwen3-Coder-Next-UD-Q4_K_XL.gguf --alias qwen3-coder-next --host 127.0.0.1 --port 8080 -c <context> -fa on -ngl 99 --n-cpu-moe 47 --fit off -t 16 -tb 16 -b 4096 -ub 2048 --temp 1.0 --top-p 0.95 --top-k 40 --jinja -np 1
```

<!-- llamafit-preset-stamp: faef3532a64ec27f1475c5e39c5bbd2ee4fa7edde57027450512b3d7e5e72f0a -->
