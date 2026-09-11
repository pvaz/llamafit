"""Fakes the placement tests stand on: a budget function and the reference machine.

The planner takes its budget as an injected callable, which is what lets these exist. Two
of them are useful for different questions.

:class:`SectionEightBudget` is section 8 of the design implemented straightforwardly, so a
test can ask what the planner would really choose on a real machine for a real model, and
so the answers can be held against the reference machine's own recorded buffer sizes. It
is a *test* implementation and deliberately independent of whatever ``llamafit.budget``
turns into: if the two ever disagree, that is a conversation worth having rather than a
tautology.

:class:`ScriptedBudget` answers a different question. Search order is easiest to test when
the tester decides exactly which configurations fit, so this one takes a rule and applies
it, and every call is recorded for assertions about what was tried and in what order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from llamafit.models.catalog import CatalogModel, Quant
from llamafit.models.gguf import GgufFacts
from llamafit.models.host import Cpu, Disk, Gpu, Host, Memory
from llamafit.models.plan import Budget, BudgetLine, Verdict
from llamafit.placement.modes import PlacementSettings

MIB = 1024**2
GIB = 1024**3

# Section 8.2's compute-buffer model: base MiB and MiB per 1K tokens of context, by -ub.
COMPUTE_BASE_MIB = {256: 1090, 512: 1090, 1024: 1240, 2048: 2080}
COMPUTE_PER_1K_MIB = {256: 1.5, 512: 1.5, 1024: 3.0, 2048: 8.0}
COMPUTE_PER_1K_MIB_LONG = {256: 1.5, 512: 1.5, 1024: 3.0, 2048: 25.0}
LONG_CONTEXT_TOKENS = 65536

CUDA_CONTEXT_BYTES = 300 * MIB
PROCESS_OVERHEAD_BYTES = 1 * GIB
VRAM_RESERVE_BYTES = 256 * MIB
PROJECTOR_COMPUTE_BYTES = 248 * MIB


def reference_host() -> Host:
    """The machine in ``docs/calibration/2026-09-09-reference-machine.md``.

    An i9-14900KF with 8 performance and 16 efficiency cores, 128 GB of DDR5 and an
    8 GB RTX 4060 with a 550 MiB desktop on it, which is the machine every measured
    number in this repository was taken on.
    """
    return Host(
        os="windows",
        os_version="11 (26200)",
        arch="x86_64",
        cpu=Cpu(
            model="Intel(R) Core(TM) i9-14900KF",
            physical_cores=24,
            logical_cores=32,
            performance_cores=8,
            isa=["avx2"],
        ),
        memory=Memory(
            total_bytes=128 * GIB,
            available_bytes=100 * GIB,
            type="DDR5",
            speed_mts=4200,
            modules=2,
            channels=2,
            bandwidth_gbps=57.0,
            bandwidth_source="measured",
        ),
        gpus=[
            Gpu(
                index=0,
                vendor="nvidia",
                name="NVIDIA GeForce RTX 4060",
                vram_total_bytes=8188 * MIB,
                vram_used_bytes=550 * MIB,
                vram_source="measured",
                bandwidth_gbps=272.0,
                compute_tflops_fp16=60.4,
                backend_hint="cuda",
                driver="610.88",
            )
        ],
        disks=[Disk(path="D:\\", free_bytes=900 * GIB, total_bytes=2000 * GIB)],
        scanned_at=datetime(2026, 9, 9, 20, 0, tzinfo=timezone.utc),
    )


def cpu_only_host(*, ram_available: int = 32 * GIB) -> Host:
    """A machine with no graphics card at all."""
    host = reference_host()
    return host.model_copy(
        update={
            "gpus": [],
            "memory": host.memory.model_copy(update={"available_bytes": ram_available}),
        }
    )


def verdict_for(
    vram_required: int, vram_available: int, ram_required: int, ram_available: int, mode: str
) -> Verdict:
    """Section 8.3's verdict, with 8.4's paging rule and the cap on CPU-only placements."""
    if ram_required > ram_available:
        return "does-not-fit"
    if vram_available and vram_required > vram_available:
        # Section 8.4: the driver pages instead of failing, so this is named, not rejected.
        return "too-tight"
    utilisation = max(
        ram_required / ram_available if ram_available else 0.0,
        vram_required / vram_available if vram_available else 0.0,
    )
    if utilisation > 0.95:
        return "does-not-fit"
    if utilisation > 0.85:
        return "tight"
    if utilisation > 0.65:
        return "fits"
    return "fits" if mode == "cpu" else "comfortable"


@dataclass
class SectionEightBudget:
    """Section 8's memory budget, implemented for the tests that need a real one.

    Attributes:
        projector_bytes: Size of the vision projector file, when the model has one.
        vram_available_override: Free VRAM to use instead of the host's, so a test can
            put the machine under pressure without inventing a whole new one.
        calls: Every settings value costed, in order.
    """

    projector_bytes: int = 862 * MIB
    vram_available_override: int | None = None
    calls: list[PlacementSettings] = field(default_factory=list)

    def __call__(
        self, model: CatalogModel, quant: Quant, host: Host, settings: PlacementSettings
    ) -> Budget:
        """Cost one configuration."""
        self.calls.append(settings)
        facts = quant.gguf_facts
        assert facts is not None, "the section 8 fake needs refreshed facts"
        layers = facts.n_layer or 1
        gpu_layers = min(settings.gpu_layers, layers)
        gpu_share = gpu_layers / layers
        cpu_moe = min(settings.cpu_moe_layers or 0, layers)
        expert_share_on_gpu = 0.0 if gpu_layers == 0 else (layers - cpu_moe) / layers

        lines: list[BudgetLine] = []

        dense_on_gpu = int(facts.bytes_dense_block_weights * gpu_share)
        lines.append(_line("dense-weights", "vram", dense_on_gpu, exact=True))
        lines.append(
            _line(
                "dense-weights",
                "ram",
                facts.bytes_dense_block_weights - dense_on_gpu,
                exact=True,
            )
        )
        experts_on_gpu = int(facts.bytes_expert_weights * expert_share_on_gpu)
        lines.append(_line("expert-weights", "vram", experts_on_gpu, exact=True))
        lines.append(
            _line("expert-weights", "ram", facts.bytes_expert_weights - experts_on_gpu, exact=True)
        )
        head = facts.bytes_output_head + facts.bytes_global_weights
        lines.append(_line("output-head", "vram" if gpu_layers else "ram", head, exact=True))
        lines.append(_line("token-embedding", "ram", facts.bytes_token_embd, exact=True))
        lines.append(_line("lazy-tables", "disk", facts.bytes_lazy_tables, exact=True))

        per_token = facts.kv_bytes_per_token_f16 or 0
        if settings.kv_type != "f16":
            per_token //= 2
        lines.append(
            _line(
                "kv-cache",
                "vram" if gpu_layers else "ram",
                per_token * settings.context,
                exact=False,
            )
        )
        if facts.recurrent_state_bytes:
            lines.append(
                _line(
                    "recurrent-state",
                    "vram" if gpu_layers else "ram",
                    facts.recurrent_state_bytes,
                    exact=True,
                )
            )
        if gpu_layers:
            lines.append(_line("compute-buffer", "vram", compute_buffer(settings), exact=False))
            lines.append(_line("cuda-context", "vram", CUDA_CONTEXT_BYTES, exact=False))
        lines.append(
            _line("output-buffer", "ram", (facts.n_vocab or 0) * 4 * settings.batch, exact=False)
        )
        if settings.projector_pool is not None:
            lines.append(
                _line(
                    "vision-projector",
                    settings.projector_pool,
                    self.projector_bytes + PROJECTOR_COMPUTE_BYTES,
                    exact=False,
                )
            )
        lines.append(_line("process-overhead", "ram", PROCESS_OVERHEAD_BYTES, exact=False))

        vram_required = sum(line.bytes_ for line in lines if line.pool == "vram")
        ram_required = sum(line.bytes_ for line in lines if line.pool == "ram")
        vram_available = self._vram_available(host)
        ram_available = host.memory.available_bytes
        return Budget(
            lines=tuple(lines),
            vram_required=vram_required,
            ram_required=ram_required,
            vram_available=vram_available,
            ram_available=ram_available,
            vram_utilisation=(vram_required / vram_available) if vram_available else None,
            ram_utilisation=ram_required / ram_available if ram_available else 0.0,
            verdict=verdict_for(
                vram_required, vram_available, ram_required, ram_available, settings.mode
            ),
        )

    def _vram_available(self, host: Host) -> int:
        if self.vram_available_override is not None:
            return self.vram_available_override
        free = host.vram_available_bytes
        return 0 if free is None else max(free - VRAM_RESERVE_BYTES, 0)


def compute_buffer(settings: PlacementSettings) -> int:
    """Section 8.2's compute buffer for one micro-batch and context."""
    base = COMPUTE_BASE_MIB.get(settings.micro_batch, 1090)
    table = (
        COMPUTE_PER_1K_MIB_LONG if settings.context > LONG_CONTEXT_TOKENS else COMPUTE_PER_1K_MIB
    )
    per_1k = table.get(settings.micro_batch, 1.5)
    return int((base + per_1k * settings.context / 1024) * MIB)


@dataclass
class ScriptedBudget:
    """A budget whose verdict a test decides, for questions about the search itself.

    Attributes:
        rule: Given the settings, the verdict. Everything else in the budget is filled in
            consistently with it so nothing downstream has to special-case a fake.
        vram_available: What to report as free on the card.
        ram_available: What to report as free in system memory.
        calls: Every settings value costed, in order, which is the search order itself.
    """

    rule: Callable[[PlacementSettings], Verdict]
    vram_available: int = 8 * GIB
    ram_available: int = 64 * GIB
    calls: list[PlacementSettings] = field(default_factory=list)

    def __call__(
        self, model: CatalogModel, quant: Quant, host: Host, settings: PlacementSettings
    ) -> Budget:
        """Cost one configuration by asking the rule."""
        self.calls.append(settings)
        verdict = self.rule(settings)
        share = {
            "comfortable": 0.5,
            "fits": 0.8,
            "tight": 0.9,
            "too-tight": 1.2,
            "does-not-fit": 1.5,
        }[verdict]
        vram_required = int(self.vram_available * share)
        ram_required = int(self.ram_available * 0.5)
        if verdict == "does-not-fit":
            vram_required = int(self.vram_available * 0.9)
            ram_required = int(self.ram_available * 1.5)
        return Budget(
            lines=(_line("weights", "vram", vram_required, exact=True),),
            vram_required=vram_required,
            ram_required=ram_required,
            vram_available=self.vram_available,
            ram_available=self.ram_available,
            vram_utilisation=vram_required / self.vram_available,
            ram_utilisation=ram_required / self.ram_available,
            verdict=verdict,
        )


def _line(component: str, pool: str, size: int, *, exact: bool) -> BudgetLine:
    return BudgetLine.model_validate(
        {"component": component, "pool": pool, "bytes": max(size, 0), "exact": exact}
    )


def make_facts(
    *,
    layers: int = 8,
    experts: int | None = None,
    dense_bytes: int = 4 * GIB,
    expert_bytes: int = 0,
    shared_expert_bytes: int = 0,
    lazy_bytes: int = 0,
    kv_per_token: int = 128 * 1024,
    recurrent_bytes: int | None = None,
) -> GgufFacts:
    """Architecture facts with the byte buckets adding up, which the model insists on."""
    return GgufFacts(
        arch="test",
        n_layer=layers,
        n_embd=4096,
        n_vocab=128000,
        n_head=32,
        n_head_kv=8,
        head_dim=128,
        value_head_dim=128,
        attention_layers=layers,
        attention_layers_source="tensors",
        context_length=262144,
        n_expert=experts,
        n_expert_used=8 if experts else None,
        has_shared_experts=bool(experts),
        bytes_expert_weights=expert_bytes,
        bytes_dense_block_weights=dense_bytes,
        bytes_shared_expert_weights=shared_expert_bytes,
        bytes_lazy_tables=lazy_bytes,
        bytes_total=dense_bytes + expert_bytes + shared_expert_bytes + lazy_bytes,
        kv_bytes_per_token_f16=kv_per_token,
        recurrent_state_bytes=recurrent_bytes,
    )


def make_model(
    *,
    model_id: str = "test-model",
    moe: bool = False,
    native: int = 262144,
    kv_types: Sequence[str] = (),
    quirks: Sequence[str] = (),
    requires: dict[str, str] | None = None,
    projector: bool = False,
    facts: GgufFacts | None = None,
    temp: float | None = None,
    reasoning_format: str | None = None,
    thinking_toggle: str | None = None,
) -> tuple[CatalogModel, Quant]:
    """A catalog entry and its one quant, built for whatever a test is about."""
    quant = Quant.model_validate(
        {"name": "Q4_K_M", "files": ["test.gguf"], "bytes": 4 * GIB, "bpw": 4.5}
    )
    if facts is not None:
        quant = quant.model_copy(update={"gguf_facts": facts})
    source: dict[str, object] = {
        "repo": "someone/test-GGUF",
        "kind": "gguf",
        "trust": "community",
        "quants": [quant.model_dump(by_alias=True)],
    }
    if projector:
        source["extras"] = [{"role": "mmproj", "file": "mmproj-F16.gguf", "bytes": 862 * MIB}]
    model = CatalogModel.model_validate(
        {
            "id": model_id,
            "name": "Test Model",
            "vendor": "Nobody",
            "family": "test",
            "release_date": date(2026, 1, 1),
            "license": {"spdx": "Apache-2.0", "url": "https://example.invalid/LICENSE"},
            "params": {"total_b": 8.0, "active_b": 1.0 if moe else 8.0},
            "architecture": {"class": "moe" if moe else "dense", "gguf_arch": "test"},
            "context": {"native": native},
            "capabilities": ["vision"] if projector else [],
            "use_cases": ["general"],
            "quality": {"baseline": 70},
            "sampling": {"temp": temp} if temp is not None else {},
            "chat_template": {
                "reasoning_format": reasoning_format,
                "thinking_toggle": thinking_toggle,
            },
            "llama_cpp": {
                "kv_types_allowed": list(kv_types),
                "quirks": list(quirks),
                "requires": requires or {},
            },
            "sources": [source],
        }
    )
    return model, model.sources[0].quants[0]
