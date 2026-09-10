# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""What ``install model`` is about to do: which files, from where, to where, how big.

Everything a download needs decided before it starts is decided here, from the catalog
alone and without one network request, so ``--dry-run`` works on a train. Four of those
decisions matter more than the rest.

**A split model is one thing.** The catalog lists a quant's shards; a person asked for a
model. The plan carries every shard, every extra the entry declares — a vision projector,
draft weights — and one total, and nothing calls the model installed until each of them has
arrived and been checked. Ending up with three files of four and no warning is the failure
this shape exists to prevent.

**Nothing is downloaded that cannot be checked.** A quant with no checksums in the catalog
is refused by default, naming ``llamafit catalog refresh`` as the fix, because the whole
point of those checksums is that a wrong file does not announce itself.
``--allow-unverified`` exists for a model somebody curated themselves, and says out loud
what it is giving up.

**The disk is checked before the first byte, not after.** :func:`check_disk_space` does the
subtraction the catalog already has both sides of. A download that fills a volume at ninety
percent has cost hours and left nothing usable behind.

**A shard's own size is not in the catalog, and is not invented here.** A quant records one
total for the whole set, which is exactly the number the disk check and the "how big is
this" line want, and no number at all for shard three of four. Rather than apportion the
total and pretend, a shard's size is left unset and the engine learns it from the server's
own ``Content-Range`` before it allocates anything. A size that was invented here would be
the size the part file was created at, and a wrong one would corrupt the file in a way only
the checksum would catch, hours later.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from llamafit.download.errors import DiskSpaceError, DownloadError
from llamafit.download.state import part_path_for, state_path_for
from llamafit.i18n import _, ngettext
from llamafit.models.catalog import CatalogModel, Extra, ModelSource, Quant
from llamafit.paths import get_paths
from llamafit.units import format_bytes

DISK_HEADROOM_BYTES = 2 * 1024 * 1024 * 1024
"""Room left over after the download: 2 GiB.

A volume filled to its last byte is a volume where nothing else works — no log, no
temporary file, on Windows no page file growth — and the person who filled it was trying
to run a model, which needs some of those.
"""

MANIFEST_NAME = "llamafit-install.json"
"""Written into a model's directory only once every file has arrived and been checked.

Its presence is what "this model is installed" means, and it is also the only place a
shard's real size is recorded, since the catalog holds one size for a whole quant. A
directory holding three shards of four has no manifest, and nothing tells a user it has
the model.
"""


def hugging_face_url(repo: str, path: str) -> str:
    """The direct download URL for ``path`` inside ``repo``.

    Deliberately the same string :meth:`llamafit.catalog.hf.HttpHfClient.file_url` builds,
    and a test asserts the two never drift apart. It is written again here so that
    planning a download does not have to construct an HTTP client, which would open a
    connection pool for a command that may only be printing what it would do.
    """
    return f"https://huggingface.co/{repo}/resolve/main/{path}"


@dataclass(frozen=True)
class FileRequest:
    """One file to fetch.

    Attributes:
        name: The file's bare name, which is both what a reader is shown and what it is
            called on disk. A catalog file name is a path inside the publishing
            repository and often carries a directory; the local copy is flat, the same
            way :func:`llamafit.services.plan.launch_paths` already assumes.
        repo_path: The file's path inside the repository, as the catalog records it.
        url: Where to fetch it from.
        target: Where the finished file goes.
        size: How many bytes it has, when the catalog knows: a single-file quant's total,
            or an extra's own size. ``None`` for one shard of a split quant, whose size
            the catalog does not record and which the engine learns from the server.
        sha256: Its checksum, or ``None`` when the catalog has not been refreshed.
        role: ``weights`` for a quant's own files, or the extra's role for the rest.
        known_size: What a previous, verified install recorded for this file, read from
            the manifest. It is how a shard is recognised as already here.
    """

    name: str
    repo_path: str
    url: str
    target: Path
    size: int | None
    sha256: str | None
    role: str
    known_size: int | None = None

    @property
    def part(self) -> Path:
        """The half-finished copy of this file."""
        return part_path_for(self.target)

    @property
    def state(self) -> Path:
        """The record of which chunks of this file have arrived."""
        return state_path_for(self.target)

    @property
    def expected_size(self) -> int | None:
        """The size to check against before any server has been asked, if any is known."""
        return self.size if self.size is not None else self.known_size

    def already_present(self) -> bool:
        """Whether the finished file is here at a size somebody has vouched for.

        A file of a size nobody recorded is not treated as present: the manifest is
        written only after a checksum matched, so a lone GGUF that no install put there
        gets fetched and checked rather than trusted because it has the right name.
        """
        expected = self.expected_size
        if expected is None:
            return False
        try:
            return self.target.is_file() and self.target.stat().st_size == expected
        except OSError:
            return False

    def bytes_on_disk(self) -> int:
        """How many of this file's bytes are finished and in place."""
        return self.expected_size or 0 if self.already_present() else 0


@dataclass(frozen=True)
class DownloadPlan:
    """Everything ``install model`` is about to do.

    Attributes:
        model_id: The catalog id.
        model_name: The model's display name.
        quant: The quantisation being fetched.
        repo: The repository publishing it.
        directory: Where the files will go.
        files: Every file, weights first, then extras in catalog order.
        total_bytes: What the whole model weighs, from the catalog: exact, whatever the
            individual shards turn out to be.
        unverifiable: The names of files the catalog holds no checksum for.
    """

    model_id: str
    model_name: str
    quant: str
    repo: str
    directory: Path
    files: tuple[FileRequest, ...]
    total_bytes: int
    unverifiable: tuple[str, ...] = field(default=())

    @property
    def missing(self) -> tuple[FileRequest, ...]:
        """The files not already on disk at a size somebody vouched for."""
        return tuple(file for file in self.files if not file.already_present())

    @property
    def missing_bytes(self) -> int:
        """What is still to be fetched: the total, less whatever is already in place.

        Deliberately the pessimistic figure. A resumed download needs less than this,
        because a part file already holds some of it, and a disk check that counted those
        bytes would let a transfer start that could not finish if the part file turned out
        not to be resumable after all.
        """
        return max(0, self.total_bytes - sum(file.bytes_on_disk() for file in self.files))

    @property
    def manifest_path(self) -> Path:
        """Where the "this model is complete" record goes."""
        return self.directory / MANIFEST_NAME

    def is_installed(self) -> bool:
        """Whether every file is here at its recorded size and the manifest is with them."""
        return self.manifest_path.is_file() and not self.missing


def read_manifest(directory: Path) -> Mapping[str, int]:
    """What a previous install recorded in ``directory``: file name to size in bytes.

    A manifest that is missing, unreadable or the wrong shape is treated as absent rather
    than as a failure. The worst that follows is a file fetched again, which is slow and
    correct; refusing to run until a user deletes a file by hand never is.
    """
    try:
        document = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(document, dict):
        return {}
    files = document.get("files")
    if not isinstance(files, list):
        return {}
    sizes: dict[str, int] = {}
    for entry in files:
        if not isinstance(entry, dict):
            continue
        name, size = entry.get("name"), entry.get("bytes")
        if isinstance(name, str) and isinstance(size, int) and size >= 0:
            sizes[name] = size
    return sizes


@dataclass(frozen=True)
class DiskCheck:
    """The subtraction that has to come out positive before anything starts.

    Attributes:
        directory: The directory the files would go in.
        needed: Bytes still to fetch.
        free: Bytes free on that volume.
        headroom: Bytes deliberately left unused.
    """

    directory: Path
    needed: int
    free: int
    headroom: int

    @property
    def short_by(self) -> int:
        """How many bytes are missing, or zero when there are enough."""
        return max(0, self.needed + self.headroom - self.free)

    @property
    def ok(self) -> bool:
        """Whether the download can finish on this volume."""
        return self.short_by == 0

    def raise_if_short(self) -> None:
        """Refuse the download when the volume cannot hold it.

        Raises:
            DiskSpaceError: If there is not enough room, saying by how much.
        """
        if self.ok:
            return
        raise DiskSpaceError(
            _(
                "%(needed)s is needed for this model but %(path)s has only %(free)s free, "
                "%(short)s short of what is wanted with %(headroom)s left spare."
            )
            % {
                "needed": format_bytes(self.needed),
                "path": str(self.directory),
                "free": format_bytes(self.free),
                "short": format_bytes(self.short_by),
                "headroom": format_bytes(self.headroom),
            },
            hint=_(
                "Free some space, or choose another directory with --dir, or a smaller "
                "quantisation with --quant."
            ),
        )


def check_disk_space(
    plan: DownloadPlan, *, headroom: int = DISK_HEADROOM_BYTES, free_bytes: int | None = None
) -> DiskCheck:
    """Measure the target volume against what the plan still has to fetch.

    The nearest existing ancestor of the target directory is measured, since the directory
    will not exist on a first install and a path that does not exist has no volume to ask
    about.

    Args:
        plan: What is about to be downloaded.
        headroom: How much to leave spare.
        free_bytes: The free figure to use instead of measuring, for tests.

    Returns:
        The check, which the caller turns into a refusal with
        :meth:`DiskCheck.raise_if_short`.

    Raises:
        DownloadError: If the volume cannot be measured at all.
    """
    if free_bytes is None:
        anchor = plan.directory
        while not anchor.exists() and anchor.parent != anchor:
            anchor = anchor.parent
        try:
            free_bytes = shutil.disk_usage(anchor).free
        except OSError as exc:
            raise DownloadError(
                _("could not measure the free space on %(path)s: %(error)s")
                % {"path": str(anchor), "error": exc},
                hint=_("Check that the download directory's volume is available."),
            ) from exc
    return DiskCheck(
        directory=plan.directory,
        needed=plan.missing_bytes,
        free=free_bytes,
        headroom=headroom,
    )


def _source_for(model: CatalogModel, quant_name: str | None) -> tuple[ModelSource, Quant]:
    """The source and quant to fetch, refusing anything the catalog cannot describe.

    Raises:
        DownloadError: If the model publishes nothing fetchable, if the named quant does
            not exist, or if the entry that does exist has no file list yet.
    """
    fetchable = [source for source in model.sources if source.kind == "gguf" and source.repo]
    if not fetchable:
        raise DownloadError(
            _("%(model)s has no downloadable source in the catalog.") % {"model": model.id},
            hint=_("Only Hugging Face sources can be installed; this entry has none."),
        )
    names: list[str] = []
    for source in fetchable:
        for quant in source.quants:
            names.append(quant.name)
            if quant_name is not None and quant.name.casefold() != quant_name.casefold():
                continue
            if not quant.files:
                raise DownloadError(
                    _("the catalog has no file list for %(model)s %(quant)s.")
                    % {"model": model.id, "quant": quant.name},
                    hint=_("Run `llamafit catalog refresh` to fill in the file names."),
                )
            return source, quant
    if quant_name is None:
        raise DownloadError(
            _("%(model)s publishes no quantisations.") % {"model": model.id},
            hint=_("Run `llamafit catalog refresh` and try again."),
        )
    raise DownloadError(
        _("%(model)s does not publish a %(quant)s quantisation.")
        % {"model": model.id, "quant": quant_name},
        hint=_("It publishes: %(quants)s") % {"quants": ", ".join(dict.fromkeys(names))},
    )


def _quant_total(quant: Quant) -> int:
    """The quant's recorded size.

    Raises:
        DownloadError: If the catalog has no size for it, since a download whose size
            nobody knows cannot be checked against a disk before it starts, which is the
            one check that has to happen first.
    """
    if quant.bytes_ is None:
        raise DownloadError(
            _("the catalog does not record how big %(quant)s is.") % {"quant": quant.name},
            hint=_("Run `llamafit catalog refresh` to fill in the sizes."),
        )
    return quant.bytes_


def _requests_for_quant(
    source: ModelSource, quant: Quant, directory: Path, known: Mapping[str, int]
) -> tuple[list[FileRequest], list[str]]:
    """One request per shard, and the names of any the catalog cannot vouch for."""
    repo = source.repo or ""
    total = _quant_total(quant)
    single = len(quant.files) == 1
    checksums = quant.sha256 if len(quant.sha256) == len(quant.files) else []
    requests: list[FileRequest] = []
    unverifiable: list[str] = []
    for index, repo_path in enumerate(quant.files):
        name = PurePosixPath(repo_path).name
        sha256 = checksums[index] if checksums else None
        if sha256 is None:
            unverifiable.append(name)
        requests.append(
            FileRequest(
                name=name,
                repo_path=repo_path,
                url=hugging_face_url(repo, repo_path),
                target=directory / name,
                size=total if single else None,
                sha256=sha256,
                role="weights",
                known_size=known.get(name),
            )
        )
    return requests, unverifiable


def _requests_for_extras(
    extras: Sequence[Extra], source: ModelSource, directory: Path, known: Mapping[str, int]
) -> tuple[list[FileRequest], list[str], int]:
    """One request per extra the entry declares, and what they weigh between them.

    An extra the catalog has not sized yet is left out rather than fetched blind: it is a
    projector or a draft model, useful but not the thing the person asked for, and adding
    an unknown quantity to a disk check is how a download runs out of room at the end.
    """
    repo = source.repo or ""
    requests: list[FileRequest] = []
    unverifiable: list[str] = []
    total = 0
    for extra in extras:
        if extra.bytes_ is None:
            continue
        name = PurePosixPath(extra.file).name
        if extra.sha256 is None:
            unverifiable.append(name)
        total += extra.bytes_
        requests.append(
            FileRequest(
                name=name,
                repo_path=extra.file,
                url=hugging_face_url(repo, extra.file),
                target=directory / name,
                size=extra.bytes_,
                sha256=extra.sha256,
                role=extra.role,
                known_size=known.get(name),
            )
        )
    return requests, unverifiable, total


def build_plan(
    model: CatalogModel,
    *,
    quant_name: str | None = None,
    directory: Path | None = None,
    with_extras: bool = True,
) -> DownloadPlan:
    """Work out every file ``install model`` would fetch, and where each would go.

    Args:
        model: The catalog entry.
        quant_name: The quantisation to fetch, or ``None`` for the first the entry
            publishes. Matched without regard to case, the way a model id is.
        directory: Where to put the files, or ``None`` for the configured downloads
            directory with the model's id underneath it.
        with_extras: Whether to fetch the projector, draft weights and the rest.

    Returns:
        The plan, with every file's URL, target, checksum and whatever is known of its
        size.

    Raises:
        DownloadError: If the catalog cannot describe the download: no fetchable source,
            no such quantisation, no file list, or no size.
    """
    source, quant = _source_for(model, quant_name)
    root = directory if directory is not None else get_paths().downloads_dir / model.id
    known = read_manifest(root)
    requests, unverifiable = _requests_for_quant(source, quant, root, known)
    total = _quant_total(quant)
    if with_extras:
        extra_requests, extra_unverifiable, extra_bytes = _requests_for_extras(
            source.extras, source, root, known
        )
        requests.extend(extra_requests)
        unverifiable.extend(extra_unverifiable)
        total += extra_bytes
    return DownloadPlan(
        model_id=model.id,
        model_name=model.name,
        quant=quant.name,
        repo=source.repo or "",
        directory=root,
        files=tuple(requests),
        total_bytes=total,
        unverifiable=tuple(unverifiable),
    )


def refuse_unverifiable(plan: DownloadPlan) -> None:
    """Stop a download of files no checksum can vouch for.

    Raises:
        DownloadError: If the catalog holds no checksum for one or more of the files.
    """
    if not plan.unverifiable:
        return
    raise DownloadError(
        ngettext(
            "the catalog holds no checksum for %(files)s.",
            "the catalog holds no checksums for %(files)s.",
            len(plan.unverifiable),
        )
        % {"files": ", ".join(plan.unverifiable)},
        hint=_(
            "Run `llamafit catalog refresh` to fetch them, or pass --allow-unverified to "
            "download anyway and check the files yourself."
        ),
    )
