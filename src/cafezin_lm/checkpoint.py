"""Read, write and *verify* checkpoints in the reference layout.

Three jobs, in order of importance:

1. **Verification.** ``remote_spec`` reconstructs a published checkpoint's weight
   specification — every tensor name, shape and storage dtype — using only HTTP
   range requests against the safetensors headers. That is a few hundred KB
   instead of several GB, and it is the ground truth this implementation is
   checked against. ``compare_specs`` then reports missing, extra, mis-shaped
   and mis-typed keys, plus the total stored byte count.

2. **Export.** ``save_hf_checkpoint`` writes a trained model back out in the
   reference's own layout: sharded ``safetensors`` plus a
   ``model.safetensors.index.json``, ``config.json`` and
   ``generation_config.json``.

3. **Import.** ``load_hf_checkpoint`` pulls a published checkpoint into this
   implementation.

The reference stores each linear-attention layer's ``A_log`` and gated norm in
float32 and the remaining tensors in bfloat16. Export can preserve the tensors'
native dtypes or apply the reference storage policy.
"""

from __future__ import annotations

import json
import re
import struct
import tempfile
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
import torch.nn as nn

from . import safetensors_io as st_io
from .config import REFERENCE_REPO, REFERENCE_REVISION, ModelConfig
from .safetensors_io import nbytes_of

# The two float32 tensors per linear-attention layer in the reference checkpoint.
FP32_KEY_PATTERNS: tuple[str, ...] = (
    r"\.linear_attn\.A_log$",
    r"\.linear_attn\.norm\.weight$",
)
STORAGE_DTYPE = "BF16"
DEFAULT_SHARD_BYTES = 5 * 2**30  # Hugging Face's default shard budget

HF_RESOLVE = "https://huggingface.co/{repo}/resolve/{revision}/{path}"
HF_API_MODEL = "https://huggingface.co/api/models/{repo}/revision/{revision}"


def ordinal_dtype(name: str) -> str:
    """Storage dtype for one parameter name."""
    return "F32" if any(re.search(p, name) for p in FP32_KEY_PATTERNS) else STORAGE_DTYPE


# ---------------------------------------------------------------------------
# spec / comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceFingerprint:
    """Recorded shape of the published checkpoint.

    Measured from the pinned repository revision using ``cafezin-lm verify``.
    The offline check uses this digest.
    """

    repo: str
    revision: str
    tensors: int
    total_bytes: int
    digest: str


REFERENCE_SPEC = ReferenceFingerprint(
    repo=REFERENCE_REPO,
    revision=REFERENCE_REVISION,
    tensors=738,
    total_bytes=9_319_737_856,
    digest="a8c75f75d93d810d1c4750b9a9acdeee188c69bbb8af5ff5935552d97ded836b",
)


@dataclass
class SpecReport:
    ok: bool
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    shape_mismatch: list[tuple[str, tuple, tuple]] = field(default_factory=list)
    dtype_mismatch: list[tuple[str, str, str]] = field(default_factory=list)
    actual_tensors: int = 0
    expected_tensors: int = 0
    actual_bytes: int = 0
    expected_bytes: int | None = None

    def summary(self) -> str:
        lines = [
            f"tensors:  ours {self.actual_tensors}  reference {self.expected_tensors}",
            f"bytes:    ours {self.actual_bytes:,}"
            + (f"  reference {self.expected_bytes:,}" if self.expected_bytes is not None else ""),
            f"missing:  {len(self.missing)}",
            f"extra:    {len(self.extra)}",
            f"shape:    {len(self.shape_mismatch)}",
            f"dtype:    {len(self.dtype_mismatch)}",
        ]
        if self.expected_bytes is not None and self.actual_bytes != self.expected_bytes:
            lines.append(
                f"byte delta: {self.actual_bytes - self.expected_bytes:+,} "
                "(non-zero means dtype policy or parameter count differs)"
            )
        return "\n".join(lines)

    def details(self, limit: int = 8) -> str:
        out = []
        for label, items in (
            ("missing", self.missing),
            ("extra", self.extra),
            ("shape mismatch", self.shape_mismatch),
            ("dtype mismatch", self.dtype_mismatch),
        ):
            if items:
                out.append(f"  {label}:")
                out.extend(f"    {i}" for i in items[:limit])
                if len(items) > limit:
                    out.append(f"    ... and {len(items) - limit} more")
        return "\n".join(out)


def tied_alias_names(model: nn.Module) -> dict[str, str]:
    """Names of parameters that alias another parameter.

    ``state_dict()`` returns *detached* tensors, so identity must be read from
    ``named_parameters()`` — and with ``remove_duplicate=False``, because the
    default hides every alias after the first. Comparing ``id()`` on state_dict
    entries never matches and would silently export a tied ``lm_head`` a second
    time, adding well over a gigabyte to the file. The trunk name wins; the
    ``lm_head`` alias is reconstructed from the embedding on load.
    """
    groups: dict[int, list[str]] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        groups.setdefault(id(param), []).append(name)
    aliases: dict[str, str] = {}
    for names in groups.values():
        if len(names) < 2:
            continue
        keep = min(names, key=lambda n: (n.startswith("lm_head"), len(n), n))
        aliases.update({name: keep for name in names if name != keep})
    return aliases


def model_spec(
    model: nn.Module,
    drop_tied: bool = True,
    dtype_override: str | None = None,
) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Storage spec of a built model: {name: (safetensors dtype, shape)}.

    Tied parameters are dropped, because the reference stores a tied ``lm_head``
    zero times — it is reconstructed from the embedding on load.

    ``dtype_override`` forces one dtype for every tensor. Passing the model's own
    dtype gives a lossless round trip; the default reproduces the reference's
    mixed float32/bfloat16 policy.
    """
    skip = tied_alias_names(model) if drop_tied else {}
    spec: dict[str, tuple[str, tuple[int, ...]]] = {}
    for name, tensor in model.state_dict().items():
        if name in skip:
            continue
        if dtype_override == "native":
            dtype = st_io.TORCH_TO_ST[tensor.dtype]
        else:
            dtype = dtype_override or ordinal_dtype(name)
        spec[name] = (dtype, tuple(int(s) for s in tensor.shape))
    return spec


def spec_digest(
    spec: dict[str, tuple[str, tuple[int, ...]]], total_bytes: int | None = None
) -> str:
    """Stable SHA-256 over a weight spec.

    Lets verification run without network access: the reference's digest is
    recorded in ``REFERENCE_SPEC``, so an implementation that drifts by one
    tensor name, shape or dtype fails the offline check too.
    """
    import hashlib

    lines = [
        f"{name}|{dtype}|{'x'.join(str(s) for s in shape)}"
        for name, (dtype, shape) in sorted(spec.items())
    ]
    if total_bytes is not None:
        lines.append(f"total_bytes|{total_bytes}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def verify_offline(
    model: nn.Module, fingerprint: ReferenceFingerprint = REFERENCE_SPEC
) -> SpecReport:
    """Check a built model against the recorded fingerprint, with no network use."""
    spec = model_spec(model)
    total = sum(nbytes_of(dtype, shape) for dtype, shape in spec.values())
    digest = spec_digest(spec, total)
    report = SpecReport(
        ok=(len(spec) == fingerprint.tensors and digest == fingerprint.digest),
        actual_tensors=len(spec),
        expected_tensors=fingerprint.tensors,
        actual_bytes=total,
        expected_bytes=fingerprint.total_bytes,
    )
    if digest != fingerprint.digest:
        report.shape_mismatch.append(("<digest>", (digest[:12],), (fingerprint.digest[:12],)))
    return report


def index_spec(index: dict, headers: dict[str, dict]) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Combine a ``weight_map`` with per-shard headers into one flat spec."""
    spec: dict[str, tuple[str, tuple[int, ...]]] = {}
    for name, shard in index["weight_map"].items():
        key = shard.replace(".safetensors", "")
        if key not in headers:
            raise KeyError(f"shard {shard} for tensor {name} was not fetched")
        meta = headers[key][name]
        spec[name] = (meta["dtype"], tuple(meta["shape"]))
    return spec


def compare_specs(
    actual: dict[str, tuple[str, tuple[int, ...]]],
    expected: dict[str, tuple[str, tuple[int, ...]]],
    expected_total_bytes: int | None = None,
    ignore: tuple[str, ...] = (),
) -> SpecReport:
    pattern = re.compile("|".join(ignore)) if ignore else None
    actual = {k: v for k, v in actual.items() if not (pattern and pattern.search(k))}
    expected = {k: v for k, v in expected.items() if not (pattern and pattern.search(k))}

    report = SpecReport(ok=True)
    report.missing = sorted(set(expected) - set(actual))
    report.extra = sorted(set(actual) - set(expected))
    for name in sorted(set(actual) & set(expected)):
        ours, theirs = actual[name], expected[name]
        if ours[1] != theirs[1]:
            report.shape_mismatch.append((name, ours[1], theirs[1]))
        if ours[0] != theirs[0]:
            report.dtype_mismatch.append((name, ours[0], theirs[0]))
    report.actual_tensors = len(actual)
    report.expected_tensors = len(expected)
    report.actual_bytes = sum(nbytes_of(dt, shape) for dt, shape in actual.values())
    report.expected_bytes = expected_total_bytes
    report.ok = not (
        report.missing
        or report.extra
        or report.shape_mismatch
        or report.dtype_mismatch
        or (expected_total_bytes is not None and report.actual_bytes != expected_total_bytes)
    )
    return report


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# ---------------------------------------------------------------------------


def _http(url: str, start: int | None = None, end: int | None = None, timeout: int = 60) -> bytes:
    headers = {"User-Agent": "cafezin-lm/1.0"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{'' if end is None else end}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if start is None:
            return response.read()
        if response.status != 206 or end is None:
            raise ValueError(f"{url}: server did not honor a bounded range request")
        size = end - start + 1
        data = response.read(size)
        if len(data) != size:
            raise ValueError(f"{url}: incomplete range response")
        return data


def fetch_safetensors_header(url: str) -> dict:
    """Read a remote safetensors header over two range requests."""
    (length,) = struct.unpack("<Q", _http(url, 0, 7))
    if not 0 < length <= st_io.MAX_HEADER_BYTES:
        raise ValueError(f"{url}: implausible header length {length}")
    raw = _http(url, 8, 8 + length - 1)
    # the header is space-padded to an 8-byte boundary; json tolerates it
    header = json.loads(raw.decode("utf-8"))
    header.pop("__metadata__", None)
    return header


def hf_files(repo: str = REFERENCE_REPO, revision: str = REFERENCE_REVISION) -> list[str]:
    payload = json.loads(_http(HF_API_MODEL.format(repo=repo, revision=revision)))
    return [s["rfilename"] for s in payload.get("siblings", [])]


def hf_json(path: str, repo: str = REFERENCE_REPO, revision: str = REFERENCE_REVISION) -> dict:
    url = HF_RESOLVE.format(repo=repo, revision=revision, path=path)
    return json.loads(_http(url).decode("utf-8"))


def download_file(
    path: str,
    dest: str | Path,
    repo: str = REFERENCE_REPO,
    revision: str = REFERENCE_REVISION,
) -> Path:
    """Stream one repository file to disk, skipping the work if it is already there."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = HF_RESOLVE.format(repo=repo, revision=revision, path=path)
    request = urllib.request.Request(url, headers={"User-Agent": "cafezin-lm"})
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        partial = Path(tmp) / "download"
        with urllib.request.urlopen(request, timeout=600) as response, partial.open("wb") as fh:
            while chunk := response.read(1 << 20):
                fh.write(chunk)
        partial.replace(dest)
    return dest


@dataclass
class RemoteSpec:
    repo: str
    revision: str
    spec: dict[str, tuple[str, tuple[int, ...]]]
    total_size: int | None
    shards: dict[str, list[str]]
    weight_map: dict[str, str]


def remote_spec(
    repo: str = REFERENCE_REPO,
    revision: str = REFERENCE_REVISION,
    cache_dir: str | Path | None = None,
) -> RemoteSpec:
    """Reconstruct a shipped weight spec without downloading the weights."""
    cache = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "cafezin-lm" / "hf-spec"
    cache.mkdir(parents=True, exist_ok=True)
    tag = repo.replace("/", "__") + "@" + revision.replace("/", "_")
    cached = cache / f"{tag}.json"
    if cached.exists():
        blob = json.loads(cached.read_text(encoding="utf-8"))
        return RemoteSpec(
            repo=blob["repo"],
            revision=blob["revision"],
            spec={k: (v[0], tuple(v[1])) for k, v in blob["spec"].items()},
            total_size=blob["total_size"],
            shards=blob["shards"],
            weight_map=blob["weight_map"],
        )

    files = hf_files(repo, revision)
    safetensors = [f for f in files if f.endswith(".safetensors")]
    if not safetensors:
        raise FileNotFoundError(f"{repo}@{revision} has no safetensors files")

    index_name = "model.safetensors.index.json"
    if index_name in files:
        index = hf_json(index_name, repo, revision)
        weight_map = index["weight_map"]
        total_size = index.get("metadata", {}).get("total_size")
        by_shard: dict[str, list[str]] = {}
        for name, shard in weight_map.items():
            by_shard.setdefault(shard, []).append(name)
    else:
        weight_map = {}
        total_size = None
        by_shard = {safetensors[0]: []}

    headers: dict[str, dict] = {}
    for shard in by_shard:
        url = HF_RESOLVE.format(repo=repo, revision=revision, path=shard)
        headers[shard.replace(".safetensors", "")] = fetch_safetensors_header(url)

    spec: dict[str, tuple[str, tuple[int, ...]]] = {}
    for shard, names in by_shard.items():
        header = headers[shard.replace(".safetensors", "")]
        for name in names or list(header):
            meta = header[name]
            spec[name] = (meta["dtype"], tuple(meta["shape"]))

    result = RemoteSpec(
        repo=repo,
        revision=revision,
        spec=spec,
        total_size=total_size,
        shards=by_shard,
        weight_map=dict(weight_map),
    )
    cached.write_text(
        json.dumps(
            {
                "repo": repo,
                "revision": revision,
                "spec": {k: [v[0], list(v[1])] for k, v in spec.items()},
                "total_size": total_size,
                "shards": by_shard,
                "weight_map": result.weight_map,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return result


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def _cast(tensor: torch.Tensor, st_dtype: str) -> torch.Tensor:
    return tensor.detach().to(st_io.ST_TO_TORCH[st_dtype])


def plan_shards(
    spec: dict[str, tuple[str, tuple[int, ...]]], max_shard_bytes: int
) -> list[list[str]]:
    """Greedy fill of shards in sorted key order — deterministic for a given spec."""
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in sorted(spec):
        size = nbytes_of(spec[name][0], spec[name][1])
        if current and current_bytes + size > max_shard_bytes:
            shards.append(current)
            current, current_bytes = [], 0
        current.append(name)
        current_bytes += size
    if current:
        shards.append(current)
    return shards


def save_hf_checkpoint(
    model: nn.Module,
    out_dir: str | Path,
    cfg: ModelConfig | None = None,
    max_shard_bytes: int = DEFAULT_SHARD_BYTES,
    metadata: dict[str, str] | None = None,
    tokenizer_files: dict[str, bytes] | None = None,
    generation_config: dict | None = None,
    dtype_override: str | None = None,
) -> dict:
    """Write `model` in the reference checkpoint's layout. Returns a manifest.

    Use ``dtype_override="native"`` to preserve the tensor dtypes, or pass a
    SafeTensors dtype such as ``"F32"``. The default uses the reference policy.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = cfg or model.cfg  # type: ignore[attr-defined]

    spec = model_spec(model, dtype_override=dtype_override)
    state = dict(model.state_dict())
    plan = plan_shards(spec, max_shard_bytes)
    total = len(plan)

    weight_map: dict[str, str] = {}
    written: list[str] = []
    for index, names in enumerate(plan, start=1):
        shard_name = f"model-{index:05d}-of-{total:05d}.safetensors"
        tensors = {name: _cast(state[name], spec[name][0]) for name in names}
        st_io.save_file(tensors, out / shard_name, metadata={**(metadata or {}), "format": "pt"})
        written.append(shard_name)
        for name in names:
            weight_map[name] = shard_name
        del tensors

    total_size = sum(nbytes_of(dt, shape) for dt, shape in spec.values())
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    embedding_key = "model.language_model.embed_tokens.weight"
    storage_dtype = str(st_io.ST_TO_TORCH[spec[embedding_key][0]]).removeprefix("torch.")
    replace(cfg, text=replace(cfg.text, dtype=storage_dtype)).save_json(out / "config.json")
    gen = generation_config or {
        "bos_token_id": cfg.text.eos_token_id,
        "do_sample": True,
        "eos_token_id": cfg.text.eos_token_id,
        "pad_token_id": cfg.text.eos_token_id,
        "temperature": 1.0,
        "top_k": 20,
        "top_p": 0.95,
        "min_p": 0.0,
    }
    (out / "generation_config.json").write_text(json.dumps(gen, indent=2), encoding="utf-8")
    for name, blob in (tokenizer_files or {}).items():
        (out / name).write_bytes(blob)

    return {
        "dir": str(out),
        "shards": written,
        "tensors": len(spec),
        "total_size": total_size,
        "weight_map": weight_map,
    }


def read_hf_state(src: str | Path) -> tuple[dict[str, torch.Tensor], dict | None]:
    """Load every tensor in a checkpoint directory (or a single safetensors file)."""
    src = Path(src)
    if src.is_file():
        return st_io.load_file(src), None
    index_path = src / "model.safetensors.index.json"
    state: dict[str, torch.Tensor] = {}
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        by_shard: dict[str, list[str]] = {}
        for name, shard in index["weight_map"].items():
            if Path(shard).name != shard or not shard.endswith(".safetensors"):
                raise ValueError(f"invalid checkpoint shard filename: {shard!r}")
            by_shard.setdefault(shard, []).append(name)
        for shard, names in by_shard.items():
            state.update(st_io.load_file(src / shard, keys=names))
        return state, index
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors or index under {src}")
    for shard in shards:
        tensors = st_io.load_file(shard)
        if state.keys() & tensors.keys():
            raise ValueError(f"{shard}: duplicate tensor names in checkpoint directory")
        state.update(tensors)
    return state, None


def load_hf_checkpoint(
    model: nn.Module, src: str | Path, strict: bool = True, *, assign: bool = False
) -> dict[str, torch.Tensor]:
    """Validate a checkpoint before loading it and preserve tied parameters.

    Assignment materializes weights built on the meta device in storage dtype.
    Use assignment before constructing any optimizer.
    """
    state, _ = read_hf_state(src)
    aliases = tied_alias_names(model)
    for alias, original in aliases.items():
        if alias in state and original not in state:
            state[original] = state[alias]
        state.pop(alias, None)

    expected = model.state_dict()
    missing = expected.keys() - state.keys() - aliases.keys()
    unexpected = state.keys() - expected.keys()
    mismatched = [
        name for name in expected.keys() & state.keys() if expected[name].shape != state[name].shape
    ]
    if mismatched or (strict and (missing or unexpected)):
        raise RuntimeError(
            "checkpoint does not match the model:\n"
            f"  missing:    {sorted(missing)[:8]}\n"
            f"  unexpected: {sorted(unexpected)[:8]}\n"
            f"  shape:      {sorted(mismatched)[:8]}"
        )
    model.load_state_dict(state, strict=False, assign=assign)
    if assign:
        for alias, original in aliases.items():
            module, _, parameter = alias.rpartition(".")
            setattr(model.get_submodule(module), parameter, model.get_parameter(original))
    return state


ASSET_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
)


def checkpoint_assets(directory: str | Path) -> dict[str, bytes]:
    """Copy known tokenizer and processor assets from a checkpoint or corpus."""
    directory = Path(directory)
    return {
        name: (directory / name).read_bytes()
        for name in ASSET_FILES
        if (directory / name).is_file()
    }


def import_config(src_dir: str | Path) -> ModelConfig:
    """Read ``config.json`` out of a checkpoint directory."""
    return ModelConfig.load_json(Path(src_dir) / "config.json")


def shard_layout(index: dict) -> dict[str, int]:
    """Human summary of an index: shard -> tensor count."""
    counts: dict[str, int] = {}
    for shard in index["weight_map"].values():
        counts[shard] = counts.get(shard, 0) + 1
    return counts
