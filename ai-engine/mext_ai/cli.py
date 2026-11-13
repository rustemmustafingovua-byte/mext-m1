"""Command line for the AI engine: ``python -m mext_ai <cmd>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

from .memory import HAS_MLX, MB

log = logging.getLogger("mext_ai.cli")

DEFAULT_STATE = Path.home() / ".mext" / "ai"
DEFAULT_SOCKET = Path.home() / ".mext" / "ai.sock"


def _default_workspace() -> Path:
    return Path.cwd()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mext-ai",
        description="MEXT-M1 AI engine: MLX layer streaming + zero-RAM LanceDB RAG",
    )
    parser.add_argument("--log-level", default="INFO", help="DEBUG/INFO/WARNING")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="run the JSON-RPC engine (spawned by `mext start`)")
    serve.add_argument("--workspace", default=str(_default_workspace()))
    serve.add_argument("--socket", default=str(DEFAULT_SOCKET))
    serve.add_argument("--state-dir", default=str(DEFAULT_STATE))
    serve.add_argument("--ram-budget", type=int, default=2 * 1024 * MB)
    serve.add_argument("--model", default="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    serve.add_argument("--lookahead", type=int, default=2)
    serve.add_argument("--nprobe", type=int, default=3)
    serve.add_argument("--demo", action="store_true", help="run a background generation loop")
    serve.add_argument(
        "--synthetic-moe",
        type=int,
        default=0,
        help="turn the last N synthetic layers into router-guided MoE blocks",
    )
    serve.add_argument("--no-neural", action="store_true")
    serve.add_argument("--no-rag", action="store_true")

    index = sub.add_parser("index", help="build/refresh the LanceDB index")
    index.add_argument("--workspace", default=str(_default_workspace()))
    index.add_argument("--state-dir", default=str(DEFAULT_STATE))
    index.add_argument("--force", action="store_true")
    index.add_argument("--max-files", type=int, default=2000)
    index.add_argument("--json", action="store_true")

    query = sub.add_parser("query", help="query the index (no daemon needed)")
    query.add_argument("text")
    query.add_argument("--workspace", default=str(_default_workspace()))
    query.add_argument("--state-dir", default=str(DEFAULT_STATE))
    query.add_argument("-k", type=int, default=8)
    query.add_argument("--json", action="store_true")

    selftest = sub.add_parser("selftest", help="verify MLX + LanceDB + prefetch end to end")
    selftest.add_argument("--json", action="store_true")

    prepare = sub.add_parser("prepare", help="how to prepare real MLX weights for streaming")
    prepare.add_argument("--model", default="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    prepare.add_argument("--state-dir", default=str(DEFAULT_STATE))

    gguf = sub.add_parser("gguf", help="parse/inspect GGUF checkpoints and warm layer pages")
    gguf_sub = gguf.add_subparsers(dest="gguf_cmd", required=True)

    gguf_inspect = gguf_sub.add_parser("inspect", help="parse header/metadata/tensor table without loading weights")
    gguf_inspect.add_argument("file")
    gguf_inspect.add_argument("--json", action="store_true")

    gguf_warm = gguf_sub.add_parser("warm", help="page in the weights of selected transformer layers")
    gguf_warm.add_argument("file")
    gguf_warm.add_argument("--layers", default="0,1,2", help="comma-separated layer indices")
    gguf_warm.add_argument("--lookahead", type=int, default=2)
    gguf_warm.add_argument("--json", action="store_true")

    gguf_prewarm = gguf_sub.add_parser(
        "prewarm",
        help="gradually page in every layer (background warmer for llama.cpp)",
    )
    gguf_prewarm.add_argument("file")
    gguf_prewarm.add_argument("--layers", default="all", help="`all` or 0,1,2 / 0-5")
    gguf_prewarm.add_argument("--lookahead", type=int, default=2)
    gguf_prewarm.add_argument(
        "--pace-ms", type=int, default=0, help="sleep between layers to avoid SSD bursts"
    )
    gguf_prewarm.add_argument("--json", action="store_true")

    return parser


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{value} B"


def _parse_layer_list(raw: str) -> list[int]:
    layers: list[int] = []
    for part in str(raw).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    if not layers:
        raise ValueError("no layers given (example: --layers 0,1,2)")
    return sorted(dict.fromkeys(layers))


def cmd_gguf_inspect(args: argparse.Namespace) -> int:
    from .gguf import GgufFile

    with GgufFile.open(args.file) as gguf:
        summary = gguf.summary()

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
        return 0

    print(f"GGUF {summary['path']}")
    print(f"  version        {summary['version']}")
    print(f"  tensors        {summary['tensor_count']}")
    print(f"  kv pairs       {summary['kv_count']}")
    print(f"  size           {_format_bytes(summary['total_bytes'])}")
    print(f"  header         {_format_bytes(summary['header_bytes'])} · padding {summary['padding']} B")
    print(f"  data offset    {summary['data_offset']} (alignment {summary['alignment']})")
    print(f"  layers         {summary['layer_count']}")
    if summary["expert_tensors"]:
        print(f"  expert tensors {len(summary['expert_tensors'])}")
    types = ", ".join(f"{name}×{count}" for name, count in sorted(summary["type_histogram"].items()))
    print(f"  types          {types}")
    meta = summary.get("metadata", {})
    for key in list(meta)[:8]:
        value = meta[key]
        text = str(value)
        if len(text) > 64:
            text = text[:61] + "…"
        print(f"  {key:<24} {text}")
    return 0


def cmd_gguf_warm(args: argparse.Namespace) -> int:
    from .gguf import GgufFile, LayerPageWarmer

    layers = _parse_layer_list(args.layers)
    with GgufFile.open(args.file) as gguf:
        warmer = LayerPageWarmer(gguf, lookahead=args.lookahead)
        result = warmer.warm_layers(layers)
        result["predict_next"] = warmer.predict_next()
        result["stats"] = warmer.stats()
        result["layers_total"] = len(gguf.layers)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0

    print(f"warmed layers {result['layers']} of {result['layers_total']}")
    print(f"  touched        {_format_bytes(result['bytes'])} in {result['chunks']} chunks")
    print(f"  time           {result['ms']:.2f} ms · {result['MBps']:.1f} MB/s")
    print(f"  predict next   {result['predict_next']}")
    stats = result["stats"]
    print(
        f"  session        {stats.get('layers_warmed', 0)} layers · "
        f"{_format_bytes(stats.get('total_bytes', 0))} · "
        f"{stats.get('MBps', 0.0):.1f} MB/s"
    )
    return 0


def cmd_gguf_prewarm(args: argparse.Namespace) -> int:
    from .gguf import GgufFile, LayerPageWarmer

    with GgufFile.open(args.file) as gguf:
        if str(args.layers).strip().lower() in ("all", "*", ""):
            ids = sorted(gguf.layers)
        else:
            ids = _parse_layer_list(args.layers)
        warmer = LayerPageWarmer(gguf, lookahead=args.lookahead)
        started = time.perf_counter()
        warmed: list[int] = []
        touched = 0
        for layer_id in ids:
            result = warmer.warm_layers([layer_id])
            if result["layers"]:
                warmed.append(layer_id)
                touched += result["bytes"]
                if not args.json:
                    print(
                        f"  layer {layer_id:>3} · {result['bytes'] / 1e6:.2f} MB · "
                        f"{result['MBps']:.0f} MB/s",
                        file=sys.stderr,
                    )
            if args.pace_ms > 0:
                time.sleep(args.pace_ms / 1000.0)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        payload = {
            "path": str(gguf.path),
            "layers": warmed,
            "layers_total": len(gguf.layers),
            "bytes": touched,
            "ms": round(elapsed_ms, 2),
            "MBps": round((touched / 1e6) / (elapsed_ms / 1000.0), 2) if elapsed_ms > 0 else 0.0,
            "pace_ms": args.pace_ms,
            "stats": warmer.stats(),
        }

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0

    print(
        f"prewarmed {len(payload['layers'])}/{payload['layers_total']} layers · "
        f"{_format_bytes(payload['bytes'])} in {payload['ms']:.1f} ms "
        f"({payload['MBps']:.1f} MB/s, pace {payload['pace_ms']} ms)"
    )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .engine import AiEngine

    engine = AiEngine(
        workspace=args.workspace,
        socket=args.socket,
        state_dir=args.state_dir,
        ram_budget=args.ram_budget,
        model=args.model,
        lookahead=args.lookahead,
        nprobe=args.nprobe,
        demo=args.demo,
        neural=not args.no_neural,
        rag=not args.no_rag,
        synthetic_moe=args.synthetic_moe,
    )
    engine.start()
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    from .rag.index import RagIndex

    ridx = RagIndex(Path(args.state_dir) / "rag", Path(args.workspace))
    ridx.connect()
    started = time.perf_counter()
    result = ridx.build(force=args.force, max_files=args.max_files)
    result["wall_ms"] = round((time.perf_counter() - started) * 1000, 2)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"indexed {result.get('docs', 0)} docs · {result.get('chunks', 0)} chunks · "
            f"{result.get('clusters', 0)} clusters · {result.get('index_type')} · "
            f"{result.get('index_bytes', 0) / MB:.1f} MiB on disk · {result['wall_ms']} ms"
        )
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    from .rag.index import RagIndex

    ridx = RagIndex(Path(args.state_dir) / "rag", Path(args.workspace))
    ridx.connect()
    if not ridx.indexed():
        ridx.build()
    vector = ridx.embedder.embed(args.text)
    results = ridx.search(vector, k=args.k)
    if args.json:
        print(json.dumps({"query": args.text, "results": results}, indent=2))
    else:
        for i, row in enumerate(results, 1):
            print(f"{i:>2}. {row['score']:.3f}  {row['path']}  #{row['chunk']}")
            print("    " + row["text"].replace("\n", " ")[:160])
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    report: dict = {"checks": []}

    def check(name: str, fn) -> None:
        started = time.perf_counter()
        try:
            value = fn()
            report["checks"].append(
                {"check": name, "ok": True, "detail": value, "ms": round((time.perf_counter() - started) * 1000, 1)}
            )
        except Exception as exc:  # noqa: BLE001
            report["checks"].append(
                {"check": name, "ok": False, "detail": f"{type(exc).__name__}: {exc}"}
            )

    check("python", lambda: sys.version.split()[0])
    check("mlx import", lambda: __import__("mlx.core", fromlist=["core"]).__version__ if hasattr(__import__("mlx.core", fromlist=["core"]), "__version__") else "ok")
    check("lancedb import", lambda: __import__("lancedb").__version__)

    workspace = Path(tempfile.mkdtemp(prefix="mext-selftest-"))
    (workspace / "README.md").write_text(
        "# Alpha\n\nThe cache evicts compressed pages with madvise DONTNEED.\n"
    )
    (workspace / "cache.rs").write_text(
        "/// ARC replacement for the compressed pool\npub struct ArcCache { hot: usize }\n"
    )
    (workspace / "rag.py").write_text(
        "def prefetch(prefix):\n    '''warm IVF clusters while typing'''\n"
    )

    state = workspace / ".mext"

    def build_index():
        from .rag.index import RagIndex

        ridx = RagIndex(state / "rag", workspace)
        ridx.connect()
        result = ridx.build(force=True)
        return f"{result['chunks']} chunks, {result['clusters']} clusters, {result['index_type']}"

    check("lancedb build", build_index)

    def query_and_prefetch():
        from .rag.index import RagIndex
        from .rag.prefetcher import StreamingQueryPrefetcher

        ridx = RagIndex(state / "rag", workspace)
        ridx.connect()
        prefetcher = StreamingQueryPrefetcher(ridx, nprobe=2)
        hit = prefetcher.feed("compressed pages evict")
        result = prefetcher.commit("compressed pages evict", k=3)
        prefetcher.close()
        assert result["results"], "no search results"
        return f"{len(result['results'])} results, prefetch clusters {hit['clusters']}"

    check("streaming prefetch", query_and_prefetch)

    def neural_stream():
        from .neural.runner import MextModelRunner

        runner = MextModelRunner("synthetic", state / "neural", 512 * MB, lookahead=2)
        runner.load()
        assert runner.metrics()["available"], runner.load_error
        out = runner.generate("def fib(n):", max_tokens=6)
        metrics = runner.metrics()
        runner.stop()
        assert out["generated"] == 6
        return (
            f"{metrics['mode']} · hot {metrics['hot_bytes'] / MB:.1f} MiB · "
            f"cold {metrics['cold_bytes'] / MB:.1f} MiB · {out['tokens_per_s']} tok/s"
        )

    def moe_paging():
        import mlx.core as mx

        from .neural.moe import ExpertPager
        from .neural.store import ColdStore

        store = ColdStore(state / "moe", 1 * MB)
        pager = ExpertPager(store, "selftest.mlp", 64, 96, 8, 2)
        x = mx.random.normal((1, 6, 64)).astype(mx.float32)
        pager(x)
        pager.prefetch(pager.predicted_experts())
        time.sleep(0.3)
        pager(x)
        pager(x)
        metrics = pager.metrics()
        store.stop()
        return (
            f"{metrics['experts_total']} experts · {metrics['experts_resident']} resident · "
            f"{metrics['expert_hit_rate'] * 100:.0f}% router hits"
        )

    def switch_paging():
        import mlx.core as mx
        import numpy as np
        from mlx_lm.models import switch_layers

        from .neural.store import ColdStore
        from .neural.switch import PagedSwitchGLU

        glu = switch_layers.SwitchGLU(64, 128, 4)
        rng = np.random.default_rng(1)
        x = mx.array(rng.normal(size=(1, 4, 64)).astype(np.float32))
        indices = mx.array(rng.integers(0, 4, size=(1, 4, 2)).astype(np.uint32))
        expected = glu(x, indices)
        store = ColdStore(state / "switch", 16 * MB)
        paged = PagedSwitchGLU.from_module(store, "selftest.switch_mlp", glu)
        got = paged(x, indices)
        mx.eval(expected, got)
        diff = float(mx.max(mx.abs(got - expected)))
        store.stop()
        assert diff < 1e-5, diff
        return f"SwitchGLU paged · max|Δ| = {diff:.2e}"

    if HAS_MLX:
        check("mlx layer streaming", neural_stream)
        check("moe expert paging", moe_paging)
        if __import__("importlib").util.find_spec("mlx_lm"):
            check("paged SwitchGLU", switch_paging)
    else:
        report["checks"].append({"check": "mlx layer streaming", "ok": False, "detail": "mlx not installed"})

    ok = all(c["ok"] for c in report["checks"])
    report["ok"] = ok
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("\n  MEXT-M1 AI selftest")
        print("  " + "─" * 62)
        for item in report["checks"]:
            print(f"  {'✓' if item['ok'] else '✗'} {item['check']:<24} {item['detail']}")
        print("  " + "─" * 62)
        print(f"  {'PASS' if ok else 'FAIL'}\n")
    return 0 if ok else 1


def cmd_prepare(args: argparse.Namespace) -> int:
    print(
        "Real-model streaming\n"
        "────────────────────\n"
        "MEXT streams both fp16 and 4/8-bit MLX checkpoints. Only *local* directories\n"
        "are wrapped (nothing is downloaded implicitly).\n\n"
        "  .venv/bin/pip install -e 'ai-engine[real-model]'\n"
        "  huggingface-cli download mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \\\n"
        "      --local-dir ~/models/qwen25-coder-7b-4bit\n"
        "  mext start --workspace ~/code/my-project --ram-limit 3G \\\n"
        "      --model ~/models/qwen25-coder-7b-4bit\n\n"
        "On load, every cold nn.Linear / nn.QuantizedLinear becomes a StreamedLinear /\n"
        "StreamedQuantizedLinear: weight (and scales/biases for 4-bit) live as .npy on\n"
        "SSD, the head is pinned, and layers are prefetched ahead of the forward pass.\n"
        "Numerics are identical to a resident model (same MLX kernels) — covered by\n"
        "ai-engine/tests/test_quantized.py.\n\n"
        "GGUF / llama.cpp checkpoints are not supported (different runtime).\n"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if args.cmd == "serve":
        return cmd_serve(args)
    if args.cmd == "index":
        return cmd_index(args)
    if args.cmd == "query":
        return cmd_query(args)
    if args.cmd == "gguf":
        if args.gguf_cmd == "inspect":
            return cmd_gguf_inspect(args)
        if args.gguf_cmd == "warm":
            return cmd_gguf_warm(args)
        return cmd_gguf_prewarm(args)
    if args.cmd == "selftest":
        return cmd_selftest(args)
    if args.cmd == "prepare":
        return cmd_prepare(args)
    parser.error(f"unknown command {args.cmd}")
    return 2
