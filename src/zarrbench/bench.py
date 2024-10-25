import time
import asyncio
import aiohttp
import aiohttp.typedefs
import numpy as np
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Tuple, Optional, List, Dict, Any, Iterator

get_perf_time = time.monotonic

# def get_perf_time():
#    asyncio.get_running_loop().time()


@dataclass
class VarItem:
    name: str
    shape: Tuple[int]
    chunks: Tuple[int]
    dimension_separator: str

    @property
    def n_chunks(self) -> List[int]:
        return [int(np.ceil(s / c)) for s, c in zip(self.shape, self.chunks)]

    @property
    def n(self) -> int:
        return int(np.prod(self.n_chunks))


@dataclass
class InternalTraceResult:
    name: str
    tstart: float
    tcon_start: float
    tcon_end: float
    tend: float
    reuseconn: bool


@dataclass
class ExternalTraceResult:
    tinit: float
    tdata: float
    tfinish: float
    size: int


@dataclass
class TraceResult:
    internal: InternalTraceResult
    external: ExternalTraceResult


def show_ls(variables: List[VarItem]) -> None:
    from rich.console import Console
    from rich.table import Table

    table = Table(title="Variables")

    table.add_column("name", justify="left", style="cyan", no_wrap=True)
    table.add_column("shape", style="magenta")
    table.add_column("n_chunks", justify="right", style="green")

    for var in variables:
        table.add_row(var.name, str(var.shape), str(var.n_chunks))

    console = Console()
    console.print(table)


def show_traces(traces: List[TraceResult]) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    import datetime as dt
    import humanize

    table = Table(title="Traces")

    table.add_column("name", justify="left", style="cyan", no_wrap=True)
    table.add_column("duration")
    table.add_column("size")

    for trace in sorted(traces, key=lambda t: t.internal.name):
        table.add_row(
            trace.internal.name,
            str(trace.external.tfinish - trace.external.tinit),
            humanize.naturalsize(trace.external.size),
        )

    all_start = min(t.external.tinit for t in traces)
    all_end = max(t.external.tfinish for t in traces)
    total_size = sum(t.external.size for t in traces)
    total_time = all_end - all_start

    console = Console()
    console.print(table)
    console.print(
        Text.assemble(
            "duration: ",
            (humanize.precisedelta(dt.timedelta(seconds=total_time)), "bold"),
        )
    )
    console.print(
        Text.assemble(
            "average speed: ",
            (humanize.naturalsize(total_size / total_time) + "/s", "bold"),
        )
    )
    console.print(
        Text.assemble("average iops: ", (f"{len(traces) / total_time:.2f}/s", "bold"))
    )


def ls(metadata: Dict[str, Any]) -> List[VarItem]:
    variables = []

    for k, array_meta in metadata["metadata"].items():
        if "/" not in k:
            continue
        varname, partkey = k.rsplit("/", 1)
        if partkey != ".zarray":
            continue

        variables.append(
            VarItem(
                varname,
                tuple(array_meta["shape"]),
                tuple(array_meta["chunks"]),
                array_meta.get("dimension_separator", "."),
            )
        )

    return variables


def random_chunk_urls(
    variables: List[VarItem], n: int, rng: np.random.Generator
) -> Iterator[str]:
    weights = np.array([v.n for v in variables])
    cumweights = np.cumsum(weights)
    samples = rng.integers(0, cumweights[-1], n)
    i_var = np.searchsorted(cumweights, samples, "right")
    i_local = samples - np.concatenate([[0], cumweights])[i_var]
    for iv, il in zip(i_var, i_local):
        var = variables[iv]
        idx = np.unravel_index(il, var.n_chunks)
        yield f"{var.name}/{var.dimension_separator.join(map(str, idx))}"


async def load_and_trace(
    url: str,
    session: aiohttp.ClientSession,
    trace_id: Any,
    traces: Dict[Any, InternalTraceResult],
    ctx: Optional[Any] = None,
) -> TraceResult:
    ctx = {**(ctx or {}), "trace_id": trace_id}

    tinit = get_perf_time()
    async with session.get(url, trace_request_ctx=ctx) as r:
        tdata = get_perf_time()
        size = len(await r.read())
        tfinish = get_perf_time()

    return TraceResult(
        traces[trace_id],
        ExternalTraceResult(tinit, tdata, tfinish, size),
    )


# trace_config = aiohttp.TraceConfig()
# trace_config.on_request_start.append(on_request_start)
# trace_config.on_request_end.append(on_request_end)
# async with aiohttp.ClientSession(
#        trace_configs=[trace_config]) as client:
#    client.get('http://example.com/some/redirect/')


async def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument(
        "-n", "--requests", default=1, type=int, help="number of requests"
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="seed for random number generation, must be != 0, no seed meens randomly chosen seed",
    )
    parser.add_argument(
        "--decompress",
        default=False,
        action="store_true",
        help="decompress HTTP compression",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    traces = {}

    async def on_request_start(
        session: aiohttp.ClientSession,
        trace_config_ctx: SimpleNamespace,
        params: aiohttp.TraceRequestStartParams,
    ) -> None:
        trace_config_ctx.start = get_perf_time()

    async def on_connection_reuseconn(
        session: aiohttp.ClientSession,
        trace_config_ctx: SimpleNamespace,
        params: aiohttp.TraceConnectionReuseconnParams,
    ) -> None:
        trace_config_ctx.con_start = trace_config_ctx.con_end = get_perf_time()
        trace_config_ctx.reuse = True

    async def on_connection_create_start(
        session: aiohttp.ClientSession,
        trace_config_ctx: SimpleNamespace,
        params: aiohttp.TraceConnectionCreateStartParams,
    ) -> None:
        trace_config_ctx.con_start = get_perf_time()
        trace_config_ctx.reuse = False

    async def on_connection_create_end(
        session: aiohttp.ClientSession,
        trace_config_ctx: SimpleNamespace,
        params: aiohttp.TraceConnectionCreateEndParams,
    ) -> None:
        trace_config_ctx.con_end = get_perf_time()

    async def on_request_end(
        session: aiohttp.ClientSession,
        trace_config_ctx: SimpleNamespace,
        params: aiohttp.TraceRequestEndParams,
    ) -> None:
        trace_config_ctx.end = get_perf_time()
        if (
            trace_id := (trace_config_ctx.trace_request_ctx or {}).get("trace_id", None)
        ) is not None:
            traces[trace_id] = InternalTraceResult(
                name=trace_config_ctx.trace_request_ctx["u"],
                tstart=trace_config_ctx.start,
                tcon_start=trace_config_ctx.con_start,
                tcon_end=trace_config_ctx.con_end,
                tend=trace_config_ctx.end,
                reuseconn=trace_config_ctx.reuse,
            )

    trace_config = aiohttp.TraceConfig()
    trace_config.on_request_start.append(on_request_start)
    trace_config.on_connection_reuseconn.append(on_connection_reuseconn)
    trace_config.on_connection_create_start.append(on_connection_create_start)
    trace_config.on_connection_create_end.append(on_connection_create_end)
    trace_config.on_request_end.append(on_request_end)

    async with aiohttp.ClientSession(
        trace_configs=[trace_config], auto_decompress=args.decompress
    ) as session:
        async with session.get(args.url + "/.zmetadata") as r:
            metadata = await r.json(content_type=None)
        variables = ls(metadata)
        show_ls(variables)

        trace_results = await asyncio.gather(
            *[
                load_and_trace(args.url + "/" + u, session, i, traces, {"u": u})
                for i, u in enumerate(random_chunk_urls(variables, args.requests, rng))
            ]
        )

    show_traces(trace_results)
    return 0
