import time
import asyncio
import aiohttp
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

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
    def n_chunks(self):
        return [int(np.ceil(s / c)) for s, c in zip(self.shape, self.chunks)]

    @property
    def n(self):
        return np.prod(self.n_chunks)


def show_ls(variables):
    from rich.console import Console
    from rich.table import Table

    table = Table(title="Variables")

    table.add_column("name", justify="chunks", style="cyan", no_wrap=True)
    table.add_column("shape", style="magenta")
    table.add_column("n_chunks", justify="right", style="green")

    for var in variables:
        table.add_row(var.name, str(var.shape), str(var.n_chunks))

    console = Console()
    console.print(table)

def show_traces(traces):
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    import datetime as dt
    import humanize

    table = Table(title="Traces")

    table.add_column("name", justify="chunks", style="cyan", no_wrap=True)
    table.add_column("duration")
    table.add_column("size")

    for trace in sorted(traces.values(), key=lambda t: t.name):
        table.add_row(trace.name, str(trace.tfinish- trace.tinit), humanize.naturalsize(trace.size))

    all_start = min(t.tinit for t in traces.values())
    all_end = max(t.tfinish for t in traces.values())
    total_size = sum(t.size for t in traces.values())
    total_time = all_end - all_start

    console = Console()
    console.print(table)
    console.print(Text.assemble("duration: ", (humanize.precisedelta(dt.timedelta(seconds=total_time)), "bold")))
    console.print(Text.assemble("average speed: ", (humanize.naturalsize(total_size / total_time) + "/s", "bold")))
    console.print(Text.assemble("average iops: ", (f"{len(traces) / total_time:.2f}/s", "bold")))


def ls(metadata):
    variables = []

    for k, array_meta in metadata["metadata"].items():
        if "/" not in k:
            continue
        varname, partkey = k.rsplit("/", 1)
        if partkey != ".zarray":
            continue

        variables.append(VarItem(varname, tuple(array_meta["shape"]), tuple(array_meta["chunks"]), array_meta.get("dimension_separator", ".")))
   
    return variables

def random_chunk_urls(variables, n, rng):
    weights = np.array([v.n for v in variables])
    cumweights = np.cumsum(weights)
    samples = rng.integers(0, cumweights[-1], n)
    i_var = np.searchsorted(cumweights, samples, "right")
    i_local = samples - np.concatenate([[0], cumweights])[i_var]
    for iv, il in zip(i_var, i_local):
        var = variables[iv]
        idx = np.unravel_index(il, var.n_chunks)
        yield f"{var.name}/{var.dimension_separator.join(map(str, idx))}"


async def load_and_trace(url, session, trace_id=None, traces=None, ctx=None):
    ctx = ctx or {}
    if trace_id is not None:
        ctx = {**ctx, "trace_id": trace_id}

    tinit = get_perf_time()
    async with session.get(url, trace_request_ctx=ctx) as r:
        before_data = get_perf_time()
        size = len(await r.read())
        tfinish = get_perf_time()
    if trace_id is not None:
        traces[trace_id].size = size
        traces[trace_id].tdata = before_data 
        traces[trace_id].tinit = tinit
        traces[trace_id].tfinish = tfinish
    return size
            

@dataclass
class TraceResult:
    name: str
    tstart: float
    tcon_start: float
    tcon_end: float
    tend: float
    reuseconn: bool
    tdata: Optional[float] = None
    size: Optional[int] = None
    tinit: Optional[float] = None
    tfinish: Optional[float] = None

#trace_config = aiohttp.TraceConfig()
#trace_config.on_request_start.append(on_request_start)
#trace_config.on_request_end.append(on_request_end)
#async with aiohttp.ClientSession(
#        trace_configs=[trace_config]) as client:
#    client.get('http://example.com/some/redirect/')


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("-n", "--requests", default=1, type=int, help="number of requests")
    parser.add_argument("--seed", default=None, type=int, help="seed for random number generation, must be != 0, no seed meens randomly chosen seed")
    parser.add_argument("--decompress", default=False, action="store_true", help="decompress HTTP compression")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    traces = {}

    async def on_request_start(
            session, trace_config_ctx, params):
        trace_config_ctx.start = get_perf_time()

    async def on_connection_reuseconn(session, trace_config_ctx, params):
        trace_config_ctx.con_start = trace_config_ctx.con_end = get_perf_time()
        trace_config_ctx.reuse = True

    async def on_connection_create_start(session, trace_config_ctx, params):
        trace_config_ctx.con_start = get_perf_time()
        trace_config_ctx.reuse = False 

    async def on_connection_create_end(session, trace_config_ctx, params):
        trace_config_ctx.con_end = get_perf_time()

    async def on_request_end(session, trace_config_ctx, params):
        trace_config_ctx.end = get_perf_time()
        if (trace_id := (trace_config_ctx.trace_request_ctx or {}).get("trace_id", None)) is not None:
            traces[trace_id] = TraceResult(
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

    async with aiohttp.ClientSession(trace_configs=[trace_config], auto_decompress=args.decompress) as session:
        async with session.get(args.url + "/.zmetadata") as r:
            metadata = await r.json(content_type=None)
        variables = ls(metadata)
        show_ls(variables)

        await asyncio.gather(*[load_and_trace(args.url + "/" + u, session, i, traces, {"u": u})
                               for i, u in enumerate(random_chunk_urls(variables, args.requests, rng))])

    show_traces(traces)