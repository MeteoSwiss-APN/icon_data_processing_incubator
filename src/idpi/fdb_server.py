"""FDB server module.

Usage:
$ uvicorn idpi.fdb_server:app [--reload] --port 8989

The reload option enables reloading the server
when changes to the module are observed.
"""


# Standard library
import anyio
import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path

logger = logging.getLogger(__name__)
root = Path(__file__).parents[2]
config_path = root / "src/idpi/data/fdb_config_balfrin.yaml"

os.environ["FDB5_DIR"] = str(root / "spack-env/.spack-env/view")
os.environ["FDB5_CONFIG"] = config_path.read_text()

# Third-party
import pyfdb
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

# Local
from . import mars

app = FastAPI()
fdb = pyfdb.FDB()
lck = anyio.Lock()


@app.get("/")
def info() -> dict:
    return {
        "FDB5_CONFIG": os.environ["FDB5_CONFIG"],
        "FDB5_DIR": os.environ["FDB5_DIR"],
    }


async def fdb_retrieve(req: dict) -> AsyncIterator[bytes]:
    datareader = fdb.retrieve(req)
    while chunk := datareader.read(16 * 1024**2):
        yield bytes(chunk)


@app.post("/retrieve")
async def retrieve(request: mars.Request):
    logger.info("Retrieving %s", str(request))
    async with lck:  # datareader is not thread safe
        return StreamingResponse(fdb_retrieve(request.to_fdb()))


async def split_messages(stream: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    buf: list[bytes] = []
    async for chunk in stream:
        agg = b"".join(buf + [chunk])
        msg, sep, remainder = agg.partition(b"7777")
        buf = [msg, sep]
        while remainder:
            yield b"".join(buf)
            msg, sep, remainder = remainder.partition(b"7777")
            buf = [msg, sep]

    yield b"".join(buf)


@app.post("/archive")
async def archive(request: Request):
    async for msg in split_messages(request.stream()):
        fdb.archive(msg)  # is it okay to archive one by one?
    fdb.flush()
