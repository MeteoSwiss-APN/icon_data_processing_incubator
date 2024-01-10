"""FDB server module.

Usage:
$ uvicorn idpi.fdb_server:app [--reload] --port 8989

The reload option enables reloading the server
when changes to the module are observed.
"""


# Standard library
import anyio
import contextlib
import logging
import os
import tempfile
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
from . import mars, data_source

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


@app.post("/archive")
async def archive(request: Request):
    with contextlib.ExitStack() as stack:
        stack.enter_context(data_source.cosmo_grib_defs())
        tmp = stack.enter_context(tempfile.SpooledTemporaryFile())
        aio_tmp = anyio.wrap_file(tmp)
        async for chunk in request.stream():
            await aio_tmp.write(chunk)
        await aio_tmp.seek(0)
        data = await aio_tmp.read()
        fdb.archive(data)
