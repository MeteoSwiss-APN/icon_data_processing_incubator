# Standard library
import dataclasses as dc
import io
from pathlib import Path

# Third-party
import requests

# First-party
from idpi import mars


@dc.dataclass
class FDBClient:
    host: str = "http://127.0.0.1:8989"
    chunk_size: int = 16 * 1024**2  # bytes

    def retrieve(self, req: mars.Request, handle: io.IOBase):
        url = f"{self.host}/retrieve"
        json = req.to_fdb() | {"param": req.param}

        with requests.post(url, json=json, stream=True) as resp:
            for chunk in resp.iter_content(self.chunk_size):
                handle.write(chunk)

    def archive(self, handle: io.IOBase):
        resp = requests.post(f"{self.host}/archive", data=handle)
        if not resp.ok:
            print("Archive failed.")


if __name__ == "__main__":
    req = mars.Request("HHL", "20230201", "0300")
    path = Path("/scratch/mch/ckanesan/output.grib")

    client = FDBClient()
    with path.open("wb") as f:
        client.retrieve(req, f)
