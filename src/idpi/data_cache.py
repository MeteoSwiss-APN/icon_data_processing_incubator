# Standard library
import dataclasses as dc
from itertools import product
from pathlib import Path

# Local
from . import data_source

DEFAULT_FILES = {
    "inputi": "<mmm>/lfff<ddhh>0000",
    "inputc": "<mmm>/lfff00000000c",
}


@dc.dataclass
class DataCache:
    cache_dir: Path
    fields: dict[str, list]
    files: dict[str, str] = dc.field(default_factory=lambda: DEFAULT_FILES)
    steps: list[int] = dc.field(default_factory=lambda: [0])
    numbers: list[int] = dc.field(default_factory=lambda: [0])
    _populated: list[Path] = dc.field(default_factory=list, init=False)

    def __post_init__(self):
        if not (self.fields.keys() <= self.files.keys()):
            raise ValueError("fields keys must be a subset of files keys")

    @property
    def conf_files(self) -> dict[str, Path]:
        return {
            label: self.cache_dir / pattern for label, pattern in self.files.items()
        }

    def _iter_files(self):
        # support more patterns ?
        # https://github.com/COSMO-ORG/fieldextra/blob/develop/documentation/README.user#L2797
        patterns = (
            ("<mmm>", "{mmm:03d}"),
            ("<ddhh>", "{dd:02d}{hh:02d}"),
        )
        for label, name in self.files.items():
            name = name.lower()
            for src, dst in patterns:
                name = name.replace(src, dst)
            for number, step in product(self.numbers, self.steps):
                dd = step // 24
                hh = step % 24
                yield label, name.format(mmm=number, dd=dd, hh=hh), number, step

    def _iter_requests(self, label: str, number: int, step: int):
        for param, levtype in self.fields[label]:
            # TODO: group params by levtype
            yield {"param": param, "levtype": levtype, "number": number, "step": step}

    def populate(self, source: data_source.DataSource):
        for label, rel_path, number, step in self._iter_files():
            path = self.cache_dir / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)

            with path.open("ba") as f:
                for req in self._iter_requests(label, number, step):
                    for field in source.retrieve(req):
                        f.write(field.message())

            self._populated.append(path)

    def clear(self):
        for path in self._populated:
            path.unlink()
