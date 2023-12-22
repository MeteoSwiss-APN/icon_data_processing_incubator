# Standard library
import io

# Third-party
import earthkit.data as ekd
from earthkit.data.writers import write


def override(message: bytes, **kwargs):
    stream = io.BytesIO(message)
    [grib_field] = ekd.from_source("stream", stream)

    out = io.BytesIO()
    md = grib_field.metadata().override(**kwargs)
    write(out, grib_field.values, md)

    return {
        "message": out.getvalue(),
        "geography": md.as_namespace("geography"),
        "parameter": md.as_namespace("parameter"),
    }
