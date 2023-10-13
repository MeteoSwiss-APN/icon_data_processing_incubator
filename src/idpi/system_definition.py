"""Definition of system constants used by IDPI."""
# Standard library
import os
import pathlib

# The dict access was chosen here because it should fail if FIELDEXTRA_PATH is not set
try:
    fieldextra_executable = os.environ["FIELDEXTRA_PATH"]
except KeyError:
    print("The FIELDEXTRA_PATH is not set, exiting")
    raise

FX_BINARY = fieldextra_executable

root_dir = (pathlib.Path(os.path.dirname(os.path.abspath(__file__))) / "..").resolve()
