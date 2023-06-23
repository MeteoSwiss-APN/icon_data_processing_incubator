"""Definition of system constants used by IDPI."""
# Standard library
import os
import pathlib

try:
    input_data_directory = os.environ["INPUT_DATA_DIR"]
except KeyError:
    input_data_directory = str(
        pathlib.Path(__file__).parent.parent.parent.resolve() / "test_data"
    )
    print(f"The INPUT_DATA_DIR is not set, setting it to {input_data_directory}")


FX_BINARY = "fieldextra_executable"
INPUT_DATA_DIR = input_data_directory

root_dir = (pathlib.Path(os.path.dirname(os.path.abspath(__file__))) / "..").resolve()
