"""Utility function and classes for tests."""

from contextlib import AbstractContextManager
import os
import shutil
import jinja2
import xarray as xr
import subprocess

data_dir = "/project/s83c/rz+/icon_data_processing_incubator/data/SWISS"


class fx_context(AbstractContextManager):
    """
    A context manager for tests which compare with an ouput from fieldextra.

    This context manager ensures that all files produced are cleaned up.
    It also provides an implementation for a typical test workflow to reduce redundant code.
    """

    test_dir = os.path.dirname(os.path.realpath(__file__))
    """The directory of this source file"""
    tmp_dir = os.path.join(test_dir, "fx_tmp")
    """A directory for temporarily storing files. Will be created on enter and deleted on exit."""
    output: xr.Dataset | None = None
    """The output of fieldextra, or None if fieldextra had no successful run yet."""
    cwd: str | None = None
    """The current working directory"""

    def __init__(
        self,
        exp_name: str,
        template_dir: str = "fe_templates",
        fx_binary: str = "/project/s83c/fieldextra/tsa/bin/fieldextra_gnu_opt_omp",
        nl_input: str = os.path.join(data_dir, "lfff<DDHH>0000.ch"),
        nl_const_input: str = os.path.join(data_dir, "lfff00000000c.ch"),
        nl_output_prefix: str = "<HH>",
        output_prefix: str = "00",
    ) -> None:
        """
        Create a new fieldextra context manager.

        Args:
            exp_name (str): The name of the experiment.
                Used for retrieving the namelist template and for naming temporary files.
            template_dir (str, optional): The directory (relative to this source file), where the namelist templates can be found.
                The namelist templates must be named f'test_{exp_name}.nl' to be retrieved.
                Use ``in_file="{{ file.inputi }}"`` inside the template for &Process blocks with standard input fields.
                Use ``in_file="{{ file.inputc }}"`` inside the template for &Process blocks with constant input fields.
                Use ``out_file="{{ file.output }}"`` inside the template for specifying the output file.
                The output must be a single netcdf file.
                Defaults to "fe_templates".
            fx_binary (str, optional): The location of the fieldextra binary.
                Defaults to "/project/s83c/fieldextra/tsa/bin/fieldextra_gnu_opt_omp".
            nl_input (str, optional): The file pattern to be inserted for ``{{ file.inputi }}`` in the template.
                Defaults to os.path.join(data_dir, "lfff<DDHH>0000.ch").
            nl_const_input (str, optional): The file pattern to be inserted for ``{{ file.inputc }}`` in the template.
                Defaults to os.path.join(data_dir, "lfff00000000c.ch").
            nl_output_prefix (str, optional): The prefix to be used for the output file pattern.
                The full output file pattern wil be f'{nl_output_prefix}_{exp_name}.nc'.
                Defaults to "<HH>".
            output_prefix (str, optional): The prefix for the output file produced by fieldextra.
                The full output file will be f'{output_prefix}_{exp_name}.nc'.
                Defaults to "00".
        """
        self.exp_name = exp_name
        self.template_dir = os.path.join(self.test_dir, template_dir)
        self.fx_binary = fx_binary
        self.nl_input = nl_input
        self.nl_const_input = nl_const_input
        self.nl_output = f"{nl_output_prefix}_{exp_name}.nc"
        self.output_path = f"{output_prefix}_{exp_name}.nc"

    def __enter__(self):
        self.cwd = os.getcwd()
        # create tmp_dir
        os.mkdir(self.tmp_dir)
        # prepare control file
        conf_files = {
            "inputi": self.nl_input,
            "inputc": self.nl_const_input,
            "output": self.nl_output,
        }
        templateLoader = jinja2.FileSystemLoader(searchpath=self.template_dir)
        templateEnv = jinja2.Environment(loader=templateLoader)
        template = templateEnv.get_template(f"./test_{self.exp_name}.nl")
        outputText = template.render(file=conf_files, ready_flags=self.tmp_dir)
        nl_file = os.path.join(self.tmp_dir, f"{self.exp_name}.nc")
        with open(nl_file, "w") as f:
            f.write(outputText)
        # run fieldextra
        os.chdir(self.tmp_dir)
        subprocess.run([self.fx_binary, nl_file], check=True)
        # export data
        self.output = xr.open_dataset(self.output_path)
        return self

    def __exit__(
        self,
        __exc_type: type[BaseException] | None,
        __exc_value: BaseException | None,
        __traceback,
    ) -> bool | None:
        if self.cwd:
            # change back to original cwd
            os.chdir(self.cwd)
            self.cwd = None
        if os.path.exists(self.tmp_dir):
            # delete tmp_dir
            shutil.rmtree(self.tmp_dir)
        return None
