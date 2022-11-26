#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source /project/g110/spack/user/tsa/spack/share/spack/setup-env.sh

cosmo_eccodes_dir=$(spack find --format "{prefix}" cosmo-eccodes-definitions@2.19.0.7%gcc | head -n1)
eccodes_dir=$(spack find --format "{prefix}" eccodes@2.19.0%gcc | head -n1)
export GRIB_DEFINITION_PATH_ECCODES=${eccodes_dir}/share/eccodes/definitions/
export GRIB_DEFINITION_PATH_COSMO=${SCRIPT_DIR}/../cosmoDefinitions/definitions
export GRIB_DEFINITION_PATH=${GRIB_DEFINITION_PATH_COSMO}:${GRIB_DEFINITION_PATH_ECCODES}
SCRIPTPATH="$( cd -- "$(dirname "${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]} ")" >/dev/null 2>&1 ; pwd -P )"
export PYTHONPATH=${SCRIPTPATH}/../src


