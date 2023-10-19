#!/bin/bash

set -ex

script_dir=$(dirname -- "$(readlink -f -- "$BASH_SOURCE")")

workspace=$(pwd)
spack_c2sm_url=https://github.com/C2SM/spack-c2sm.git
spack_c2sm_tag=v0.20.1.0
spack_c2sm_dir=${workspace}/spack-c2sm

while getopts "w:s:" flag; do
    case ${flag} in
        w) workspace=${OPTARG};;
        s) spack_c2sm_dir=${OPTARG};;
    esac
done

mkdir -p ${workspace}
pushd ${workspace}

git clone --depth 1 --recurse-submodules -b ${spack_c2sm_tag} ${spack_c2sm_url} ${spack_c2sm_dir}

. ${spack_c2sm_dir}/setup-env.sh

mkdir spack-env
cp ${script_dir}/spack.yaml spack-env/
spack env activate -p spack-env

spack install

popd
