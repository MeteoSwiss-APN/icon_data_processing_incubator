#!/bin/bash

set -ex

spack_c2sm_url=https://github.com/C2SM/spack-c2sm.git
spack_c2sm_tag=v0.20.1.0
spack_c2sm_dir=$(pwd)/spack-c2sm

while getopts "w:s:" flag; do
    case ${flag} in
        s) spack_c2sm_dir=${OPTARG};;
    esac
done

git clone --depth 1 --recurse-submodules -b ${spack_c2sm_tag} ${spack_c2sm_url} ${spack_c2sm_dir}

. ${spack_c2sm_dir}/setup-env.sh
