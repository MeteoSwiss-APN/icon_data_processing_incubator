#!/bin/bash

script_dir=$(dirname -- "$(readlink -f -- "$BASH_SOURCE")")

spack_c2sm_url=https://github.com/C2SM/spack-c2sm.git
spack_c2sm_tag=v0.20.1.0
workspace=$(pwd)

while getopts "w:" flag; do
    case ${flag} in
        w) workspace=${OPTARG};;
    esac
done

pushd $workspace

git clone --depth 1 --recurse-submodules --shallow-submodules -b ${spack_c2sm_tag} ${spack_c2sm_url}

. spack-c2sm/setup-env.sh

mkdir spack-env
cp ${script_dir}/spack.yaml spack-env/
spack env activate -p spack-env
spack install

popd