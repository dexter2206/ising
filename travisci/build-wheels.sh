#!/bin/bash
set -e -x


# Compile wheels
for PYBIN in /opt/python/*/bin; do
    "${PYBIN}/pip" install numpy
    CPATH=$CPATH:/ising/thrust "${PYBIN}/pip" wheel /ising/ -w /ising/wheelhouse/
done

ls /ising/wheelhouse/ising*manylinux*.whl

# Bundle external shared libraries into the wheels
for whl in /ising/wheelhouse/ising*manylinux*.whl; do
    auditwheel repair "$whl" -w /ising/wheelhouse/
done

# Install packages and test
for PYBIN in /opt/python/*/bin/; do
    "${PYBIN}/pip" install ising --no-index -f /ising/wheelhouse
done