#!/bin/bash
set -e -x


# Compile wheels
for PYBIN in /opt/python/*/bin; do
    "${PYBIN}/pip" install numpy
    CPATH=$CPATH:/ising/thrust "${PYBIN}/pip" wheel /ising/ -w /ising/wheelhouse/
    "${PYBIN}/python" /ising/setup.py sdist -d /ising/wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in /ising/wheelhouse/ising*.whl; do
    auditwheel repair "$whl" -w /ising/wheelhouse/
done

# Install packages and test
for PYBIN in /opt/python/*/bin/; do
    "${PYBIN}/pip" install ising --no-index -f /ising/wheelhouse
done
