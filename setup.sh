#!/usr/bin/env bash
# setup.sh -- environment bootstrapper for python virtualenv

set -euo pipefail

SUDO=sudo
if ! command -v $SUDO; then
    echo no sudo on this system, proceeding as current user
    SUDO=""
fi


if command -v apt-get; then
    $SUDO apt-get -qq update
    $SUDO apt-get -y install libjpeg-dev
    $SUDO apt-get -y install python3-venv
else
    echo Skipping tool installation because your platform is missing apt-get.
    echo If you see failures below, install the equivalent of python3-venv for your system.
fi

