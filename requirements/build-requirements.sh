#!/bin/bash 
SCRIPT=$(readlink -f "$0")
ROOTPATH=$(dirname $SCRIPT)/..
cd $ROOTPATH
pip-compile --output-file=requirements/requirements.txt pyproject.toml
pip-compile --extra=test --output-file=requirements/requirements-test.txt pyproject.toml
pip-compile --extra=docs --output-file=requirements/requirements-docs.txt pyproject.toml
pip-compile --extra=quality --output-file=requirements/requirements-dev.txt pyproject.toml
