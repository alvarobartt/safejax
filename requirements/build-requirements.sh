#!/bin/bash 
SCRIPT=$(readlink -f "$0")
REQPATH=$(dirname $SCRIPT)
ROOTPATH=$REQPATH/..
pip-compile --output-file=$REQPATH/requirements.txt $ROOTPATH/pyproject.toml
pip-compile --extra=test --output-file=$REQPATH/requirements-test.txt $ROOTPATH/pyproject.toml
pip-compile --extra=docs --output-file=$REQPATH/requirements-docs.txt $ROOTPATH/pyproject.toml
pip-compile --extra=quality --output-file=$REQPATH/requirements-dev.txt $ROOTPATH/pyproject.toml
