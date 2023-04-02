#!/bin/bash

set -e -o pipefail

build_wheel()
{
  echo "Build package wheel file..."
  python3 setup.py bdist_wheel

  latest_built_file=$(ls -t ./dist/*.whl | head -n1)
  arr_file_token=(${latest_built_file//// })
  latest_file_name=${arr_file_token[2]}
  arr_file_token=(${latest_file_name//-/ })
  latest_file_name="${arr_file_token[0]}-latest-${arr_file_token[2]}-${arr_file_token[3]}-${arr_file_token[4]}"

  echo "Copy ${latest_built_file} to .dist/${latest_file_name}"
  cp "${latest_built_file}" "./dist/${latest_file_name}"
}

build_egg()
{
  echo "Build package egg file..."
  python3 setup.py bdist_egg

  latest_built_file=$(ls -t ./dist/*.egg | head -n1)
  arr_file_token=(${latest_built_file//// })
  latest_file_name=${arr_file_token[2]}
  arr_file_token=(${latest_file_name//-/ })
  latest_file_name="${arr_file_token[0]}-latest.egg"

  echo "Copy ${latest_built_file} to .dist/${latest_file_name}"
  cp "${latest_built_file}" "./dist/${latest_file_name}"
}

CMD=$1
shift
$CMD $*