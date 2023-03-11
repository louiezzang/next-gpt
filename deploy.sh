#!/bin/bash

set -e -o pipefail

upload_wheel()
{
  local mode=$1

  if [ $# != 1 ]; then
    echo "Usage: $0 ${FUNCNAME[0]} <mode: dev, prod>"
    exit 1
  fi

  echo "$0 ${FUNCNAME[0]}"
  echo "mode = ${mode}"
  echo "Deploying package to Databricks File System..."

  echo "Build package wheel file..."
  python3 setup.py bdist_wheel

  latest_built_file=$(ls -t ./dist/*.whl | head -n1)
  arr_file_token=(${latest_built_file//// })
  latest_file_name=${arr_file_token[2]}
  arr_file_token=(${latest_file_name//-/ })
  latest_file_name="${arr_file_token[0]}-latest-${arr_file_token[2]}-${arr_file_token[3]}-${arr_file_token[4]}"

  echo "Copy ${latest_built_file} to .dist/${latest_file_name}"
  cp "${latest_built_file}" "./dist/${latest_file_name}"

  local dbfs_lib_home="dbfs:/FileStore/libs/${mode}"
  databricks fs mkdirs "${dbfs_lib_home}"

  dbfs cp -r --overwrite dist "${dbfs_lib_home}"

  echo "Deployed package to Databricks File System: ${dbfs_lib_home}"
}

upload_egg()
{
  local mode=$1

  if [ $# != 1 ]; then
    echo "Usage: $0 ${FUNCNAME[0]} <mode: dev, prod>"
    exit 1
  fi

  echo "$0 ${FUNCNAME[0]}"
  echo "mode = ${mode}"
  echo "Deploying package to Databricks File System..."

  echo "Build package egg file..."
  python3 setup.py bdist_egg

  latest_built_file=$(ls -t ./dist/*.egg | head -n1)
  arr_file_token=(${latest_built_file//// })
  latest_file_name=${arr_file_token[2]}
  arr_file_token=(${latest_file_name//-/ })
  latest_file_name="${arr_file_token[0]}-latest.egg"

  echo "Copy ${latest_built_file} to .dist/${latest_file_name}"
  cp "${latest_built_file}" "./dist/${latest_file_name}"

  local dbfs_lib_home="dbfs:/FileStore/libs/${mode}"
  databricks fs mkdirs "${dbfs_lib_home}"

  dbfs cp -r --overwrite dist "${dbfs_lib_home}"

  echo "Deployed package to Databricks File System: ${dbfs_lib_home}"
}

CMD=$1
shift
$CMD $*