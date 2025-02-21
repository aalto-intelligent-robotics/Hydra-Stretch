#!/usr/bin/bash

set -e

usage="$(basename "$0") [-h] [-n OUTPUT_NAME]
    -h  show this help text
    -n  provide the prefix of the output directory"

options=':hn:d:'
while getopts $options option; do
  case "$option" in
    h) echo "$usage"; exit;;
    n) OUTPUT_PREFIX=$OPTARG;;
    d) OUTPUT_DIR=$OPTARG;;
    :) printf "missing argument for -%s\n" "$OPTARG" >&2; echo "$usage" >&2; exit 1;;
   \?) printf "illegal option: -%s\n" "$OPTARG" >&2; echo "$usage" >&2; exit 1;;
  esac
done

# mandatory arguments
if [ ! "$OUTPUT_PREFIX" ] || [ ! "$OUTPUT_DIR" ]; then
  echo "arguments -n and -d must be provided"
  echo "$usage" >&2; exit 1
fi

source $HOME/hydra_ws/devel/setup.bash

if [ ! -d "$OUTPUT_DIR/$OUTPUT_PREFIX" ]; then
  mkdir -p $OUTPUT_DIR/$OUTPUT_PREFIX
  cd $OUTPUT_DIR/$OUTPUT_PREFIX
  mkdir backend frontend topology lcd pgmo map2d
  echo "Created output directory $OUTPUT_DIR/$OUTPUT_PREFIX"
else
  echo "Ouptut directory $OUTPUT_DIR/$OUTPUT_PREFIX exists, skipping output directory creation"
fi
