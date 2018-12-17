#!/usr/bin/env bash

TEMP_DIR="/tmp/wmt16_mmt"
mkdir -p ${TEMP_DIR}
pushd ${TEMP_DIR}
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/mmt16_task1_test.tar.gz
DATA_DIR="wmt16/mmt"
mkdir -p ${DATA_DIR}
tar -xf training.tar.gz -C ${DATA_DIR} && rm training.tar.gz
tar -xf validation.tar.gz -C ${DATA_DIR} && rm validation.tar.gz
tar -xf mmt16_task1_test.tar.gz -C ${DATA_DIR} && rm mmt16_task1_test.tar.gz
gsutil cp -r ${DATA_DIR} gs://huss-ml-dev/datasets/wmt16/
rm -r ${DATA_DIR}
popd
rm -r ${TEMP_DIR}
