#!/usr/bin/env bash

gcloud ml-engine local train \
    --module-name trainer.task \
    --package-path trainer/ \
    --job-dir ../output/temp_job \
    -- \
    --data-dir gs://huss-ml-dev/datasets/wmt16/mmt/ \
    --test-run True


gcloud ml-engine jobs submit training rec_att_mt_real_job_gpu_refactor \
    --job-dir gs://huss-ml-dev/ml-engine/recurrent_attention_machine_translation/real_job_gpu_refactor \
    --runtime-version 1.10 \
    --module-name trainer.task \
    --package-path trainer/ \
    --packages gs://huss-ml-dev/libs/keras/dist/Keras-2.2.4-py2-none-any.whl \
    --scale-tier basic_gpu \
    --region us-east1 \
    -- \
    --data-dir gs://huss-ml-dev/datasets/wmt16/mmt/