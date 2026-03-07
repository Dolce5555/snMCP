#!/bin/bash

# vllm-gpuコンテナ内で実行する
nohup vllm serve --config /app/vllm/configs/gemma-3-1b.yaml > /app/vllm/vllm.log 2>&1 &
