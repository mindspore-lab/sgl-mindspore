# PD Disaggregation

## MemFabric Adaptor install
*Notice: Prebuilt wheel package is based on aarch64, please leave an issue [here](https://github.com/mindspore-lab/sgl-mindspore/issues) to let us know the requests for amd64 build.*

MemFabric Adaptor is a drop-in replacement of Mooncake Transfer Engine that enables KV cache transfer on Ascend NPU clusters.

```shell
MF_WHL_NAME="mf_adapter-1.0.0-cp311-cp311-linux_aarch64.whl"
MEMFABRIC_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/sglang/${MF_WHL_NAME}"
wget -O "${MF_WHL_NAME}" "${MEMFABRIC_URL}" && pip install "./${MF_WHL_NAME}"
```


## PD Disaggregation Examples
### Running Qwen3-32B
Running Qwen3-32B with PD disaggregation on 2 x Atlas 800I A2. Model weights could be found [here](https://modelers.cn/models/MindSpore-Lab/Qwen3-32B).
#### Prefill:
```shell
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_MF_STORE_URL="tcp://<PREFILL_HOST_IP>:6650"
export PYTHONPATH=/home/sglang-mindspore/python:$PYTHONPATH
export ASCEND_MF_TRANSFER_PROTOCOL=device_rdma

    python3 -m sglang.launch_server --model-path /home/ckpt/Qwen3-32B \
    --trust-remote-code \
    --attention-backend ascend \
    --model-impl mindspore \
    --mem-fraction-static 0.8 \
    --tp-size 4 \
    --dp-size 1 \
    --nnodes 1 \
    --node-rank 0 \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-port 6657 \
    --disaggregation-transfer-backend ascend \
    --dist-init-addr <PREFILL_HOST_IP>:6688 \
    --host <PREFILL_HOST_IP> \
    --port 8000
```

### Decode:
```shell
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/home/sglang-mindspore/python:$PYTHONPATH
export ASCEND_MF_STORE_URL="tcp://<PREFILL_HOST_IP>:6650"
export HCCL_BUFFSIZE=200
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=24
export SGLANG_NPU_USE_MLAPO=1
export ASCEND_MF_TRANSFER_PROTOCOL=device_rdma

    python3 -m sglang.launch_server --model-path /home/ckpt/Qwen3-32B \
    --trust-remote-code \
    --attention-backend ascend \
    --model-impl mindspore \
    --mem-fraction-static 0.8 \
    --tp-size 4 \
    --dp-size 1 \
    --nnodes 1 \
    --node-rank 0 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend ascend \
    --dist-init-addr <DECODE_HOST_IP>:6688 \
    --host <DECODE_HOST_IP> \
    --port 8001
```

### Mini_LB
```shell
export PYTHONPATH=/home/sglang-mindspore/python:$PYTHONPATH
export PYTHONPATH=/home/sglang-mindspore/sgl-router/py_src:$PYTHONPATH


python -m sglang_router.launch_router \
--pd-disaggregation \
--mini-lb \
--prefill http://<PREFILL_HOST_IP>:8000 6657 \
--decode http://<DECODE_HOST_IP>:8001 \
--host 127.0.0.1 \
--port 5000
```

### Send request

```shell
curl http://127.0.0.1:5000/generate   -H "Content-Type: application/json"   -d '{ "text": "You are a helpful assistant.<｜User｜>Classify the text as neutral, negative or positive. \nText: I think this vacation is okay. \nSentiment:<｜Assistant｜><think>\n" }'
```

## Support
For MindSpore-specific issues:

- Refer to the [MindSpore documentation](https://www.mindspore.cn/)
