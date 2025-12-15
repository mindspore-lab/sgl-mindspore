# MindSpore Models

## Introduction

SGLang support run MindSpore framework models, this doc guide users to run mindspore models with SGLang.

## Requirements

MindSpore with SGLang current only support Ascend Npu device, users need first install Ascend CANN software packages.
The CANN software packages can download from the [Ascend Official Websites](https://www.hiascend.com). The version depends on the MindSpore version [MindSpore Installation](https://www.mindspore.cn/install)

## Supported Models

Currently, the following models are supported:

- **Qwen3**: Dense models supported. MoE models coming soon.
- *More models coming soon...*

## Installation

You will need to install the following packages, due to the support of tensor conversion through `dlpack` on 3rd devices, the minimum version of  `PyTorch` is 2.7.1

```shell
pip install mindspore
pip install torch==2.7.1
pip install torch_npu==2.7.1rc1
pip install triton_ascend
pip install torchvision==0.22.1
```

```shell
pip install -e "python[all_npu]"
```

## Run Model

Current SGLang-MindSpore support Qwen3 dense model, this doc uses Qwen3-8B as example.

### Offline infer

Use the following script for offline infer:

```python
import sglang as sgl

# Initialize the engine with MindSpore backend
llm = sgl.Engine(
    model_path="/path/to/your/model",  # Local model path
    device="npu",                      # Use NPU device
    model_impl="mindspore",            # MindSpore implementation
    attention_backend="ascend",        # Attention backend
    tp_size=1,                         # Tensor parallelism size
    dp_size=1                          # Data parallelism size
)

# Generate text
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "The future of AI is"
]

sampling_params = {"temperature": 0.01, "top_p": 0.9}
outputs = llm.generate(prompts, sampling_params)

for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}")
    print(f"Generated: {output['text']}")
    print("---")
```

### Start server

Launch a server with MindSpore backend:

```bash
# Basic server startup
python3 -m sglang.launch_server \
    --model-path /path/to/your/model \
    --host 0.0.0.0 \
    --device npu \
    --model-impl mindspore \
    --attention-backend ascend \
    --tp-size 1 \
    --dp-size 1
```

For distributed server with multiple nodes:

```bash
# Multi-node distributed server
python3 -m sglang.launch_server \
    --model-path /path/to/your/model \
    --host 0.0.0.0 \
    --device npu \
    --model-impl mindspore \
    --attention-backend ascend \
    --dist-init-addr 127.0.0.1:29500 \
    --nnodes 2 \
    --node-rank 0 \
    --tp-size 4 \
    --dp-size 2
```

## Troubleshooting

#### Debug Mode

Enable sglang debug logging by log-level argument.

```bash
python3 -m sglang.launch_server \
    --model-path /path/to/your/model \
    --host 0.0.0.0 \
    --device npu \
    --model-impl mindspore \
    --attention-backend ascend \
    --log-level DEBUG
```

Enable mindspore info and debug logging by setting environments.

```bash
export GLOG_v=1  # INFO
export GLOG_v=0  # DEBUG
```

#### Explicitly select devices

Use the following environment variable to explicitly select the devices to use.

```shell
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7  # to set device
```

#### Some communication environment issues

In case of some environment with special communication environment, users need set some environment variables.

```shell
export MS_ENABLE_LCCL=off # current not support LCCL communication mode in SGLang-MindSpore
```

#### Some dependencies of protobuf

In case of some environment with special protobuf version, users need set some environment variables to avoid binary version mismatch.

```shell
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python  # to avoid protobuf binary version mismatch
```

## PD Disaggregation

#### Using docker
**Notice:** --privileged and --network=host are required by RDMA, which is typically needed by Ascend NPU clusters.
```shell
docker run -itd --privileged --network=host --name=npu_sgl_0.8 --net=host \
   --shm-size 500g \
   --device=/dev/davinci0 \
   --device=/dev/davinci1 \
   --device=/dev/davinci2 \
   --device=/dev/davinci3 \
   --device=/dev/davinci4 \
   --device=/dev/davinci5 \
   --device=/dev/davinci6 \
   --device=/dev/davinci7 \
   --device=/dev/davinci_manager \
   --device=/dev/hisi_hdc \
   --device /dev/devmm_svm \
   -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
   -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
   -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
   -v /usr/local/sbin:/usr/local/sbin \
   -v /etc/hccn.conf:/etc/hccn.conf \
   -v /home/ckpt:/home/ckpt \
   sgl_mindspore:v0.8 \
   bash

```
#### MemFabric Adaptor install
*Notice: Prebuilt wheel package is based on aarch64, please leave an issue [here at sglang](https://github.com/sgl-project/sglang/issues) to let us know the requests for amd64 build.*

MemFabric Adaptor is a drop-in replacement of Mooncake Transfer Engine that enables KV cache transfer on Ascend NPU clusters.

```shell
MF_WHL_NAME="mf_adapter-1.0.0-cp311-cp311-linux_aarch64.whl"
MEMFABRIC_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/sglang/${MF_WHL_NAME}"
wget -O "${MF_WHL_NAME}" "${MEMFABRIC_URL}" && pip install "./${MF_WHL_NAME}"
```


## PD Disaggregation Examples
### Running Qwen3-8B
Running Qwen3-8B with PD disaggregation on 2 x Atlas 800I A2. Model weights could be found [here](https://modelers.cn/models/MindSpore-Lab/Qwen3-8B).
#### Prefill:
```shell
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_MF_STORE_URL="tcp://<PREFILL_HOST_IP>:6650"
export PYTHONPATH=/home/sglang-mindspore/python:$PYTHONPATH
export ASCEND_MF_TRANSFER_PROTOCOL=device_rdma

    python3 -m sglang.launch_server --model-path /home/ckpt/Qwen3-8B \
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

    python3 -m sglang.launch_server --model-path /home/ckpt/Qwen3-8B \
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

#### Mini_LB
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

#### Send request

```shell
curl http://127.0.0.1:5000/generate   -H "Content-Type: application/json"   -d '{ "text": "You are a helpful assistant.<｜User｜>Classify the text as neutral, negative or positive. \nText: I think this vacation is okay. \nSentiment:<｜Assistant｜><think>\n" }'
```

## Support
For MindSpore-specific issues:

- Refer to the [MindSpore documentation](https://www.mindspore.cn/)
