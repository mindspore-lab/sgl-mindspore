# Qwen3 MoE Guide (MoE Parallel)

## Prerequisites
- SGLang and this repository (sgl-mindspore) are installed in the same Python environment.
- Ascend NPU environment is available with HCCL communication.
- Model weights are a Qwen3 MoE variant (e.g., `Qwen3-30B-A3B`) and accessible.
- Expert Parallel (EP) prerequisites:
  - Install the corresponding packages required by your environment.
  - EP parallel has been validated only on Atlas A3 Training Series products and Atlas A3 Inference Series products.
  - Package sources :
    - MindSpore index: `https://repo.mindspore.cn/mindspore/mindspore/version/202512/20251211/master_20251211160023_a45d7b87e3bcc5905cfe93e2a61acc2d4108b693_newest/`
    - ms_custom_ops index: `https://repo.mindspore.cn/mindspore/ms_custom_ops/version/202512/20251222/ms_2.8.0_20251222050508_a7c3a2cfb0640bb6912329ed40a7b6497019923b_newest/`

## Environment and Variables
- HCCL_BUFFSIZE (Atlas A3 Training/Inference Series):
  - Check the environment value before use. It represents the memory size per communication domain, in MB. If not set, the default is 200 MB.
  - Requirement (must satisfy both):
    - `HCCL_BUFFSIZE >= 2`
    - `HCCL_BUFFSIZE >= 2 * ( localExpertNum * maxBs * epWorldSize * Align512( Align32(2 * H) + 64 ) + (K + sharedExpertNum) * maxBs * Align512(2 * H) )`
  - Definitions:
    - `Align512(x) = ((x + 512 - 1) / 512) * 512`
    - `Align32(x) = ((x + 32 - 1) / 32) * 32`
  - Reference: `https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/API/aolapi/context/aclnnMoeDistributeDispatchV3.md`
  - Example:
    ```shell
    export HCCL_BUFFSIZE=2000
    ```
- Optionally set `ASCEND_RT_VISIBLE_DEVICES` to restrict visible devices.
- For multi-node setups, ensure network connectivity and a correct init address.

## Launch Server (MoE + Tensor Parallel)
- Use the Ascend attention backend with the MindSpore implementation, enabling Expert Parallel (EP) and Tensor Parallel (TP):
  ```shell
  python3 -m sglang.launch_server --model-path /home/ckpt/Qwen3-30B-A3B \
      --host 0.0.0.0 \
      --device npu \
      --attention-backend ascend \
      --model-impl mindspore \
      --mem-fraction-static 0.8 \
      --chunked-prefill-size 512 \
      --tp-size 16 \
      --ep-size 16 \
      --dist-init-addr <HOST_IP>:6688 \
      --port 8000
  ```

## Send Request Example
```shell
curl http://0.0.0.0:8000/generate \
  -H "Content-Type: application/json" \
  -d '{ "text": "You are a helpful assistant.<｜User｜>Classify the text as neutral, negative, or positive. \nText: I think this vacation is okay. \nSentiment:<｜Assistant｜><think>\n" }'
```

## Benchmark Example (Single Batch)
```shell
python3 -m sglang.bench_one_batch_server \
    --model-path /home/ckpt/Qwen3-30B-A3B \
    --base-url http://0.0.0.0:8000 \
    --batch-size 128 \
    --input-len 256 \
    --output-len 256 \
    --skip-warmup
```

## Support
For MindSpore-specific issues:

- Refer to the [MindSpore documentation](https://www.mindspore.cn/)
