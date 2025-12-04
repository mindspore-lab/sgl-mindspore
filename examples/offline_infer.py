import argparse

import sglang as sgl

parser = argparse.ArgumentParser("sglang-mindspore offline infer")

parser.add_argument(
    "--model-path",
    required=False,
    default="/home/ckpt/Qwen3-8B",
    help="the model path",
    type=str,
)

args = parser.parse_args()


def main():
    llm = sgl.Engine(
        model_path=args.model_path,
        device="npu",
        model_impl="mindspore",
        attention_backend="ascend",
        tp_size=1,
        dp_size=1,
        log_level="INFO",
        mem_fraction_static=0.8,
    )

    prompts = [
        "Please introduce what MindSpore is.",
        "请介绍一下MindSpore。",
    ]

    sampling_params = {"temperature": 0, "top_p": 0.9}

    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")


if __name__ == "__main__":
    main()
