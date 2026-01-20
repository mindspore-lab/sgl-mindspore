import sglang as sgl


def main():
    llm = sgl.Engine(
        model_path="/home/ckpt/Qwen3-8B",
        device="npu",
        model_impl="mindspore",
        attention_backend="ascend",
        tp_size=1,
        dp_size=1,
        log_level="INFO",
        mem_fraction_static=0.75,
        port=24678,
        disable_cuda_graph=True,
        speculative_draft_model_path="/home/ckpt/Qwen3-8B_eagle3",
        speculative_algorithm="EAGLE3",
        speculative_num_steps=4,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=5,
    )

    prompts = [
        "what is mindspore?",
    ]

    sampling_params = {"temperature": 0, "top_p": 1.0}

    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
    same = outputs[0]["text"] == outputs[1]["text"]
    print(f"Identical outputs: {same}")


if __name__ == "__main__":
    main()
