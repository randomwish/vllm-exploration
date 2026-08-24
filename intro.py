from vllm import LLM, SamplingParams


def main():

    llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct", gpu_memory_utilization=0.3,
              max_model_len=512, enforce_eager=True)

    prompts = [f"Tell me one fact about the number {i}." for i in range(4)]

    params = SamplingParams(max_tokens=20, temperature=1.5)  # runs A1, A2
    # params = SamplingParams(max_tokens=20, temperature=1.5)  # runs B1, B2

    outputs = llm.generate(prompts, params)

    for o in outputs:
        print("TEXT:", repr(o.outputs[0].text))



if __name__ == "__main__":
    main()