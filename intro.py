import vllm_source as vllm
from vllm import LLM, SamplingParams


def main():
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=8,
    )

    llm = LLM(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        gpu_memory_utilization=0.5,
        max_model_len=512,
    )

    messages = [
        {"role": "user", "content": "Say hello in three words!"}
    ]

    prompt = llm.get_tokenizer().apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    outputs = llm.generate(
        [prompt],
        sampling_params,
    )


    result = outputs[0].outputs[0]

    print("TEXT:", repr(result.text))
    print("TOKEN IDS:", result.token_ids)
    print("FINISH REASON:", result.finish_reason)
    print("STOP REASON:", result.stop_reason)
    print("OUTPUT:", repr(outputs[0].outputs[0].text))

if __name__ == "__main__":
    main()