import transformers
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM


def load(model_name):
    print("llava.model.language_model.llava_llama")
    print(f"Loading LLaVA model with LLM backbone: {model_name}")
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_name,
        attn_implementation=None,
        torch_dtype=None,
    )
    print(f"Loaded model {model_name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="lmsys/vicuna-7b-v1.5")
    args = parser.parse_args()

    load(args.model_name)
