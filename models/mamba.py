from transformers import MambaForCausalLM, AutoTokenizer, GenerationConfig

def generate_answer_mamba(question : str, base_context: str):
    if not base_context == "":
        base_context_str = " ".join(base_context)
        prompt = question + "rag - " + base_context_str
    else:
        prompt = question

    print(f"Prompt: {prompt}")

    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-1.4b-hf")
    model = MambaForCausalLM.from_pretrained("state-spaces/mamba-1.4b-hf")

    inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=512)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    gen_config = GenerationConfig(top_p=1.0, do_sample=False, num_beams=1)

    output = model.generate(input_ids, max_new_tokens=30, attention_mask=attention_mask, generation_config=gen_config)
    
    generated_only = output[0, input_ids.shape[1]:]

    full_text = tokenizer.decode(generated_only, skip_special_tokens=True)

    full_text = full_text.split(".")[0].strip() + "."

    print("Full Text: ", full_text)

    return full_text
