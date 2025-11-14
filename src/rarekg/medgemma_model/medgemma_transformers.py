import gc
import os
import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList


MODEL_DIR = os.path.expandvars("/scratch/${USER}/medgemma-4b-it")

class StopOnSubstrings(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings):
        self.tokenizer = tokenizer
        self.stop_strings = tuple(stop_strings or [])

    def __call__(self, input_ids, scores, **kwargs):
       
        text = self.tokenizer.decode(input_ids[0][-200:], skip_special_tokens=True)
        return any(stop_str in text for stop_str in self.stop_strings)

def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def _is_oom(e: Exception) -> bool:
    m = str(e).lower()
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in m or "cuda oom" in m



def load_model():
    tok = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        use_fast=True,
        trust_remote_code=True
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    try:
        _cleanup()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to("cuda")
        mode = f"fp{16 if dtype == torch.float16 else 'bf16'}"
        print("Loaded model ok - using dtype:", dtype)

    except Exception as e:
        if not _is_oom(e):
            raise
        print("OOM on fp16/bf16 load. Falling back to bitsandbytes 4-bit…")
        _cleanup()
        try:
            bnb4 = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_DIR,
                trust_remote_code=True,
                local_files_only=True,
                device_map="auto",
                quantization_config=bnb4,
                low_cpu_mem_usage=True,
            )
            mode = "4bit_nf4_auto"
            print("Loaded with bitsandbytes 4-bit (NF4).")

        except Exception as e4:
            if _is_oom(e4):
                print("Still OOM on 4-bit. Trying 8-bit…")
            else:
                print(f"4-bit failed ({type(e4).__name__}: {e4}). Trying 8-bit…")
            _cleanup()
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_DIR,
                    trust_remote_code=True,
                    local_files_only=True,
                    device_map="auto",
                    load_in_8bit=True,
                    low_cpu_mem_usage=True,
                )
                mode = "8bit_auto"
                print("Loaded with bitsandbytes 8-bit.")
            except Exception as e8:
                print(f"8-bit failed ({type(e8).__name__}: {e8}). Trying auto offload…")
                _cleanup()
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_DIR,
                    trust_remote_code=True,
                    local_files_only=True,
                    device_map="auto",
                    max_memory={0: "90%", "cpu": "64GiB"},
                    torch_dtype="auto",
                    low_cpu_mem_usage=True,
                )
                mode = "half_auto_offload"
                print("Loaded with auto offload (GPU+CPU).")

    print("Mode:", mode)
    model.eval() 
    return model, tok, mode


""" 
    Inference with Greedy approach
"""
def chat(model, tok, messages):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt_text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True, 
    )

    enc = tok(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    
    stopping_criteria = StoppingCriteriaList([StopOnSubstrings(tok, ["]}"])])

    # Generating in inference mode 
    with torch.inference_mode():
        out_ids = model.generate(
            **enc,
            max_new_tokens=400, # might affect the outputs 
            do_sample=False,               
            top_p=1.0,
            repetition_penalty=1.0,
            # no_repeat_ngram_size=3,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )

    new_tokens = out_ids[0, enc["input_ids"].shape[-1]:]
    out_text = tok.decode(new_tokens, skip_special_tokens=True).strip()
 
    try:
        start = out_text.find('{')
        end = out_text.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = out_text[start:end]
            return json.dumps(json.loads(json_str), indent=2)
        return out_text
    except:
        return out_text

""" Inference with sampling approch instead of greedy """

def chat_sampling(model, tok, messages):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt_text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True, 
    )

    enc = tok(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    
    stopping_criteria = StoppingCriteriaList([StopOnSubstrings(tok, ["]}"])])

    # Generating in inference mode 
    with torch.inference_mode():
        out_ids = model.generate(
            **enc,
            max_new_tokens=300,
            do_sample=True,          
            temperature=0.2,         
            top_p=0.9,              
            #no_repeat_ngram_size=3,
            repetition_penalty=1.05,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )

    new_tokens = out_ids[0, enc["input_ids"].shape[-1]:]
    out_text = tok.decode(new_tokens, skip_special_tokens=True).strip()
 
    try:
        start = out_text.find('{')
        end = out_text.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = out_text[start:end]
            return json.dumps(json.loads(json_str), indent=2)
        return out_text
    except:
        return out_text

