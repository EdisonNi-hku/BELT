# Mostly taken from: https://github.com/iie-ycx/DEER/blob/main/vanilla_deer.py

from transformers.cache_utils import DynamicCache
import transformers
from typing import Any
import torch.nn.functional as F
import torch
from copy import deepcopy
from cache_utils import cache_to_device


def calcu_max_probs_w_kv(model, pred_input_ids, kv_cache, tokenizer, method=0, tot_step_cap=20):
    # method==0 is Qwen3, method==1 is everything else
    if method == 0:
        stop_words = ['</think>']
    else: # method == 1
        stop_words = ['</think>', '\n', '}\n', '\n\n', '}.', '.\n', '.\n\n']
    stop_ids = []
    for word in stop_words:
        stop_ids.extend(tokenizer(word, add_special_tokens=False)['input_ids'])
    
    total_steps = 0
    total_prob_max = 0.0

    pred_tokens, token_probs = [], []
    last_token = -1
   
    # Move cache to GPU for computation (deepcopy to preserve original)
    if kv_cache is not None:
        curr_kv_cache = cache_to_device(deepcopy(kv_cache), pred_input_ids.device)
    else:
        curr_kv_cache = None
    torch.cuda.empty_cache()
   
    with torch.no_grad():
        while last_token not in stop_ids:
            if last_token == -1:
                output_dicts = model(input_ids=pred_input_ids, past_key_values=curr_kv_cache)
            else:
                output_dicts = model(
                    input_ids=torch.tensor([last_token]).unsqueeze(0).to(pred_input_ids.device),
                    past_key_values=curr_kv_cache
                )
            logits = output_dicts['logits'][0][-1]
            curr_kv_cache = output_dicts['past_key_values']
            probs = F.softmax(logits, dim=-1)
            max_value, max_index = torch.max(probs, dim=0)

            if last_token == -1:
                total_prob_max = total_prob_max
            else:
                if method == 0:
                    total_prob_max += torch.log(max_value)
                else:
                    total_prob_max += max_value
            
            pred_tokens.append(max_index)
            token_probs.append(max_value.item())
            last_token = max_index

            total_steps += 1
            if total_steps > tot_step_cap:
                break
    
    # Check if generation ended with </think>
    ended_with_think = (last_token in stop_ids)

    if method == 0:
        total_prob_max = torch.exp((total_prob_max - torch.log(max_value)) / (total_steps - 1))
    else:
        total_prob_max = (total_prob_max - max_value) / (total_steps - 1)

    # Decode the predicted answer
    pred_token_ids = [t.item() for t in pred_tokens]
    predicted_answer = tokenizer.decode(pred_token_ids, skip_special_tokens=True)
    
    # Decode the prompt that was used
    prompt_text = tokenizer.decode(pred_input_ids[0], skip_special_tokens=True)

    del curr_kv_cache
    if 'output_dicts' in locals():
        del output_dicts
    torch.cuda.empty_cache() 
    
    return {
        'total_prob_max': total_prob_max.item(),
        'predicted_answer': predicted_answer,
        'token_ids': pred_token_ids,
        'token_probs': token_probs,
        'ended_with_think': ended_with_think,
    }


def method_deer(model, tokenizer, sample: dict[str, Any], B: int, sample_response: dict[str, Any], threshold: float, stop_word: str = "Wait", ewt: bool = True, cp_cache: bool = True) -> dict[str, Any]:
    content = sample['question'] + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    messages = [{'role': 'user', 'content': content}]
    if 'nemotron' in model.name_or_path.lower():
        messages = [{"role": "system", "content": f"detailed thinking on"}] + messages
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    if transformers.__version__.split('.')[0] == '5' and hasattr(prompt_ids, 'input_ids'): prompt_ids = prompt_ids['input_ids'] # change in apply_chat_template for transformersv5
    stop_tokens = tokenizer(stop_word, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    sample_response_ids = tokenizer(sample_response['method_output']['response'])["input_ids"]
    stop_token_idxs = list(i for i, tok in enumerate(sample_response_ids) if tok in stop_tokens)
    
    if sample.get('is_lcb'):
        trial_answer_ids = torch.tensor(
            tokenizer("\n**Final Answer**\nThe final code is:\n```python\n", add_special_tokens=False)["input_ids"]
        ).unsqueeze(0).to(model.device)
        final_answer_ids = tokenizer("</think>\n**Final Answer**\nThe final code is:\n```python\n", add_special_tokens=False)["input_ids"]
        max_new_tokens = 300
    else:
        trial_answer_ids = torch.tensor(
            tokenizer("\n**Final Answer**\n\nThe final answer is \\boxed", add_special_tokens=False)["input_ids"]
        ).unsqueeze(0).to(model.device)
        final_answer_ids = tokenizer("</think>\n**Final Answer**\n\nThe final answer is \\boxed", add_special_tokens=False)["input_ids"]
        max_new_tokens = 30
    
    past_key_values = DynamicCache() if cp_cache else None
    prob_checks = []

    prev_stop_token_idx = 0
    for i, stop_token_idx in enumerate(stop_token_idxs):
        if cp_cache:
            # fill in KV cache up to this point
            if prev_stop_token_idx == 0:
                input_ids = torch.tensor(prompt_ids+sample_response_ids[:stop_token_idx]).unsqueeze(0).to(model.device)
            else:
                input_ids = torch.tensor(sample_response_ids[prev_stop_token_idx:stop_token_idx]).unsqueeze(0).to(model.device)
            prev_stop_token_idx = stop_token_idx
            with torch.no_grad():
                # Move cache to GPU for forward pass
                gpu_cache = cache_to_device(past_key_values, model.device)
                outputs = model(input_ids=input_ids, past_key_values=gpu_cache)
                # Move updated cache back to CPU to save GPU memory
                past_key_values = cache_to_device(outputs['past_key_values'], 'cpu')
            del outputs, gpu_cache
            torch.cuda.empty_cache()

            # check answer confidence at this point
            prob_result = calcu_max_probs_w_kv(model, trial_answer_ids, past_key_values, tokenizer, method=(0 if ewt else 1), tot_step_cap=(300 if sample.get('is_lcb') else 20))
        else:
            full_pred_input_ids = torch.tensor(prompt_ids + sample_response_ids[:stop_token_idx] + trial_answer_ids[0].tolist()).unsqueeze(0).to(model.device)
            prob_result = calcu_max_probs_w_kv(model, full_pred_input_ids, None, tokenizer, method=(0 if ewt else 1), tot_step_cap=(300 if sample.get('is_lcb') else 20))

        prob_checks.append(prob_result)

        # If exceeding threshold, stop early and generate final answer
        if (not ewt or prob_result['ended_with_think']) and prob_result['total_prob_max'] > threshold:
            input_ids = torch.tensor(prompt_ids + sample_response_ids[:stop_token_idx] + final_answer_ids).unsqueeze(0).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    return_dict_in_generate=True, 
                    past_key_values=cache_to_device(past_key_values, 'cuda') if cp_cache else None,
                )
            response_tokens = len(outputs.sequences[0]) - len(prompt_ids)
            work = response_tokens + sum(len(prob_check['token_ids']) for prob_check in prob_checks)
            return {
                'response': tokenizer.decode(outputs.sequences[0][len(prompt_ids):], skip_special_tokens=True),
                'prob_checks': prob_checks,
                'threshold': threshold,
                'stopped_early': True,
                'response_tokens': response_tokens,
                'work': work,
                'sample_response': sample_response['method_output']['response'],
            }
    
    # Did not stop early, return original response
    response_tokens = len(tokenizer.encode(sample_response['method_output']['response']))
    work = response_tokens + sum(len(prob_check['token_ids']) for prob_check in prob_checks)
    return {
        'response': sample_response['method_output']['response'],
        'prob_checks': prob_checks,
        'threshold': threshold,
        'stopped_early': False,
        'response_tokens': response_tokens,
        'work': work,
        'sample_response': sample_response['method_output']['response'],
    }
