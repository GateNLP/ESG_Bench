"""Implement HuggingfaceModel models."""
import copy
import logging
from collections import Counter
import torch

import accelerate

from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList
from huggingface_hub import snapshot_download
from utils.base_model import BaseModel, STOP_SEQUENCES


class StoppingCriteriaSub(StoppingCriteria):
    """Stop generations when they match a particular text or token."""
    def __init__(self, stops, tokenizer, match_on='text', initial_length=None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        if self.match_on == 'tokens':
            self.stops = [torch.tensor(self.tokenizer.encode(i)).to('cuda') for i in self.stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del scores  # `scores` arg is required by StoppingCriteria but unused by us.
        for stop in self.stops:
            if self.match_on == 'text':
                generation = self.tokenizer.decode(input_ids[0][self.initial_length:], skip_special_tokens=False)
                match = stop in generation
            elif self.match_on == 'tokens':
                # Can be dangerous due to tokenizer ambiguities.
                match = stop in input_ids[0][-len(stop):]
            else:
                raise
            if match:
                return True
        return False


def remove_split_layer(device_map_in):
    """Modify device maps s.t. individual layers are not spread across devices."""

    device_map = copy.deepcopy(device_map_in)
    destinations = list(device_map.keys())

    counts = Counter(['.'.join(i.split('.')[:2]) for i in destinations])

    found_split = False
    for layer, count in counts.items():
        if count == 1:
            continue

        if found_split:
            # Only triggers if we find more than one split layer.
            raise ValueError(
                'More than one split layer.\n'
                f'Currently at layer {layer}.\n'
                f'In map: {device_map_in}\n'
                f'Out map: {device_map}\n')

        logging.info(f'Split layer is {layer}.')

        # Remove split for that layer.
        for name in list(device_map.keys()):
            if name.startswith(layer):
                device = device_map.pop(name)

        device_map[layer] = device
        found_split = True

    return device_map

def dataset_format(input_data, brief_prompt, model_name):
    if brief_prompt == 'default':
        system_message = 'Answer the following question in a single brief but complete sentence.'
    elif brief_prompt == 'chat':
        system_message = "You are going to read a report, the report starts with [S_REPORT] and ends with [E_REPORT]. Then answer the question followed by the report, the answer should be brief and do not provide any explanations. If you can't find the answer from the report, please answer not provided."
    elif brief_prompt == 'cot_2steps':
        system_message = "Think step by step: 1. Determine if the report provides an answer to the question 2. If you can' find the answer from the report, please answer not provided. Based on your reasoning, the correct answer should be:"
    elif brief_prompt == 'cot_4steps':
        system_message = f"Think step by step: 1. Identify the key topic or entity mentioned in the question. 2. Search the report for sentences or paragraphs relevant to that topic. 3. Determine if the report provides an answer to the question 4. Based on your reasoning, answer the question. If you can't find the answer from the report, please answer not provided."
    if "gemma" in model_name:
        # formatted_chat = [{"role": "user", "content": f"system message:{system_message} user:{input_data}"}]
        formatted_chat = [
            {"role": "user", "content": f"{input_data}"},
        ]        
    else:
        formatted_chat = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"{input_data}"},
        ]

    return formatted_chat

class HuggingfaceModel(BaseModel):
    def __init__(self, cfg):
        # required config
        self.max_new_tokens = cfg.max_new_tokens
        self.token_layer = getattr(cfg, "token_layer", 17)

        family  = cfg.family
        dataset = cfg.dataset
        variant = cfg.variant
        
        # Determine model path
        if variant == "before_finetune":
            model_path = cfg.paths[family]["before_finetune"]
        else:
            model_path = cfg.paths[family][dataset][variant]

        # Save a readable model_name
        self.model_name = f"{family}_{dataset}_{variant}"

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            device_map="auto",
            token_type_ids=None
        )

        # Load actual model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto"
        )

        # stop sequences
        stop_sequences = getattr(cfg, "stop_sequences", "default")

        if stop_sequences == "default":
            stop_sequences = STOP_SEQUENCES

        if stop_sequences is None:
            stop_sequences = []
        elif isinstance(stop_sequences, str):
            stop_sequences = [stop_sequences]

        self.stop_sequences = stop_sequences + [self.tokenizer.eos_token]

        # token limit
        self.token_limit = getattr(cfg, "token_limit", 10000)


    def predict(self, input_data, temperature, brief_prompt, return_full=False):
        tokenizer = self.tokenizer
        model_name = self.model_name.lower()
        formatted_data = dataset_format(input_data, brief_prompt, model_name)
        tokenized_chat = tokenizer.apply_chat_template(formatted_data, add_generation_prompt=True, return_tensors="pt", max_length = 14000, return_dict=True).to("cuda") 
        inputs = tokenized_chat
        tokenizer.pad_token_id = tokenizer.eos_token_id  # Use EOS as padding

        if 'llama' in self.model_name.lower() or 'mistral' in self.model_name.lower() or 'gemma' in self.model_name.lower():
            if 'token_type_ids' in inputs:  # Some HF models have changed.
                del inputs['token_type_ids']
            pad_token_id = self.tokenizer.eos_token_id
        else:
            # pad_token_id = None
            pad_token_id = self.tokenizer.pad_token_id

        if self.stop_sequences is not None:
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
                stops=self.stop_sequences,
                initial_length=len(inputs['input_ids'][0]),
                tokenizer=self.tokenizer)])
        else:
            stopping_criteria = None

        
        logging.debug('temperature: %f', temperature)
        if temperature == 0:
            do_sample = False
        else:
            do_sample = True
        
        with torch.no_grad():
            
            outputs = self.model.generate(
                # **inputs,
                **tokenized_chat,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                temperature=temperature,
                do_sample = do_sample,
                # stopping_criteria=stopping_criteria,
                stopping_criteria = None,
                pad_token_id=pad_token_id,
            )

        if isinstance(outputs, list): 
            generated_sequence = outputs[0]
        else:
            generated_sequence = outputs.sequences[0]
        
        if len(generated_sequence) > self.token_limit:
            # raise ValueError(
            #     'Generation exceeding token limit %d > %d',
            #     len(outputs.sequences[0]), self.token_limit)
            print(f'Generation exceeding token limit {len(outputs.sequences[0])} > {self.token_limit}')

        full_answer = self.tokenizer.decode(
            generated_sequence, skip_special_tokens=True)

        if return_full:
            return full_answer

        # For some models, we need to remove the input_data from the answer.
        if "gemma" in self.model_name.lower():
            start_index = full_answer.find('model\n')
            input_data_offset = start_index + len('model\n')
            if start_index != -1:
                answer = full_answer[input_data_offset:]
            else:
                raise ValueError("Pattern 'model\\n\\n' not found in the full_answer")
        elif "mistral" in self.model_name.lower():
            start_index = full_answer.find('Answer:')
            input_data_offset = start_index + len('Answer:')
            # input_data_offset = start_index
            if start_index != -1:
                answer = full_answer[input_data_offset:]
            else:
                raise ValueError("Pattern 'Answer:' not found in the full_answer")
        else:
            start_index = full_answer.find('assistant\n\n')
            input_data_offset = start_index + len('assistant\n\n')
            # input_data_offset = start_index
            if start_index != -1:
                answer = full_answer[input_data_offset:]
            else:
                raise ValueError("Pattern 'assistant\\n\\n' not found in the full_answer")

        sliced_answer = answer.strip()
        
        # Remove stop_words from answer.
        stop_at = len(answer)
        sliced_answer = answer

        if self.stop_sequences is not None:
            for stop in self.stop_sequences:
                if answer.endswith(stop):
                    stop_at = len(answer) - len(stop)
                    sliced_answer = answer[:stop_at]
                    break
            if not all([stop not in sliced_answer for stop in self.stop_sequences]):
                error_msg = 'Error: Stop words not removed successfully!'
                error_msg += f'Answer: >{answer}< '
                error_msg += f'Sliced Answer: >{sliced_answer}<'
                if 'llama' not in self.model_name.lower() and 'falcon' not in self.model_name.lower() and 'gemma' not in self.model_name.lower() and 'dpo' not in self.model_name.lower():
                    raise ValueError(error_msg)
                else:
                    logging.error(error_msg)
        
        # Remove whitespaces from answer (in particular from beginning.)
        sliced_answer = sliced_answer.strip()
        print('sliced_answer:',answer)
        
        # Get the number of tokens until the stop word comes up.
        # Note: Indexing with `stop_at` already excludes the stop_token.
        # Note: It's important we do this with full answer, since there might be
        # non-trivial interactions between the input_data and generated part
        # in tokenization (particularly around whitespaces.)
        token_stop_index = self.tokenizer(full_answer[:input_data_offset + stop_at], return_tensors="pt")['input_ids'].shape[1]
        # n_input_token = len(inputs['input_ids'][0])
        n_input_token = self.tokenizer(full_answer[:input_data_offset],return_tensors="pt")['input_ids'].shape[1]
        n_generated = token_stop_index - n_input_token
        
        
        if n_generated == 0:
            logging.warning('Only stop_words were generated. For likelihoods and embeddings, taking stop word instead.')
            n_generated = 1
        
        
        if 'decoder_hidden_states' in outputs.keys():
            hidden = outputs.decoder_hidden_states
        else:
            hidden = outputs.hidden_states

        if len(hidden) == 1:
            logging.warning(
                'Taking first and only generation for hidden! '
                # 'n_generated: %d, n_input_token: %d, token_stop_index %d, '
                'last_token: %s, generation was: %s',
                # n_generated, n_input_token, token_stop_index,
                self.tokenizer.decode(outputs['sequences'][0][-1]),
                full_answer,
                )
            last_input = hidden[0]
        elif ((n_generated - 1) >= len(hidden)):
            # If access idx is larger/equal.
            logging.error(
                'Taking last state because n_generated is too large'
                # 'n_generated: %d, n_input_token: %d, token_stop_index %d, '
                'last_token: %s, generation was: %s, slice_answer: %s',
                # n_generated, n_input_token, token_stop_index,
                self.tokenizer.decode(outputs['sequences'][0][-1]),
                full_answer, sliced_answer
                )
            last_input = hidden[-1]
        else:
            last_input = hidden[n_generated - 1]

        # Then access last layer for input
        middle_layer_embedding = last_input[self.token_layer]
        last_token_embedding = middle_layer_embedding[:, -1, :].cpu()

        # Get log_likelihoods.
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True)

        log_likelihoods = [score.item() for score in transition_scores[0]]
        if len(log_likelihoods) == 1:
            logging.warning('Taking first and only generation for log likelihood!')
            log_likelihoods = log_likelihoods
        else:
            log_likelihoods = log_likelihoods[:n_generated]

        if len(log_likelihoods) == self.max_new_tokens:
            logging.warning('Generation interrupted by max_token limit.')

        if len(log_likelihoods) == 0:
            raise ValueError
        
        return sliced_answer, log_likelihoods, last_token_embedding
