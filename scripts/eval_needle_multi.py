from absl.app import run
import glob
import time
import json
import math
import os
from tqdm import tqdm
import random
from functools import cached_property
import numpy as np
import jax
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
import gcsfs
import tiktoken
from transformers import GenerationConfig, AutoTokenizer
from tux import (
    define_flags_with_default, StreamingCheckpointer, JaxDistributedConfig,
    set_random_seed, get_float_dtype_by_name, JaxRNG, next_rng,
    match_partition_rules, make_shard_and_gather_fns,
    with_sharding_constraint, tree_apply, open_file
)
from lwm.llama import LLaMAConfig, FlaxLLaMAForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import DiffLlamaForCausalLM
import torch

FLAGS, FLAGS_DEF = define_flags_with_default(
    haystack_file="",
    max_tokens_per_batch=2000000,
    output_file="results.json",
    context_lengths_min=1000,
    context_lengths_max=32000,
    n_context_length_intervals=3,
    n_document_depth_intervals=3,
    n_rounds=2,
    n_needles_total=4,
    n_needles_retrieve=4,
    seed=1234,
    # mesh_dim='1,-1,1,1',
    # dtype='fp32',
    # load_llama_config='',
    # update_llama_config='',
    load_checkpoint='',
    tokenizer='LargeWorldModel/LWM-Text-1M',
    # checkpointer=StreamingCheckpointer.get_default_config(),
    # llama=LLaMAConfig.get_default_config(),
    # jax_distributed=JaxDistributedConfig.get_default_config(),
)


class LLMNeedleHaystackTester:
    OURS_TEMPLATE = "You are a helpful assistant. USER: {context} {question} Don't give information outside the document. ASSISTANT: "
    RANDOM_NEEDLE_CITIES  = [
        'Chicago', 'Yangon', 'Antananarivo', 'Colombo', 'Almaty', 'Sydney', 'Chicago', 'Mexico City',
        'Seattle', 'Lagos', 'Amsterdam', 'Belgrade', 'Cairo', 'Baghdad', 'Damascus', 'Kigali', 'Dakar',
        'Dakar', 'Sofia', 'Kigali', 'Victoria', 'Tashkent', 'Mumbai', 'Barcelona', 'Almaty', 'Amman',
        'Toronto', 'Bratislava', 'Johannesburg', 'Thimphu', 'Bangkok', 'Santiago', 'Cairo', 'San Francisco',
        'Lagos', 'Amsterdam', 'Paris', 'Rabat', 'Santiago', 'Copenhagen', 'Madrid', 'Kigali',
        'Ho Chi Minh City', 'Sarajevo', 'Delhi', 'Istanbul', 'Ho Chi Minh City', 'Khartoum', 'Helsinki',
        'Doha', 'Istanbul', 'Kuala Lumpur', 'Budapest', 'Shanghai', 'Moscow', 'Los Angeles', 'Oslo',
        'Johannesburg', 'Berlin', 'Bangalore', 'Tokyo', 'Melbourne', 'Barcelona', 'Chicago', 'Port Louis',
        'Lisbon', 'Nairobi', 'Kampala', 'Lima', 'Maputo', 'Vancouver', 'Dubai', 'Khartoum', 'Jakarta',
        'Madrid', 'Yerevan', 'Beirut', 'Athens', 'Chicago', 'Paris', 'Bucharest', 'Copenhagen', 'Brussels',
        'Damascus', 'Seattle', 'Los Angeles', 'Yerevan', 'Victoria', 'Tunis', 'Astana', 'Seoul',
        'Buenos Aires', 'Bangkok', 'Colombo', 'Brussels', 'Khartoum', 'Doha', 'San Francisco', 'Vienna', 'Jakarta'
    ]

    def __init__(self,
                 needle="",
                 haystack_file="",
                 retrieval_question="What are the special magic numbers for {}?",
                 results_version = 1,
                 rnd_number_digits = 7,
                 context_lengths_min = 1000,
                 context_lengths_max = 126000,
                 context_lengths_num_intervals = 10,
                 document_depth_percent_min = 0,
                 document_depth_percent_max = 100,
                 document_depth_percent_intervals = 10,
                 document_depth_percent_interval_type = "linear",
                 save_results = False,
                 final_context_length_buffer = 200,
                 print_ongoing_status = True):
        needle="\nThe special magic {city} number is: {rnd_number}\n"
        self.needle = needle
        if not needle or not haystack_file or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")

        self.rnd_number_digits = rnd_number_digits
        self.context_lengths_num_intervals = context_lengths_num_intervals
        self.document_depth_percent_intervals = document_depth_percent_intervals
        self.haystack_file = haystack_file
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.print_ongoing_status = print_ongoing_status
        self.testing_results = []

        self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
        self.context_lengths = self.context_lengths.tolist()
        if document_depth_percent_interval_type == 'linear':
            self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
        elif document_depth_percent_interval_type == 'sigmoid':
            self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            raise ValueError(f"Unsupported document_depth_percent_interval_type: {document_depth_percent_interval_type}")
        self.document_depth_percents = self.document_depth_percents.tolist()

        self.model = Sampler(model_path=FLAGS.load_checkpoint, tokenizer_name=FLAGS.tokenizer)

        self.enc = AutoTokenizer.from_pretrained(FLAGS.tokenizer)
        self.enc_tiktoken = tiktoken.encoding_for_model("gpt-4-1106-preview")

    def generate_random_number(self, num_digits):
        lower_bound = 10**(num_digits - 1)
        upper_bound = 10**num_digits - 1
        return random.randint(lower_bound, upper_bound)

    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)

    def read_context_files(self, n):
        max_context_length = max(self.context_lengths)
        contexts = []
        f = open_file(self.haystack_file, 'r')
        for i in range(n):
            context = ""
            while len(self.enc.encode(context)) < max_context_length:
                context += json.loads(f.readline())['text']
            contexts.append(context)
        return contexts

    def encode_and_trim(self, context, context_length):
        tokens = self.enc.encode(context)
        if len(tokens) > context_length:
            context = self.enc.decode(tokens[:context_length])
        return context

    def create_contexts(self, needles_info, random_cities_retrieve, context, context_length, seed):
        assert all([random_city in needles_info for random_city in random_cities_retrieve])
        for random_city, (needle_rnd_number, depth_percent) in needles_info.items():
            context = self.generate_context(
                self.needle.format(city=random_city, rnd_number=needle_rnd_number),
                context, context_length, depth_percent
            )

        if len(random_cities_retrieve) == 1:
            question = f"What is the special magic number for {random_cities_retrieve[0]}?"
        else:
            q = ', '.join(random_cities_retrieve[:-1]) + ', and ' + random_cities_retrieve[-1]
            question = self.retrieval_question.format(q)
        results = {
            'context' : context,
            'context_length' : int(context_length),
            'needles_info': needles_info,
            'question' : question,
            'cities_to_retrieve' : random_cities_retrieve,
            'seed': seed,
         }
        return results

    def insert_needle(self, needle, context, depth_percent, context_length):
        tokens_needle = self.enc_tiktoken.encode(needle)
        tokens_context = self.enc_tiktoken.encode(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            period_tokens = self.enc_tiktoken.encode('.')

            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.enc_tiktoken.decode(tokens_new_context)
        return new_context

    def generate_context(self, needle, trim_context, context_length, depth_percent):
        context = self.insert_needle(needle, trim_context, depth_percent, context_length)
        return context

    def compute_max_input_length(self, context_length, buffer=1024):
        block_size = self.model.block_size
        context_length += buffer
        # context_length = 2 ** math.ceil(math.log2(context_length))
        context_length = math.ceil(context_length / block_size) * block_size
        return int(context_length)
    # for the 1st experiment
    # def run_test(self):
    #     fs = gcsfs.GCSFileSystem()
    #     contexts = []
    #     template = self.OURS_TEMPLATE

    #     def _key_from_result(result):
    #         return (result['context_length'], result['depth_percent'], result['seed'])

    #     results = []
    #     completed = set()
    #     def exists(fname):
    #         if fname.startswith('gs://'):
    #             return fs.exists(fname)
    #         else:
    #             return os.path.exists(fname)
    #     if exists(FLAGS.output_file):
    #         with open_file(FLAGS.output_file, 'r') as f:
    #             results = json.load(f)
    #             completed = set([_key_from_result(result) for result in results])
    #     print('completed', len(completed))

    #     full_contexts = self.read_context_files(FLAGS.n_rounds)
    #     full_tokens = [self.enc.encode(full_context) for full_context in full_contexts]

    #     start = time.time()
    #     for context_length in self.context_lengths:
    #         trim_contexts = [self.enc.decode(full_token[:context_length]) for full_token in full_tokens]
    #         max_input_length = self.compute_max_input_length(context_length)
    #         contexts = []
    #         for i in range(FLAGS.n_rounds):
    #             if (int(context_length), i) in completed:
    #                 continue
    #             random_cities = random.sample(LLMNeedleHaystackTester.RANDOM_NEEDLE_CITIES, FLAGS.n_needles_total)
    #             document_depths = random.sample(self.document_depth_percents, FLAGS.n_needles_total)
    #             random_cities_retrieve = random.sample(random_cities, FLAGS.n_needles_retrieve)
    #             needles_info = {}
    #             for random_city, depth_percent in zip(random_cities, document_depths):
    #                 needles_info[random_city] = (
    #                     str(self.generate_random_number(self.rnd_number_digits)),
    #                     depth_percent
    #                 )
    #             context = self.create_contexts(needles_info, random_cities_retrieve, trim_contexts[i], context_length, i)
    #             contexts.append(context)

    #         if len(contexts) == 0:
    #             continue

    #         B = FLAGS.max_tokens_per_batch / (max_input_length + self.model.block_size)
    #         B = int(B / self.model.data_dim) * self.model.data_dim
    #         if B < self.model.data_dim:
    #             B = self.model.data_dim
    #         elif B > len(contexts):
    #             B = int(math.ceil(len(contexts) / self.model.data_dim) * self.model.data_dim)
    #         n_pad = B - len(contexts) % B
    #         for _ in range(n_pad):
    #             contexts.insert(0, contexts[0])

    #         pbar = tqdm(total=len(contexts))
    #         for i in range(0, len(contexts), B):
    #             contexts_i = contexts[i:i + B]
    #             prompts = [
    #                 template.format(context=context['context'], question=context['question'])
    #                 for context in contexts_i
    #             ]
    #             outs = self.model(prompts, max_input_length)
    #             for j, (context, out) in enumerate(zip(contexts_i, outs)):
    #                 if i + j < n_pad:
    #                     continue
    #                 rnd_nums_to_retrieve = [
    #                     context['needles_info'][city][0] for city in context['cities_to_retrieve']
    #                 ]
    #                 results.append({
    #                     'context_length': context['context_length'],
    #                     'needles_info': context['needles_info'],
    #                     'question': context['question'],
    #                     'answer': rnd_nums_to_retrieve,
    #                     'response': out,
    #                     'correct': [rnd_num in out for rnd_num in rnd_nums_to_retrieve],
    #                     'seed': context['seed'],
    #                 })
    #                 print(results[-1]['correct'], out, rnd_nums_to_retrieve)
    #             if jax.process_index() == 0:
    #                 if FLAGS.output_file.startswith('gs://'):
    #                         fs = gcsfs.GCSFileSystem()
    #                         with fs.open(FLAGS.output_file, 'w') as f:
    #                             f.write(json.dumps(results, ensure_ascii=False, indent=2))
    #                 else:
    #                     with open(FLAGS.output_file, 'w', encoding='utf-8') as f:
    #                         json.dump(results, f, ensure_ascii=False, indent=2)
    #             pbar.update(len(contexts_i))
    #         pbar.close()
    #     print('elapsed', time.time() - start)
    #     print('done')


    # for the 2nd experiment
    # def run_test(self):
    #     from collections import defaultdict
    #     from tqdm import tqdm

    #     fs = gcsfs.GCSFileSystem()
    #     template = self.OURS_TEMPLATE
    #     results = []

    #     def exists(fname):
    #         return fs.exists(fname) if fname.startswith('gs://') else os.path.exists(fname)

    #     completed = set()
    #     if exists(FLAGS.output_file):
    #         with open_file(FLAGS.output_file, 'r') as f:
    #             results = json.load(f)
    #             completed = set(
    #                 (r['context_length'], r['depth_percent'], r['seed']) for r in results
    #             )

    #     full_contexts = self.read_context_files(FLAGS.n_rounds)
    #     full_tokens = [self.enc.encode(ctx) for ctx in full_contexts]

    #     start = time.time()
    #     for context_length in self.context_lengths:
    #         trim_contexts = [
    #             self.enc.decode(tokens[:context_length]) for tokens in full_tokens
    #         ]
    #         max_input_length = self.compute_max_input_length(context_length)

    #         for depth_percent in self.document_depth_percents:
    #             for i in range(FLAGS.n_rounds):
    #                 key = (context_length, depth_percent, i)
    #                 if key in completed:
    #                     continue

    #                 # 生成多个 needles，分布在任意深度（包括重复）
    #                 random_cities = random.sample(
    #                     LLMNeedleHaystackTester.RANDOM_NEEDLE_CITIES, FLAGS.n_needles_total
    #                 )
    #                 document_depths = random.choices(self.document_depth_percents, k=FLAGS.n_needles_total)

    #                 needles_info = {
    #                     city: (
    #                         str(self.generate_random_number(self.rnd_number_digits)),
    #                         doc_depth
    #                     )
    #                     for city, doc_depth in zip(random_cities, document_depths)
    #                 }

    #                 # 从所有 needles 中随机选择一个作为“答案针”
    #                 city_to_retrieve = random.choice(list(needles_info.keys()))
    #                 rnd_number_to_retrieve = needles_info[city_to_retrieve][0]

    #                 # 创建上下文
    #                 context_data = self.create_contexts(
    #                     needles_info,
    #                     [city_to_retrieve],  # only one answer needle for question
    #                     trim_contexts[i],
    #                     context_length,
    #                     i
    #                 )

    #                 prompt = template.format(
    #                     context=context_data['context'],
    #                     question=context_data['question']
    #                 )

    #                 out = self.model([prompt], max_input_length)[0]
    #                 correct = rnd_number_to_retrieve in out

    #                 result = {
    #                     'context_length': context_length,
    #                     'depth_percent': depth_percent,
    #                     'needles_info': needles_info,
    #                     'question': context_data['question'],
    #                     'answer': [rnd_number_to_retrieve],
    #                     'response': out,
    #                     'correct': [correct],
    #                     'seed': i
    #                 }
    #                 results.append(result)

    #                 if jax.process_index() == 0:
    #                     if FLAGS.output_file.startswith('gs://'):
    #                         with fs.open(FLAGS.output_file, 'w') as f:
    #                             f.write(json.dumps(results, ensure_ascii=False, indent=2))
    #                     else:
    #                         with open(FLAGS.output_file, 'w', encoding='utf-8') as f:
    #                             json.dump(results, f, ensure_ascii=False, indent=2)

    #                 print(result['correct'], out, rnd_number_to_retrieve)

    #     print('elapsed', time.time() - start)
    #     print('done')

    #     # === Accuracy stats ===
    #     accuracy_stats = defaultdict(list)
    #     for result in results:
    #         cl = result['context_length']
    #         dp = result['depth_percent']
    #         correct_flags = result['correct']
    #         for flag in correct_flags:
    #             accuracy_stats[(cl, dp)].append(flag)

    #     stats_output = []
    #     for (cl, dp), flags in sorted(accuracy_stats.items()):
    #         acc = np.mean(flags)
    #         stats_output.append({
    #             'context_length': cl,
    #             'document_depth_percent': dp,
    #             'accuracy': round(float(acc), 4)
    #         })

    #     stats_filename = FLAGS.output_file.replace('.json', '_stats.json')
    #     if stats_filename.startswith('gs://'):
    #         with fs.open(stats_filename, 'w') as f:
    #             f.write(json.dumps(stats_output, ensure_ascii=False, indent=2))
    #     else:
    #         with open(stats_filename, 'w', encoding='utf-8') as f:
    #             json.dump(stats_output, f, ensure_ascii=False, indent=2)

    #     print(f'\nAccuracy stats written to {stats_filename}')

    # for the 3rd experiment

    def get_diffllama_attention_components(model, tokenizer, text, device, layer_idx=-1, head_idx=0):
        """
        Specialized function to get differential attention components for DiffLlama.
        Focuses on calculating lambda and relies on the main model output for the final attention matrix.
        """
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        captured_components = {
            'lambda_val': None,
            'module_type_hooked': None,
            # A1, A2, and final_attention_hook are removed as they are not reliably capturable currently
        }


    def get_attention_scores(self, text, layer_idx=-1, head_idx=0):
        """
        Extract attention scores from model for the given text.
        Modified to work with the class structure of run_test.
        
        Args:
            text: input text to analyze
            layer_idx: which layer to analyze (-1 for last layer)
            head_idx: which attention head to analyze
            
        Returns:
            attention_matrix, tokens, metadata
        """
        model = self.model
        tokenizer = self.enc
        device = torch.device("cuda:0"),
        model_type = "llama" 
        
        if model_type == "diffllama":
            # first try to use the specialized DiffLlama method
            try:
                diffllama_attention, components = get_diffllama_attention_components(
                    model, tokenizer, text, device, layer_idx, head_idx
                )
                if diffllama_attention is not None:
                    tokens = [tokenizer.decode([token_id]) for token_id in 
                            tokenizer(text, return_tensors="pt", truncation=True, max_length=512)['input_ids'][0]]
                    
                    metadata = {
                        'layer_idx': layer_idx, 'head_idx': head_idx,
                        'seq_len': len(tokens), 'model_type': model_type,
                        'extraction_method': 'diffllama_specialized',
                        'lambda_value': components.get('lambda'),
                        'captured_components': list(components.keys())
                    }
                    
                    print(f"✅ Successfully extracted DiffLlama differential attention using specialized method")
                    return diffllama_attention, tokens, metadata
            except Exception as e:
                print(f"⚠️ DiffLlama specialized extraction failed: {e}, falling back to standard method")
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_ids = inputs['input_ids']
        
        # Get tokens for visualization
        tokens = [tokenizer.decode([token_id]) for token_id in input_ids[0]]
        
        # Temporarily enable attention output
        original_output_attentions = getattr(model.config, 'output_attentions', False)
        model.config.output_attentions = True
        
        attention_matrix = None
        
        # Initialize metadata dictionary
        metadata = {
            'layer_idx': layer_idx, 
            'head_idx': head_idx,
            'seq_len': len(tokens),
            'model_type': model_type,
            'captured_components': [],
            'lambda_std_dev': None,
            'lambda_params': {} 
        }
        
        if model_type == "diffllama":
            metadata['lambda_std_dev'] = getattr(model.config, 'lambda_std_dev', None)
        
        # For DiffLlama, set up specialized hooks
        captured_attention_weights = {}
        hooks = []
        
        if model_type == "diffllama":
            def create_diffllama_attention_hook(layer_name):
                def hook_fn(module, input, output):
                    captured_attention_weights[f"{layer_name}_module"] = module
                    
                    for attr_name in ['attn_weights', 'attention_weights', 'attention_probs']:
                        if hasattr(module, attr_name):
                            attr_value = getattr(module, attr_name)
                            if torch.is_tensor(attr_value) and len(attr_value.shape) == 4:
                                captured_attention_weights[f"{layer_name}_{attr_name}"] = attr_value.detach()
                    
                    if isinstance(output, tuple):
                        for i, out_tensor in enumerate(output):
                            if torch.is_tensor(out_tensor):
                                if len(out_tensor.shape) == 4:
                                    captured_attention_weights[f"{layer_name}_output_attn_{i}"] = out_tensor.detach()
                                elif len(out_tensor.shape) == 3:
                                    captured_attention_weights[f"{layer_name}_hidden_{i}"] = out_tensor.detach()
                    
                    captured_attention_weights[f"{layer_name}_full_output"] = output
                    return output
                return hook_fn
            
            target_layer_idx = layer_idx if layer_idx >= 0 else len(model.model.layers) + layer_idx
            
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                if 0 <= target_layer_idx < len(model.model.layers):
                    layer_to_hook = model.model.layers[target_layer_idx]
                    
                    for name, module in layer_to_hook.named_modules():
                        module_type_name = type(module).__name__
                        if 'DiffLlama' in module_type_name and ('Attention' in module_type_name or 'Attn' in module_type_name):
                            hook = module.register_forward_hook(
                                create_diffllama_attention_hook(f"layer_{target_layer_idx}_{name}")
                            )
                            hooks.append(hook)
        
        try:
            with torch.no_grad():
                outputs = model(**inputs)
                
                if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    attentions_from_output = outputs.attentions
                    effective_layer_idx = layer_idx if layer_idx >= 0 else len(attentions_from_output) + layer_idx
                    
                    if 0 <= effective_layer_idx < len(attentions_from_output):
                        layer_attention = attentions_from_output[effective_layer_idx]
                        if head_idx < layer_attention.shape[1]:
                            attention_matrix = layer_attention[0, head_idx].cpu().numpy()
                
                if model_type == "diffllama" and captured_attention_weights:
                    if attention_matrix is None:
                        hooked_attn_candidates = {}
                        for key_hook, tensor_hook in captured_attention_weights.items():
                            if torch.is_tensor(tensor_hook) and len(tensor_hook.shape) == 4:
                                if any(attr_part in key_hook for attr_part in ['_attention_weights', '_attn_weights', '_attention_probs', '_output_attn_']):
                                    hooked_attn_candidates[key_hook] = tensor_hook
                        
                        if hooked_attn_candidates:
                            chosen_key = sorted(hooked_attn_candidates.keys())[0]
                            chosen_tensor = hooked_attn_candidates[chosen_key]
                            if head_idx < chosen_tensor.shape[1]:
                                attention_matrix = chosen_tensor[0, head_idx].cpu().numpy()

        except Exception as e:
            print(f"Error extracting attention: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            for hook in hooks:
                hook.remove()
            model.config.output_attentions = original_output_attentions
        
        # Final metadata update
        metadata['layer_idx'] = layer_idx
        if model_type == "diffllama":
            metadata['captured_components'] = list(captured_attention_weights.keys())
            if metadata.get('lambda_std_dev') is None and hasattr(model.config, 'lambda_std_dev'):
                metadata['lambda_std_dev'] = model.config.lambda_std_dev
        else: 
            metadata.pop('lambda_params', None)
            metadata.pop('lambda_std_dev', None)
        
        return attention_matrix, tokens, metadata
    
    
    def run_test(self):
        fs = gcsfs.GCSFileSystem()
        template = self.OURS_TEMPLATE
        
        # Initialize results structure
        results = {
            "metadata": {
                "model_type": "diffllama",  # or "llama" for baseline
                "test_name": "attention_pattern_analysis",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "raw_results": [],
            "aggregated_results": {
                "by_depth": {},
                "by_context_length": {}
            }
        }
        
        full_contexts = self.read_context_files(FLAGS.n_rounds)
        full_tokens = [self.enc.encode(full_context) for full_context in full_contexts]
        
        for context_length in self.context_lengths:
            trim_contexts = [self.enc.decode(full_token[:context_length]) for full_token in full_tokens]
            
            for i in range(FLAGS.n_rounds):
                # Sample random cities and depths
                random_cities = random.sample(LLMNeedleHaystackTester.RANDOM_NEEDLE_CITIES, FLAGS.n_needles_total)
                document_depths = random.sample(self.document_depth_percents, FLAGS.n_needles_total)
                random_cities_retrieve = random.sample(random_cities, FLAGS.n_needles_retrieve)
                
                # Create needles info
                needles_info = {}
                for random_city, depth_percent in zip(random_cities, document_depths):
                    needles_info[random_city] = (
                        str(self.generate_random_number(self.rnd_number_digits)),
                        depth_percent
                    )
                
                # Create context
                context = self.create_contexts(needles_info, random_cities_retrieve, trim_contexts[i], context_length, i)
                prompt = template.format(context=context['context'], question=context['question'])
                
                # Get attention scores
                attention_matrix, tokens, metadata = get_attention_scores(
                    text=prompt,
                    layer_idx=-1,
                    head_idx=0
                )
                
                # Analyze attention patterns
                answer_tokens = self.enc.encode(" ".join(context['answer']))  # Handle multiple answers
                answer_positions = []
                for token in answer_tokens:
                    answer_positions.extend([i for i, t in enumerate(tokens) if t == token])
                
                # Calculate metrics
                total_attention = np.sum(attention_matrix)
                if answer_positions:
                    answer_attention = np.sum(attention_matrix[:, answer_positions])
                    non_answer_attention = total_attention - answer_attention
                    attention_to_answer = answer_attention / len(answer_positions)
                    attention_noise = non_answer_attention / (attention_matrix.size - len(answer_positions))
                else:
                    attention_to_answer = 0
                    attention_noise = total_attention / attention_matrix.size
                
                # Store raw result
                raw_result = {
                    "context_length": context_length,
                    "depth_percent": depth_percent,
                    "seed": i,
                    "attention_matrix_shape": attention_matrix.shape,
                    "attention_to_answer": float(attention_to_answer),  # Convert to native Python float
                    "attention_noise": float(attention_noise),
                    "tokens": tokens,
                    "answer_positions": answer_positions,
                    "prompt": prompt
                }
                results["raw_results"].append(raw_result)
                
                # Update aggregated results
                depth_key = f"{depth_percent}%"
                length_key = f"{context_length}tokens"
                
                if depth_key not in results["aggregated_results"]["by_depth"]:
                    results["aggregated_results"]["by_depth"][depth_key] = {
                        "attention_to_answer": [],
                        "attention_noise": []
                    }
                
                if length_key not in results["aggregated_results"]["by_context_length"]:
                    results["aggregated_results"]["by_context_length"][length_key] = {
                        "attention_to_answer": [],
                        "attention_noise": []
                    }
                
                results["aggregated_results"]["by_depth"][depth_key]["attention_to_answer"].append(attention_to_answer)
                results["aggregated_results"]["by_depth"][depth_key]["attention_noise"].append(attention_noise)
                results["aggregated_results"]["by_context_length"][length_key]["attention_to_answer"].append(attention_to_answer)
                results["aggregated_results"]["by_context_length"][length_key]["attention_noise"].append(attention_noise)
        
        # Calculate final aggregated statistics
        for depth_key in results["aggregated_results"]["by_depth"]:
            depth_data = results["aggregated_results"]["by_depth"][depth_key]
            depth_data["mean_attention_to_answer"] = float(np.mean(depth_data["attention_to_answer"]))
            depth_data["mean_attention_noise"] = float(np.mean(depth_data["attention_noise"]))
            depth_data["std_attention_to_answer"] = float(np.std(depth_data["attention_to_answer"]))
            depth_data["std_attention_noise"] = float(np.std(depth_data["attention_noise"]))
            # Remove raw lists to keep JSON smaller
            del depth_data["attention_to_answer"]
            del depth_data["attention_noise"]
        
        for length_key in results["aggregated_results"]["by_context_length"]:
            length_data = results["aggregated_results"]["by_context_length"][length_key]
            length_data["mean_attention_to_answer"] = float(np.mean(length_data["attention_to_answer"]))
            length_data["mean_attention_noise"] = float(np.mean(length_data["attention_noise"]))
            # Remove raw lists
            del length_data["attention_to_answer"]
            del length_data["attention_noise"]
        
        # Save final results
        output_filename = f"attention_results_{int(time.time())}.json"
        if FLAGS.output_file.startswith('gs://'):
            with fs.open(output_filename, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            with open(output_filename, 'w') as f:
                json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_filename}")
        return results
    
    

    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.needle.strip()}")
        print ("\n\n")

    def start_test(self):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        self.run_test()



class Sampler:
    def __init__(self, model_path, tokenizer_name, block_size=2048):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        # self.model = DiffLlamaForCausalLM.from_pretrained(model_path)
        self.sharded_rng = jax.random.PRNGKey(0)
        self.block_size = block_size
    
    @property
    def data_dim(self):
        return 1

    def _forward_generate(self, model, rng, batch):  
        output = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_new_tokens=self.block_size,
            do_sample=False,
        )
        return output, rng

    def __call__(self, prompts, max_input_length=512):
        # 限制输入最大长度不能超过模型的 block_size（例如 2048）
        max_input_length = min(max_input_length, self.block_size)

        inputs = self.tokenizer(
            prompts,
            padding='max_length',          # 补齐到 max_length
            truncation=True,               # 超长部分截断
            max_length=max_input_length,   # 最大输入长度（不能超过模型支持的最大长度）
            return_tensors='pt'            # 返回 PyTorch tensor
        )

        batch = {
            'input_ids': inputs.input_ids,
            'attention_mask': inputs.attention_mask,
        }

        output_ids, self.sharded_rng = self._forward_generate(self.model, self.sharded_rng, batch)
        output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return output_texts

def main(argv):
    #JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    set_random_seed(FLAGS.seed)

    ht = LLMNeedleHaystackTester(
        haystack_file=FLAGS.haystack_file,
        context_lengths_min=FLAGS.context_lengths_min,
        context_lengths_max=FLAGS.context_lengths_max,
        context_lengths_num_intervals=FLAGS.n_context_length_intervals,
        document_depth_percent_intervals=FLAGS.n_document_depth_intervals,
    )
    ht.start_test()

if __name__ == "__main__":
    run(main)