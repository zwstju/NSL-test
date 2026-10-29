import numpy as np
import torch
import time
import math
torch.set_printoptions(8)

def gelu(x):
    """
        Task: Use the torch API to implement the approximate calculation formula of the `GELU`
        activation function. The formula is as follows (you need to paste it into the latex
        online conversion website)
        Website: https://www.latexlive.com/
        Formula: \frac{1}{2} x\left[1+\tanh \left(\sqrt{\frac{2}{\pi}}\left(x+0.044715 x^{3}\right)\right)\right]
        
        Input: Tensor
        Output: Tensor
    """
    return 0.5 * x * (
        1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)
        )
    )

def softmax(x):
    """
        Task: Use torch API to implement `softmax` function, search the specific formula by yourself
        Input: Tensor
        Output: Tensor
    """
    x = x - x.max(dim=-1, keepdim = True).values
    return torch.exp(x)/(
        torch.exp(x).sum(dim=-1, keepdim = True)
    )


def layer_norm(x, g_b, eps:float = 1e-5):
    """
        Task: Use torch API to implement `layernorm` function, search `layernorm` by yourself
        Input: 
            x: Tensor
            g_b: dictionary that load from gpt2 weight. g-gamma and b-bias are the keys
        Output: Tensor
    """
    g, b = torch.Tensor(g_b['g']), torch.Tensor(g_b['b'])
    d = x.shape[-1]
    mean = x.sum(dim = -1, keepdim = True)/d
    var = ((x - mean)**2).sum(dim = -1, keepdim = True)/d

    return g * (x - mean) / torch.sqrt(var + eps) + b
    

def linear(x, w_b):  # [m, in], [in, out], [out] -> [m, out]
    """
        Task: implement linear layer 
        Input: 
            x: Tensor
            w_b: dictionary that load from gpt2 weight. w-weight and b-bias are the keys
        Output: Tensor
    """
    w, b = w_b['w'], w_b['b']

    return x @ w + b
    
    

def ffn(x, mlp):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: use `gelu` `linear` to implement ffn
        Notes: x --linear--> --gelu--> --linear--> output
        Input: 
            x: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    w_b1, w_b2 = mlp['c_fc'], mlp['c_proj']

    return linear(gelu(linear(x, w_b1)), w_b2)


def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    """
        Task: use torch API to implement attention computation according to formula(1) of the following paper
              where d_k account for the last dimension of `k`
        Paper: https://arxiv.org/abs/1706.03762
        Input: 
            q: Tensor
            k: Tensor
            v: Tensor
            mask: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    d = k.shape[-1]
    return softmax(
            q @ k.transpose(-2,-1)/(d ** 0.5) + mask
        ) @ v
    

def mha(x, attn, n_head, layer_past):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: Complete the code of the multi-head attention
        
        Input: 
            x: Tensor
            attn: dictionary that load from gpt2 weight. c_attn and c_proj are the params of two linear layer
            n_head: number of head
        Output: Tensorying multi-head attention and linear transformation, shape [n_seq, n_embd].
    """

    c_attn, c_proj = attn['c_attn'], attn['c_proj']
    # qkv projection
    x = linear(x, c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]
    
    # Split into qkv
    """
        Task: Split the q,k,v matrix from the tensor x
        Notes: [n_seq, 3*n_embd] -> 3 * [n_seq, n_embd]
    """
    qkv = x.chunk(3, dim = -1) # need to modify

    # Split into heads
    qkv_heads = [qkv_part.chunk(n_head, dim=-1) for qkv_part in qkv]  # 3 * [n_seq, n_embd] -> 3 * n_head * [n_seq, n_embd/n_head]
    qkv_heads = list(zip(*qkv_heads))  # [3, n_head, n_seq, n_embd/n_head]

    # Causal mask to hide future inputs from being attended to
    """
        Task: Construct mask matrix
        Notes: 
            | 0  -inf -inf ... -inf |
            | 0    0  -inf ... -inf |
            | 0    0    0  ... -inf |
            |...  ...  ... ...  ... | 
            | 0    0    0  ...   0  |
        Mask is a tensor whose dimension is [n_seq, n_seq]
    """

    new_key = []
    new_value = []
    out_heads = []

    # causal_mask = torch.triu(
    #         torch.full((x.shape[0], x.shape[0]), float('-inf'), device=x.device, dtype=x.dtype), diagonal=1
    #         )# need to modify
    n_new = x.shape[0]
    
    past_len = (
        0 if layer_past is None
        else layer_past[0].shape[-2]
    )

    causal_mask = torch.triu(
        torch.full(
            (n_new, past_len + n_new),
            float('-inf'),
            device=x.device,
            dtype=x.dtype
        ),
        diagonal=past_len + 1
    )

    for i, (q, k, v) in enumerate(qkv_heads):
        
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key[i], k), dim = -2)
            v = torch.cat((past_value[i], v), dim = -2)

        new_key.append(k)
        new_value.append(v)

        # Perform attention over each head
        out = attention(q, k, v, causal_mask) # n_head * [n_seq, n_embd/n_head]
        out_heads.append(out)

    key = torch.stack(new_key, dim = 0)
    value = torch.stack(new_value, dim = 0)
    layer_present = (key, value)

    # Merge heads
    """
        Task: merge multi-heads results
        Notes: n_head * [n_seq, n_embd/n_head] --> [n_seq, n_embd]
    """
    x = torch.cat(out_heads, dim=-1) # need to modify
    
    # Out projection
    x = linear(x, c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]
    
    return x, layer_present


def transformer_block(x, block, n_head, layer_past):  # [n_seq, n_embd] -> [n_seq, n_embd]
    mlp, attn, ln_1, ln_2 = block['mlp'], block['attn'], block['ln_1'], block['ln_2']
    
    # multi-head causal self attention
    y, layer_present = mha(layer_norm(x, ln_1), attn, n_head, layer_past)  # [n_seq, n_embd] -> [n_seq, n_embd]
    x = x + y
    # position-wise feed forward network
    x = x + ffn(layer_norm(x, ln_2), mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x, layer_present


def gpt2(inputs, params, n_head, past_key_values = None):  # [n_seq] -> [n_seq, n_vocab]
    wte, wpe, blocks, ln_f = params['wte'], params['wpe'], params['blocks'], params['ln_f']
    # token + positional embeddings
    # x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]
    
    past_len = (
        0 if past_key_values is None
        else past_key_values[0][0].shape[-2]
    )
    positions = range(past_len, past_len + len(inputs))
    x = wte[inputs] + wpe[positions]

    x = torch.Tensor(x)
    # forward pass through n_layer transformer blocks

    # for block in blocks:
    #     x = transformer_block(x, block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]
    
    new_cache = []

    for i, block in enumerate(blocks):
        layer_past = (
            None if past_key_values is None
            else past_key_values[i]
        )

        x, layer_present = transformer_block(
            x, block, n_head, layer_past
        )

        new_cache.append(layer_present)
    # projection to vocab
    x = layer_norm(x, ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T, new_cache  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(inputs, params, n_head, n_tokens_to_generate, past_key_values = None, show_progress = True):
    from tqdm import tqdm

    # for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
    #     logits, past_key_values = gpt2(inputs, params, n_head=n_head, past_key_values = past_key_values)  # model forward pass
    #     next_id = np.argmax(logits[-1])  # greedy sampling
    #     inputs.append(int(next_id))  # append prediction to input

    # return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids

    current_inputs = list(inputs)
    generated_ids = []

    for _ in tqdm(range(n_tokens_to_generate), "generating", disable = not show_progress):
        logits, past_key_values = gpt2(
            current_inputs,
            params,
            n_head=n_head,
            past_key_values=past_key_values
        )

        next_id = torch.argmax(logits[-1]).item()

        generated_ids.append(next_id)
        current_inputs = [next_id]

    return generated_ids, past_key_values

def rollback_cache(cache, keep_len):
    if cache is None:
        return None
    new_cache = []

    for key, value in cache:
        key = key[..., :keep_len, :]
        value = value[..., :keep_len, :]

        new_cache.append((key, value))

    return new_cache

def greedy_speculative_generate(inputs, draft_params, target_params, hparams_draft, hparams_target, n_tokens_to_generate, K):
    
    """
        Task: Load 124M and 1558M models at the same time, use greedy sampling, and complete speculative decoding
    
        Inputs:
            inputs (list): The initial list of token IDs from the prompt.
            draft_params, target_params: Model weights for the draft and target models.
            hparams_draft, hparams_target: Hyperparameters for both models.
            n_tokens_to_generate (int): The number of new tokens to generate.
            K (int): The number of tokens the draft model speculates at each step (e.g., 4).

        Returns:
            list: A list of newly generated token IDs.
            
    """
    if K < 1:
        raise ValueError("K must be >= 1")

    if not inputs:
        raise ValueError("Prompt cannot be empty")
    
    generated_ids = []
    current_inputs = list(inputs)
    
    prefix = current_inputs[:-1]
    last_token = current_inputs[-1]

    draft_cache = None
    target_cache = None

    if(len(prefix) > 0):
        _, draft_cache = gpt2(
            prefix,
            draft_params,
            hparams_draft["n_head"]
        )

        _, target_cache = gpt2(
            prefix,
            target_params,
            hparams_target["n_head"]
        )
    
    while len(generated_ids) < n_tokens_to_generate:

        
        prefix_len = len(inputs) + len(generated_ids)
        remaining = n_tokens_to_generate - len(generated_ids)
        k_step = min(K, remaining)

        out_draft, draft_cache= generate([last_token], draft_params, hparams_draft["n_head"], k_step, draft_cache, show_progress = False)
        target_inputs = [last_token] + out_draft
        
        out_target, target_cache = gpt2(
            target_inputs,
            target_params,
            hparams_target["n_head"],
            past_key_values = target_cache
        )

        target_predictions = torch.argmax(
            out_target, dim=-1
        ).tolist()

        j = -1
        for i in range(len(out_draft)):
            if out_draft[i] != target_predictions[i]:
                j = i
                break

        

        if j == -1:
            
            generated_ids.extend(out_draft)

            if len(generated_ids) >= n_tokens_to_generate:
                break

            corrected = target_predictions[-1]
            generated_ids.append(corrected)

            _, draft_cache = gpt2(
                [out_draft[-1]],
                draft_params,
                hparams_draft["n_head"],
                past_key_values=draft_cache
            )

        else:

            accepted = out_draft[:j]
            corrected = target_predictions[j]
            
            generated_ids.extend(accepted)
            generated_ids.append(corrected)
            keep_len = prefix_len + j
            
            draft_cache = rollback_cache(draft_cache, keep_len)
            target_cache = rollback_cache(target_cache, keep_len)

        last_token = corrected
        

    return generated_ids


def main(prompt: str, n_tokens_to_generate: int = 5, K = 4, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams_draft, draft_params = load_encoder_hparams_and_params("124M", models_dir)
    _, hparams_target, target_params = load_encoder_hparams_and_params("1558M", models_dir)
    
    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    # assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]
    max_ctx = min(
        hparams_draft["n_ctx"],
        hparams_target["n_ctx"]
    )

    assert len(input_ids) + n_tokens_to_generate < max_ctx

    # generate output ids
    start = time.time()
    # output_ids, _ = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate, None)
    with torch.inference_mode():
        output_ids = greedy_speculative_generate(
            inputs=input_ids,
            draft_params=draft_params,
            target_params=target_params,
            hparams_draft=hparams_draft,
            hparams_target=hparams_target,
            n_tokens_to_generate=n_tokens_to_generate,
            K=K
        )


    end = time.time()
    print(f"Time taken to generate {n_tokens_to_generate} tokens: {end - start:.2f}s")

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)
    return output_text


if __name__ == "__main__":
    import fire
    fire.Fire(main)