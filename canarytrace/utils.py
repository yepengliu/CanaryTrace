import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_name, dtype="float16"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map='auto')
    model.eval()
    return model, tokenizer

def vocab_segmentation(tokenizer, device, gamma, hash_key):
    """Deterministically split the vocabulary into a green list given a hash key.

    Returns the green-list token ids (the first ``gamma`` fraction of a keyed
    random permutation of the vocabulary), as used by the green-list watermark
    detector.
    """
    vocab = list(tokenizer.get_vocab().values())
    vocab_size = len(vocab)
    vocab_permutation = torch.randperm(
        vocab_size,
        generator=torch.Generator(device=device).manual_seed(hash_key),
        device=device,
    )
    green_list_size = int(vocab_size * gamma)
    green_list_ids = vocab_permutation[:green_list_size]
    return green_list_ids
