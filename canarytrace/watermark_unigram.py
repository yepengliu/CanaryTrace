import torch
from tokenizers import Tokenizer
from torch.nn import functional as F
# from nltk.tokenize import word_tokenize
# from collections import Counter
import numpy as np
import hashlib
from tqdm import tqdm

class Watermark():
    def __init__(self,
                 device: torch.device = None,
                 watermark_tokenizer: Tokenizer = None,
                 watermark_model = None,
                 gamma: float = 0.5,
                 bias: int = 2,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 repetition_penalty: float = 1.0,
                 no_repeat_ngram_size: int = 0,
                 max_new_tokens: int = 230,
                 key: int = 0,
                 ):
        self.device = device
        self.watermark_tokenizer = watermark_tokenizer
        self.watermark_model = watermark_model
        self.gamma = gamma
        self.bias = bias
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.max_new_tokens = max_new_tokens
        self.key = key
    
    def _calc_banned_ngram_tokens(self, prev_input_ids: torch.Tensor, num_hypos: int, no_repeat_ngram_size: int, cur_len: int) -> None:
        """Copied from fairseq for no_repeat_ngram in beam_search"""
        if cur_len + 1 < no_repeat_ngram_size:
            # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
            return [[] for _ in range(num_hypos)]
        generated_ngrams = [{} for _ in range(num_hypos)]
        for idx in range(num_hypos):
            gen_tokens = prev_input_ids[idx].tolist()
            generated_ngram = generated_ngrams[idx]
            for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
                prev_ngram_tuple = tuple(ngram[:-1])
                generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

        def _get_generated_ngrams(hypo_idx):
            # Before decoding the next token, prevent decoding of ngrams that have already appeared
            start_idx = cur_len + 1 - no_repeat_ngram_size
            ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
            return generated_ngrams[hypo_idx].get(ngram_idx, [])

        banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
        return banned_tokens

    def _postprocess_next_token_scores(self, lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty, no_repeat_ngram_size):
        # _enforce_repetition_penalty
        if repetition_penalty != 1.0:
            for i in range(batch_size * num_beams):
                for previous_token in set(prev_output_tokens[i].tolist()):
                    # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                    if lprobs[i, previous_token] < 0:
                        lprobs[i, previous_token] *= repetition_penalty
                    else:
                        lprobs[i, previous_token] /= repetition_penalty
        
        # lower eos token prob to zero if min_length is not reached
        # if prev_output_tokens.size(1) < self.min_new_tokens:
        #     lprobs[:, self.watermark_tokenizer.eos_token_id] = -float("Inf")
        
        if no_repeat_ngram_size > 0:
            # calculate a list of banned tokens to prevent repetitively generating the same ngrams
            num_batch_hypotheses = batch_size * num_beams
            # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
            banned_batch_tokens = self._calc_banned_ngram_tokens(
                prev_output_tokens, num_batch_hypotheses, no_repeat_ngram_size, prev_output_tokens.size(1)
            )
            for i, banned_tokens in enumerate(banned_batch_tokens):
                lprobs[i, banned_tokens] = -float("inf")

    def _top_k_top_p_filtering(
        self,
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ) -> torch.Tensor:
        """ 
        Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        """
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits
    
    def _stopping_criteria(self, ids):
        if ids[0][-1] == self.watermark_tokenizer.eos_token_id:
            return True
        return False

    # Un-watermarked text generation
    def generate_unwatermarked(self, prompt):
        input_ids = self.watermark_tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        output_ids = torch.tensor([[]], dtype=torch.int64, device=self.device)

        attn = torch.ones_like(input_ids)
        past = None
        for i in range(self.max_new_tokens):
            with torch.no_grad():
                if past:
                    output = self.watermark_model(input_ids[:,-1:], attention_mask=attn, past_key_values=past)
                else:
                    output = self.watermark_model(input_ids)
            
            logits = output.logits[:,-1, :]
            self._postprocess_next_token_scores(logits, 1, 1, output_ids, repetition_penalty=self.repetition_penalty, no_repeat_ngram_size=self.no_repeat_ngram_size)   # repetition penalty: 1.1
            logits = self._top_k_top_p_filtering(logits, top_k=self.top_k, top_p=self.top_p)   # top-k, top-p filtering
            probs = torch.nn.functional.softmax(logits, dim=-1)   # softmax
            next_id = torch.multinomial(probs, num_samples=1)   # sampling

            input_ids = torch.cat((input_ids, next_id), dim=-1)   # update input_ids
            output_ids = torch.cat((output_ids, next_id), dim=-1)   # update output_ids

            past = output.past_key_values
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

            # stopping criteria
            stop = self._stopping_criteria(output_ids)
            if stop:
                output_text = self.watermark_tokenizer.decode(output_ids[0].tolist())
                return output_text
        
        output_text = self.watermark_tokenizer.decode(output_ids[0])
        return output_text

    def _vocab_segmentation(self, device, gamma, hash_key):
        vocab = list(self.watermark_tokenizer.get_vocab().values())
        vocab_size = len(vocab)
        vocab_permutation = torch.randperm(vocab_size, generator=torch.Generator(device=device).manual_seed(hash_key), device=device)
        green_list_ids = vocab_permutation[:int(vocab_size * gamma)]

        green_mask = torch.zeros(1, vocab_size, dtype=torch.bool, device=device)
        green_mask[:, green_list_ids] = True
        return green_mask

    def _watermarking(self, logits, green_mask, bias):
        logits[green_mask] = logits[green_mask] + bias
        return logits

    # watermark text generation
    def generate_watermarked(self, prompt):
        input_ids = self.watermark_tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        green_mask = self._vocab_segmentation(self.device, self.gamma, self.key)

        output_ids = torch.tensor([[]], dtype=torch.int64, device=self.device)
        attn = torch.ones_like(input_ids)
        past = None
        for t in range(self.max_new_tokens):
            with torch.no_grad():
                if past:
                    output = self.watermark_model(input_ids[:,-1:], attention_mask=attn, past_key_values=past)
                else:
                    output = self.watermark_model(input_ids)
            
            logits = output.logits[:,-1, :]
            self._postprocess_next_token_scores(logits, 1, 1, output_ids, repetition_penalty=self.repetition_penalty, no_repeat_ngram_size=self.no_repeat_ngram_size)
            logits = self._watermarking(logits, green_mask, bias=self.bias)   # watermarking
            logits = self._top_k_top_p_filtering(logits, top_k=self.top_k, top_p=self.top_p)   # top-k, top-p filtering
            probs = torch.nn.functional.softmax(logits, dim=-1)   # softmax
            next_id = torch.multinomial(probs, num_samples=1)   # sampling

            input_ids = torch.cat((input_ids, next_id), dim=-1)   # update input_ids
            output_ids = torch.cat((output_ids, next_id), dim=-1)   # update output_ids

            past = output.past_key_values
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

            # stopping criteria
            stop = self._stopping_criteria(output_ids)
            if stop:
                output_text = self.watermark_tokenizer.decode(output_ids[0], skip_special_tokens=True)
                return output_text
        
        output_text = self.watermark_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text
