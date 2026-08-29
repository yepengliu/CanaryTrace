"""Green-list (unigram) watermark detection.

The data owner scores a piece of text against a fixed green list of token ids and
reports the z-statistic of the green-token count under the null hypothesis that
the text was not produced by the watermarked model:

    z = (|green| - gamma * T) / sqrt(T * gamma * (1 - gamma))

``T`` counts *unique token bigrams* rather than raw tokens: a token that recurs in
the same bigram context is credited once, which keeps repetitive RA-LLM responses
from inflating the score. Text is unicode-sanitised first so that invisible
whitespace / formatting characters cannot be used to break tokenisation.

"""

import re
import unicodedata
import collections
from math import sqrt

from nltk.util import ngrams


class UnicodeSanitizer:
    """Strip invisible / ambiguous unicode so tokenisation cannot be gamed.

    Removes non-breaking and zero-width spaces, bidi and variation selectors, and
    the other formatting characters listed below, collapses runs of whitespace,
    and drops remaining control characters.
    """

    _PATTERN = re.compile(
        "[\u00A0\u1680\u180E\u2000-\u200B\u200C\u200D\u200E\u200F\u2060\u2063"
        "\u202A\u202B\u202C\u202D\u202E\u202F\u205F\u3000\u3164"
        "\uFE00-\uFE0F\uFEFF\uFFA0\uFFF9\uFFFA\uFFFB]"
    )

    def __call__(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = self._PATTERN.sub(" ", text)
        text = re.sub(" +", " ", text)
        return "".join(c for c in text if unicodedata.category(c) != "Cc")


_SANITIZE = UnicodeSanitizer()

# The green list is a ~64k-element tensor and is fixed for a whole run, while the
# detector scores thousands of response groups. Convert it to a set once.
_GREEN_CACHE = {}


def _green_set(green_list_ids):
    cached = _GREEN_CACHE.get("current")
    if cached is not None and cached[0] is green_list_ids:
        return cached[1]
    ids = green_list_ids.tolist() if hasattr(green_list_ids, "tolist") else green_list_ids
    green = set(ids)
    _GREEN_CACHE["current"] = (green_list_ids, green)   # strong ref keeps the identity valid
    return green


def z_score(green_count, num_scored, gamma):
    """Green-token z-statistic. Returns -inf when there is nothing to score."""
    if num_scored < 1:
        return float("-inf")
    return (green_count - gamma * num_scored) / sqrt(num_scored * gamma * (1 - gamma))


def score_tokens(token_ids, green_list_ids, gamma=0.5):
    """Score a token id sequence, counting each unique bigram once.

    Returns ``(z, num_scored, green_count)``.
    """
    green = _green_set(green_list_ids)
    bigrams = collections.Counter(ngrams(list(token_ids), 2))
    num_scored = len(bigrams)
    green_count = sum(1 for bigram in bigrams if bigram[1] in green)
    return z_score(green_count, num_scored, gamma), num_scored, green_count


def detect(text, tokenizer, green_list_ids, gamma=0.5):
    """Detect the green-list watermark in ``text``; returns the z-score as a float."""
    text = _SANITIZE(text)
    if not text:
        return float("-inf")

    token_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if len(token_ids) and token_ids[0] == tokenizer.bos_token_id:
        token_ids = token_ids[1:]
    if len(token_ids) < 2:
        return float("-inf")

    z, _, _ = score_tokens(token_ids.tolist(), green_list_ids, gamma)
    return z
