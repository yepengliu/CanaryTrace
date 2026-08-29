"""Stage 1 -- Canary dataset synthesis.

For each canary: sample a reference document from the IP dataset, extract its
attributes with an OpenAI model, synthesize fictional entities / descriptions,
generate a watermarked article with a watermarked LLM, and generate a probing
question answerable only from that article.

Consolidates ``synthesize_data.py`` and its per-experiment variants. Canaries are
written by the green-list (unigram) watermarked generator in ``watermark_unigram``.

The IP corpus loads either from BEIR (``ip_loader: beir_split``) or a local
corpus/query/qrels triple (``ip_loader: local``, used for CQADupStack-Mathematica).

Requires an OpenAI API key in the ``OPENAI_API_KEY`` environment variable.
"""

import os
import json
import random
from typing import List

import torch
from tqdm import tqdm
from pydantic import BaseModel
from openai import OpenAI
from beir.datasets.data_loader import GenericDataLoader

from . import config
from .cli import parse_and_load
from .utils import load_model


class Attributes(BaseModel):
    topic: str
    subtopics: List[str]
    writing_styles: str
    length_range: str

    def to_dict(self):
        return self.model_dump()


class Entities(BaseModel):
    real_entity: List[str]
    fictional_entity: List[str]

    def to_dict(self):
        return self.model_dump()


class Descriptions(BaseModel):
    description_1: str
    description_2: str
    description_3: str

    def to_dict(self):
        return self.model_dump()


RESPONSE_FORMATS = {"attributes": Attributes, "entities": Entities, "descriptions": Descriptions}

# --- Optional "harmless canary" constraint (synthesize.harmless: true) ---------------
# Keep fictional entities (uniqueness / retrieval / watermark carrier) but forbid
# fabricated actionable claims, so a leaked canary carries no misinformation. Replaces
# the description "ensure factual accuracy" requirement and adds a post-filter.
HARMLESS_DESC_REQ = (
    "3. Non-actionable safety constraint (all must hold): present each entity ONLY as "
    "something that exists in the topic area (a named product, brand, program, or "
    "initiative); do NOT state or imply any health, medical, therapeutic, nutritional, "
    "safety, or efficacy claim (no treats/prevents/cures/reduces/improves/protects/"
    "is-safe/beneficial-for/recommended-for, no dosage or consumption guidance); do NOT "
    "invent studies, trials, findings, experiments, statistics, percentages, or expert "
    "endorsements; keep every statement generic and non-verifiable; a reader must not be "
    "able to derive any health, medical, or consumption decision from the text."
)
HARMLESS_ARTICLE_LINE = (
    "3. Do NOT add any health, medical, safety, nutritional, or efficacy claim, study, "
    "statistic, or expert endorsement beyond the reference descriptions; keep the named "
    "entities described in neutral, non-actionable terms."
)
_HARMFUL_CHECK = (
    "Does the following text make or imply any health, medical, therapeutic, nutritional, "
    "safety, or efficacy claim about any product/entity, or cite any study/trial/statistic "
    "as evidence? Answer strictly as JSON: {{\"harmful\": 1 or 0}}.\n\nText:\n{text}"
)


def harmful_claim_check(client, model, text):
    """Post-filter: True if the text still contains a fabricated actionable/medical claim."""
    import re
    try:
        r = client.chat.completions.create(
            model=model, temperature=0, max_tokens=20,
            messages=[{"role": "user", "content": _HARMFUL_CHECK.format(text=text)}])
        m = re.search(r'"harmful"\s*:\s*([01])', r.choices[0].message.content)
        return bool(m and m.group(1) == "1")
    except Exception:
        return False


def make_watermark(cfg, device):
    """Instantiate the green-list watermarked generator used to write canaries."""
    wm = cfg["watermark"]
    model, tokenizer = load_model(wm["watermark_model"], cfg["models"].get("dtype", "float16"))
    from .watermark_unigram import Watermark
    watermark = Watermark(
        device=device, watermark_tokenizer=tokenizer, watermark_model=model,
        gamma=wm.get("gamma", 0.5), bias=wm["bias"], top_k=50, top_p=0.9, repetition_penalty=1.0,
        no_repeat_ngram_size=0, max_new_tokens=wm["max_new_tokens"], key=wm["key"],
    )
    return watermark, watermark.generate_watermarked


def load_ip_corpus(cfg):
    """Load the IP corpus we sample reference documents from (queries/qrels unused)."""
    data = cfg["data"]
    if data.get("ip_loader", "beir_split") in ("corpus_file", "local"):
        return GenericDataLoader(corpus_file=str(config.resolve_path(data["ip_corpus_path"]))).load_corpus()
    from .rag import beir_data_path
    ip_path = beir_data_path(data["ip_dataset"])
    corpus, _, _ = GenericDataLoader(data_folder=ip_path).load(split="test")
    return corpus


def main(cfg):
    api_key = config.get_openai_api_key(cfg)
    client = OpenAI(api_key=api_key)
    syn = cfg["synthesize"]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()

    watermark, wm_generate = make_watermark(cfg, device)
    corpus = load_ip_corpus(cfg)

    def generate(message, task, max_attempts=3):
        response_format = RESPONSE_FORMATS[task]
        attempt = 0
        while attempt <= max_attempts:
            try:
                completion = client.beta.chat.completions.parse(
                    model=cfg["models"]["openai_model"], messages=message,
                    temperature=syn["temperature"], top_p=syn["top_p"], response_format=response_format,
                )
                response = completion.choices[0].message
                if response.parsed:
                    return response.parsed
                attempt += 1
            except Exception:
                attempt += 1
        return None

    def question_generation(message):
        completion = client.chat.completions.create(model=cfg["models"]["openai_model"], messages=message)
        return completion.choices[0].message.content

    def synthetic_pipeline():
        message = [{
            "role": "system",
            "content": (
                "You are a content creator. Here is a comprehensive set of instructions for content creators who "
                "are tasked with creating synthetic documents. You will be given a reference text sampled from a "
                "dataset. These instructions are designed to ensure that the synthesized documents maintain high "
                "quality, align with project requirements, and remain consistent with the reference text in some "
                "attributes, enabling seamless integration into the original dataset."
            ),
        }]

        sampled_corpus = dict(random.sample(list(corpus.items()), syn["sample_ref_num"]))
        sampled_text_list = list(t["text"] for t in sampled_corpus.values())
        sampled_text = "\n".join([f"text_{i+1}: {text}" for i, text in enumerate(sampled_text_list)])

        attr_prompt = f"""
            ### Task Description:
            A reference text is given. You will carefully analyze the reference text and identify the following four key attributes.
            1. Topic: Read the reference text and provide a high-level theme or general category of the reference text.
            2. Subtopics: Based on the general topic, identify {syn['subtopic_num']} distinct general sub-category.
            3. Writing Style: Analyze the overall writing style of the reference text.
            4. Length Range: Provide an estimate of the length range of the reference text in terms of word count.

            ### Output Format Requirements:
            Output the results with the JSON format (with four keys: topic, subtopics, writing_styles and length_range) and nothing else , such as {{\"topic\": \" \", \"subtopics\": [\"subtopic_1\", \"subtopic_2\", ...], \"writing_styles\": \" \", \"length_range\": \"m - n words\"}}.

            ### Reference Text:
            {sampled_text}
            """
        message.append({"role": "user", "content": attr_prompt})
        output_attributes = generate(message, "attributes")
        if output_attributes is None:
            return None
        attributes = output_attributes.to_dict()
        message.append({"role": "assistant", "content": json.dumps(attributes)})

        sampled_subtopics = random.sample(attributes["subtopics"], 1)
        ent_prompt = f"""
            ### Task Description:
            1. Identify and list the {syn['entity_num']} entities mentioned within the reference text.
            2. Synthesize {syn['entity_num']} fictional entities that align with the {sampled_subtopics[0]} topic.

            ### Synthesized Entities Requirements:
            1. The synthesized entities should be creative and distinct.
            2. Ensure the synthesized entities are fictional and do not overlap with real-world entities.

            ### Output Format Requirements:
            Directly output the results with JSON format (with two keys: real_entity and fictional_entity) and nothing else, such as {{\"real_entity\": [\"real_entity_1\", \"real_entity_2,...\"], \"fictional_entity\": [\"fictional_entity_1\", \"fictional_entity_2,...\"]}}.
            """
        message.append({"role": "user", "content": ent_prompt})
        output_entities = generate(message, "entities")
        if output_entities is None:
            return None
        entities = output_entities.to_dict()
        message.append({"role": "assistant", "content": json.dumps(entities)})

        sampled_entities = random.sample(entities["fictional_entity"], syn["sample_ent_num"])
        harmless = syn.get("harmless", False)
        desc_req3 = HARMLESS_DESC_REQ if harmless else \
            "3. Ensure factual accuracy where applicable, even in synthetic scenarios."
        article_extra = ("\n            " + HARMLESS_ARTICLE_LINE) if harmless else ""
        desc_prompt = f"""
            ### Task Description:
            1. Write {syn['sample_ent_num']} fictional descriptions in an {attributes['writing_styles']} style about the following entities: {sampled_entities[0]}, {sampled_entities[1]}.
            2. Create {syn['common_topic_num']} fictional interactions and discuss how those specified entities fictionally interact within the context of the {sampled_subtopics[0]}.

            ### Synthesized Description Requirements:
            1. Create unique and imaginative content that has not been derived from existing material to avoid any issues with plagiarism.
            2. Use creativity to simulate realistic scenarios that fit within the project's thematic boundaries.
            {desc_req3}
            4. Incorporate Diverse and Inclusive Content.
            5. Do not mention "fictional" or any other indication that the entity or interaction is not real.

            ### Output Format Requirements:
            Directly output the results in one JSON format and nothing else, in the form of {{\"description_1\": \"\", \"description_2\": \" \", \"description_3\": \" \"}}
            """
        wm_system_prompt = (
            "You are a content creator. Here is a comprehensive set of instructions for content creators who are "
            "tasked with creating synthetic documents. You will be given some descriptions. These instructions are "
            "designed to ensure that the synthesized documents maintain high quality, align with project "
            "requirements, and remain consistent with the reference descriptions."
        )
        # Generate descriptions -> watermarked article; if harmless, regenerate when the
        # post-filter still detects a fabricated actionable claim.
        descriptions, wm_synthetic_text = None, None
        for _ in range(5 if harmless else 1):
            output_descriptions = generate(message + [{"role": "user", "content": desc_prompt}], "descriptions")
            if output_descriptions is None:
                return None
            descriptions = output_descriptions.to_dict()
            wm_synthetic_prompt = f"""
            ### Task Description:
            A reference descriptions is given. You will carefully understand the reference descriptions and synthesize a text that satisfy the following instructions.
            1. Generate a finctional text in the style of {attributes['writing_styles']} in the context of {sampled_subtopics[0]}, with a length range of {attributes['length_range']} in terms of word count.
            2. Include the information in the given reference descriptions.{article_extra}

            ### Output Format Requirements:
            Directly output the synthesized article in one paragraph and nothing else.

            ### Reference Descriptions:
            {json.dumps(descriptions)}
        """
            wm_message = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{wm_system_prompt}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>{wm_synthetic_prompt}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>"
            )
            wm_synthetic_text = wm_generate(wm_message)
            if not harmless or not harmful_claim_check(client, cfg["models"]["openai_model"], wm_synthetic_text):
                break

        # For harmless canaries, forbid questions that presuppose an efficacy/medical
        # claim (the dominant residual-leak channel: a leading question makes the model
        # elaborate a fabricated benefit even from a clean document). Harmless-only; the
        # non-harmless prompt is unchanged.
        question_extra = (
            "\n            Non-actionable constraint: the question MUST NOT presuppose, assert, or ask about any "
            "health, medical, therapeutic, nutritional, safety, or efficacy effect, benefit, result, mechanism, "
            "study, trial, or finding of the named entities. Avoid words such as effects, benefits, improves, "
            "enhances, prevents, treats, contributes to, demonstrated, proven, trials, or results. Ask only "
            "descriptively about what the named entities are, what they represent, or how the article describes "
            "them. The question must still name the specific entities so it remains answerable only from the article."
            if harmless else ""
        )
        question_prompt = f"""
            ### Task Description:
            Given an article, generate a question that can only be answered by reading the document. The answer should be a longer detailed response, so avoid factual and simple yes/no questions and steer more towards questions that ask for opinions or explanations of events or topics described in the documents. Do not provide the answer, provide just the question.{question_extra}

            ### Article:
            {wm_synthetic_text}
            """
        question = question_generation([{"role": "user", "content": question_prompt}])
        return sampled_text_list, attributes, entities, descriptions, wm_synthetic_text, question

    results_list = []
    save_path = config.resolve_path(syn["output"])
    os.makedirs(save_path.parent, exist_ok=True)
    for _ in tqdm(range(syn["synthetic_num"])):
        out = synthetic_pipeline()
        if out is None:
            print("Failed to generate one synthetic data")
            continue
        sampled_text_list, attributes, entities, descriptions, wm_synthetic_text, question = out
        results_list.append({
            "sampled_corpus": sampled_text_list,
            "attributes": attributes,
            "entities": entities,
            "descriptions": descriptions,
            "synthetic_text": wm_synthetic_text,
            "question": question,
        })
        with open(save_path, "w") as f:
            json.dump(results_list, f)
    print(f"[synthesize] {len(results_list)} canaries -> {save_path}")


if __name__ == "__main__":
    main(parse_and_load("CanaryTrace stage 1: canary dataset synthesis"))
