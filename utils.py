import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import json
import os
import re
import string
import time
import numpy as np
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Literal, Optional
from math import ceil
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from fuzzywuzzy import fuzz
from tqdm import tqdm
from nltk import sent_tokenize

import torch


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")


def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-3}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory


# Model and Tokenizer Configuration
AUTOAIS_MODEL="google/t5_xxl_true_nli_mixture"
global autoais_model, autoais_tokenizer
autoais_model, autoais_tokenizer = None, None

# Thresholds and Flags
REJECTION_FUZZ_THRESHOLD=85
REJECTION_FLAG="I apologize, but I couldn't find an answer"

def _run_nli_autoais(passage, claim):
    """
    Run inference for assessing AIS between a premise and hypothesis.
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
    """
    global autoais_model, autoais_tokenizer
    input_text = "premise: {} hypothesis: {}".format(passage, claim)
    input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(autoais_model.device)
    with torch.inference_mode():
        outputs = autoais_model.generate(input_ids, max_new_tokens=10)
    result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    inference = 1 if result == "1" else 0
    return inference

def compute_autoais(data,
                    key,
                    qampari=False,
                    at_most_citations=None):
    """
    Compute AutoAIS score.

    Args:
        data: requires field `output` and `docs`
              - docs should be a list of items with fields `title` and `text` (or `phrase` and `sent` for QA-extracted docs)
        citation: check citations and use the corresponding references.
        decontext: decontextualize the output
    """

    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        print("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto")
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)

    logger.info(f"Running AutoAIS...")

    def _format_document(doc):
        """Format document for AutoAIS."""

        if "sent" in doc:
            # QA-extracted docs
            return "Title: %s\n%s" % (doc['title'], doc['sent'])
        else:
            return "Title: %s\n%s" % (doc['title'], doc['text'])

    for item in tqdm(data):
        if 'nli' not in item:
            item['nli'] = {}  # Initialize as an empty dictionary
        
        nli_output = {}

        # Get sentences by using NLTK
        if qampari:
            sents = [item['question'] + " " + x.strip() for x in item[key].rstrip().rstrip(".").rstrip(",").split(",")]
        else:
            sents = sent_tokenize(item[key])
        if len(sents) == 0:
            continue

        target_sents = [remove_citations(sent).strip() for sent in sents]
        
        total_citations = 0
        for sent_id, sent in enumerate(sents):
            target_sent = target_sents[sent_id] # Citation removed and (if opted for) decontextualized
            joint_entail = -1 # Undecided

            # Find references
            ref = [int(r[1:])-1 for r in re.findall(r"\[\d+", sent)] # In text citation id starts from 1
            logger.info(f"For `{sent}`, find citations {ref}")
            if len(ref) == 0:
                # No citations
                joint_entail = 0
            elif any([ref_id >= len(item['docs']) for ref_id in ref]):
                # Citations out of range
                joint_entail = 0
            else:
                if at_most_citations is not None:
                    ref = ref[:at_most_citations]
                total_citations += len(ref)
                joint_passage = '\n'.join([_format_document(item['docs'][psgs_id]) for psgs_id in ref])

            # If not directly rejected by citation format error, calculate the recall score
            if joint_entail == -1: 
                joint_entail = _run_nli_autoais(joint_passage, target_sent)
                nli_output[f'citation_recall_sent{sent_id}'] = joint_entail

            # calculate the precision score if applicable
            if len(ref) > 1:
                # Precision check: did the model cite any unnecessary documents?
                for psg_idx, psgs_id in enumerate(ref):
                    # condition A
                    passage = _format_document(item['docs'][psgs_id]) 
                    nli_result = _run_nli_autoais(passage, target_sent)
                    # print(f'citation_prec_sent{sent_id}_citation_{psg_idx}')
                    nli_output[f'citation_prec_sent{sent_id}_citation{psg_idx}'] = nli_result
        item['nli'][key] = nli_output

    return data


