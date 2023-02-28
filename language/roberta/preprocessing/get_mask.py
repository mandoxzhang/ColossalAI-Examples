import torch
import os
from enum import IntEnum
from random import choice
import random
import collections
import time
import logging
import jieba

jieba.setLogLevel(logging.CRITICAL)
import re
import numpy as np

# import mask

PAD = 0
MaskedLMInstance = collections.namedtuple("MaskedLMInstance", ["index", "label"])


def map_to_numpy(data):
    return np.asarray(data)


class PreTrainingDataset:
    def __init__(
        self,
        tokenizer,
        max_seq_length,
        backend="python",
        max_predictions_per_seq: int = 80,
        do_whole_word_mask: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.masked_lm_prob = 0.15
        self.backend = backend
        self.do_whole_word_mask = do_whole_word_mask
        self.max_predictions_per_seq = max_predictions_per_seq
        self.vocab_words = list(tokenizer.vocab.keys())
        self.rec = re.compile("[\u4E00-\u9FA5]")
        self.whole_rec = re.compile("##[\u4E00-\u9FA5]")

        self.mlm_p = 0.15
        self.mlm_mask_p = 0.8
        self.mlm_tamper_p = 0.05
        self.mlm_maintain_p = 0.1

    def tokenize(self, doc):
        temp = []
        for d in doc:
            temp.append(self.tokenizer.tokenize(d))
        return temp

    def create_training_instance(self, instance):
        is_next = 1
        raw_text_list = self.get_new_segment(instance)
        tokens_a = raw_text_list
        assert len(tokens_a) == len(instance)
        # tokens_a, tokens_b, is_next = instance.get_values()
        # print(f'is_next label:{is_next}')
        # Create mapper
        tokens = []
        original_tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        original_tokens.append("[CLS]")
        segment_ids.append(0)
        for index, token in enumerate(tokens_a):
            tokens.append(token)
            original_tokens.append(instance[index])
            segment_ids.append(0)

        tokens.append("[SEP]")
        original_tokens.append("[SEP]")
        segment_ids.append(0)

        # for token in tokens_b:
        #     tokens.append(token)
        #     segment_ids.append(1)

        # tokens.append("[SEP]")
        # segment_ids.append(1)

        # Get Masked LM predictions
        if self.backend == "c++":
            pass
            # output_tokens, masked_lm_output = mask.create_whole_masked_lm_predictions(
            #     tokens,
            #     original_tokens,
            #     self.vocab_words,
            #     self.tokenizer.vocab,
            #     self.max_predictions_per_seq,
            #     self.masked_lm_prob,
            # )
        elif self.backend == "python":
            output_tokens, masked_lm_output = self.create_whole_masked_lm_predictions(
                tokens
            )

        # Convert to Ids
        input_ids = self.tokenizer.convert_tokens_to_ids(output_tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < self.max_seq_length:
            input_ids.append(PAD)
            segment_ids.append(PAD)
            input_mask.append(PAD)
            masked_lm_output.append(-1)
        return [
            map_to_numpy(input_ids),
            map_to_numpy(input_mask),
            map_to_numpy(segment_ids),
            map_to_numpy(masked_lm_output),
            map_to_numpy([is_next]),
        ]

    def create_masked_lm_predictions(self, tokens):
        cand_indexes = []
        for i, token in enumerate(tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue
            if (
                self.do_whole_word_mask
                and len(cand_indexes) >= 1
                and token.startswith("##")
            ):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

            # cand_indexes.append(i)

        random.shuffle(cand_indexes)
        output_tokens = list(tokens)

        num_to_predict = min(
            self.max_predictions_per_seq,
            max(1, int(round(len(tokens) * self.masked_lm_prob))),
        )

        masked_lms = []
        covered_indexes = set()
        for index in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            if index in covered_indexes:
                continue
            covered_indexes.add(index)

            masked_token = None
            # 80% mask
            if random.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% Keep Original
                if random.random() < 0.5:
                    masked_token = tokens[index]
                # 10% replace w/ random word
                else:
                    masked_token = self.vocab_words[
                        random.randint(0, len(self.vocab_words) - 1)
                    ]

            output_tokens[index] = masked_token
            masked_lms.append(MaskedLMInstance(index=index, label=tokens[index]))

        masked_lms = sorted(masked_lms, key=lambda x: x.index)
        masked_lm_output = [-1] * len(output_tokens)
        for p in masked_lms:
            masked_lm_output[p.index] = self.tokenizer.vocab[p.label]

        return (output_tokens, masked_lm_output)

    def get_new_segment(self, segment):
        """
        输入一句话，返回一句经过处理的话: 为了支持中文全称mask，将被分开的词，将上特殊标记("#")，使得后续处理模块，能够知道哪些字是属于同一个词的。
        :param segment: 一句话
        :return: 一句处理过的话
        """
        seq_cws = jieba.lcut("".join(segment))
        seq_cws_dict = {x: 1 for x in seq_cws}
        new_segment = []
        i = 0
        while i < len(segment):
            if len(self.rec.findall(segment[i])) == 0:  # 不是中文的，原文加进去。
                new_segment.append(segment[i])
                i += 1
                continue

            has_add = False
            for length in range(3, 0, -1):
                if i + length > len(segment):
                    continue
                if "".join(segment[i : i + length]) in seq_cws_dict:
                    new_segment.append(segment[i])
                    for l in range(1, length):
                        new_segment.append("##" + segment[i + l])
                    i += length
                    has_add = True
                    break
            if not has_add:
                new_segment.append(segment[i])
                i += 1
        return new_segment

    def create_whole_masked_lm_predictions(self, tokens):
        """Creates the predictions for the masked LM objective."""

        cand_indexes = []
        for (i, token) in enumerate(tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue
            # Whole Word Masking means that if we mask all of the wordpieces
            # corresponding to an original word. When a word has been split into
            # WordPieces, the first token does not have any marker and any subsequence
            # tokens are prefixed with ##. So whenever we see the ## token, we
            # append it to the previous set of word indexes.
            #
            # Note that Whole Word Masking does *not* change the training code
            # at all -- we still predict each WordPiece independently, softmaxed
            # over the entire vocabulary.
            if (
                self.do_whole_word_mask
                and len(cand_indexes) >= 1
                and token.startswith("##")
            ):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)

        output_tokens = [
            t[2:] if len(self.whole_rec.findall(t)) > 0 else t for t in tokens
        ]  # 去掉"##"

        num_to_predict = min(
            self.max_predictions_per_seq,
            max(1, int(round(len(tokens) * self.masked_lm_prob))),
        )

        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)

                masked_token = None
                # 80% of the time, replace with [MASK]
                if random.random() < 0.8:
                    masked_token = "[MASK]"
                else:
                    # 10% of the time, keep original
                    if random.random() < 0.5:
                        masked_token = (
                            tokens[index][2:]
                            if len(self.whole_rec.findall(tokens[index])) > 0
                            else tokens[index]
                        )  # 去掉"##"
                    # 10% of the time, replace with random word
                    else:
                        masked_token = self.vocab_words[
                            random.randint(0, len(self.vocab_words) - 1)
                        ]

                output_tokens[index] = masked_token

                masked_lms.append(
                    MaskedLMInstance(
                        index=index,
                        label=tokens[index][2:]
                        if len(self.whole_rec.findall(tokens[index])) > 0
                        else tokens[index],
                    )
                )
        assert len(masked_lms) <= num_to_predict
        masked_lms = sorted(masked_lms, key=lambda x: x.index)
        masked_lm_output = [-1] * len(output_tokens)
        for p in masked_lms:
            masked_lm_output[p.index] = self.tokenizer.vocab[p.label]

        return (output_tokens, masked_lm_output)


class PreTrainingDatasetForIdiom:
    def __init__(
        self,
        tokenizer,
        max_seq_length,
        backend="python",
        max_predictions_per_seq: int = 80,
        do_whole_word_mask: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.masked_lm_prob = 0.15
        self.backend = backend
        self.do_whole_word_mask = do_whole_word_mask
        self.max_predictions_per_seq = max_predictions_per_seq
        self.vocab_words = list(tokenizer.vocab.keys())
        self.rec = re.compile("[\u4E00-\u9FA5]")
        self.whole_rec = re.compile("##[\u4E00-\u9FA5]")

        self.mlm_p = 0.15
        self.mlm_mask_p = 0.8
        self.mlm_tamper_p = 0.05
        self.mlm_maintain_p = 0.1

    def tokenize(self, doc_list):
        temp = []
        for d in doc_list:
            temp.append(self.tokenizer.tokenize(d))
        return temp

    def create_training_instance_only_mask_idiom(self, instance):
        is_next = 1
        raw_tokens = instance["tokens"]
        start_pos = instance["idiom_start_pos"]
        end_pos = start_pos + 4
        tokens = []
        tokens.append("[CLS]")
        for index, token in enumerate(raw_tokens):
            if index >= start_pos and index < end_pos:
                tokens.append("[MASK]")
                continue
            tokens.append(token)
        tokens.append("[SEP]")
        # Convert to Ids
        output_tokens = tokens
        masked_lm_output = [-1] * len(output_tokens)
        for i in range(start_pos, end_pos):
            masked_lm_output[i + 1] = self.tokenizer.vocab[raw_tokens[i]]
        input_ids = self.tokenizer.convert_tokens_to_ids(output_tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        while len(input_ids) < self.max_seq_length:
            input_ids.append(PAD)
            segment_ids.append(PAD)
            input_mask.append(PAD)
            masked_lm_output.append(-1)
        return [
            map_to_numpy(input_ids),
            map_to_numpy(input_mask),
            map_to_numpy(segment_ids),
            map_to_numpy(masked_lm_output),
            map_to_numpy([is_next]),
        ]

    def create_training_instance(self, instance):
        instance = instance["tokens"]
        is_next = 1
        raw_text_list = self.get_new_segment(instance)
        tokens_a = raw_text_list
        assert len(tokens_a) == len(instance)
        # tokens_a, tokens_b, is_next = instance.get_values()
        # print(f'is_next label:{is_next}')
        # Create mapper
        tokens = []
        original_tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        original_tokens.append("[CLS]")
        segment_ids.append(0)
        for index, token in enumerate(tokens_a):
            tokens.append(token)
            original_tokens.append(instance[index])
            segment_ids.append(0)

        tokens.append("[SEP]")
        original_tokens.append("[SEP]")
        segment_ids.append(0)

        # for token in tokens_b:
        #     tokens.append(token)
        #     segment_ids.append(1)

        # tokens.append("[SEP]")
        # segment_ids.append(1)

        # Get Masked LM predictions
        if self.backend == "c++":
            pass
            # output_tokens, masked_lm_output = mask.create_whole_masked_lm_predictions(
            #     tokens,
            #     original_tokens,
            #     self.vocab_words,
            #     self.tokenizer.vocab,
            #     self.max_predictions_per_seq,
            #     self.masked_lm_prob,
            # )
        elif self.backend == "python":
            output_tokens, masked_lm_output = self.create_whole_masked_lm_predictions(
                tokens
            )

        # Convert to Ids
        input_ids = self.tokenizer.convert_tokens_to_ids(output_tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < self.max_seq_length:
            input_ids.append(PAD)
            segment_ids.append(PAD)
            input_mask.append(PAD)
            masked_lm_output.append(-1)
        assert len(input_ids) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(masked_lm_output) == self.max_seq_length

        return [
            map_to_numpy(input_ids),
            map_to_numpy(input_mask),
            map_to_numpy(segment_ids),
            map_to_numpy(masked_lm_output),
            map_to_numpy([is_next]),
        ]

    def create_masked_lm_predictions(self, tokens):
        cand_indexes = []
        for i, token in enumerate(tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue
            if (
                self.do_whole_word_mask
                and len(cand_indexes) >= 1
                and token.startswith("##")
            ):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

            # cand_indexes.append(i)

        random.shuffle(cand_indexes)
        output_tokens = list(tokens)

        num_to_predict = min(
            self.max_predictions_per_seq,
            max(1, int(round(len(tokens) * self.masked_lm_prob))),
        )

        masked_lms = []
        covered_indexes = set()
        for index in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            if index in covered_indexes:
                continue
            covered_indexes.add(index)

            masked_token = None
            # 80% mask
            if random.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% Keep Original
                if random.random() < 0.5:
                    masked_token = tokens[index]
                # 10% replace w/ random word
                else:
                    masked_token = self.vocab_words[
                        random.randint(0, len(self.vocab_words) - 1)
                    ]

            output_tokens[index] = masked_token
            masked_lms.append(MaskedLMInstance(index=index, label=tokens[index]))

        masked_lms = sorted(masked_lms, key=lambda x: x.index)
        masked_lm_output = [-1] * len(output_tokens)
        for p in masked_lms:
            masked_lm_output[p.index] = self.tokenizer.vocab[p.label]

        return (output_tokens, masked_lm_output)

    def get_new_segment(self, segment):
        """
        输入一句话，返回一句经过处理的话: 为了支持中文全称mask，将被分开的词，将上特殊标记("#")，使得后续处理模块，能够知道哪些字是属于同一个词的。
        :param segment: 一句话
        :return: 一句处理过的话
        """
        seq_cws = jieba.lcut("".join(segment))
        seq_cws_dict = {x: 1 for x in seq_cws}
        new_segment = []
        i = 0
        while i < len(segment):
            if len(self.rec.findall(segment[i])) == 0:  # 不是中文的，原文加进去。
                new_segment.append(segment[i])
                i += 1
                continue

            has_add = False
            for length in range(3, 0, -1):
                if i + length > len(segment):
                    continue
                if "".join(segment[i : i + length]) in seq_cws_dict:
                    new_segment.append(segment[i])
                    for l in range(1, length):
                        new_segment.append("##" + segment[i + l])
                    i += length
                    has_add = True
                    break
            if not has_add:
                new_segment.append(segment[i])
                i += 1
        return new_segment

    def create_idiom_masked_lm_predictions(self, tokens):
        output_tokens = []

        for (i, token) in enumerate(tokens):
            if token == "#idiom#":
                output_tokens.extend(["[MASK]"] * 4)
            else:
                output_tokens.append(token)

        masked_lm_output = [-1] * len(output_tokens)
        for p in range():
            masked_lm_output[p.index] = self.tokenizer.vocab[p.label]

        return output_tokens, masked_lm_output

    def create_whole_masked_lm_predictions(self, tokens):
        """Creates the predictions for the masked LM objective."""

        cand_indexes = []
        for (i, token) in enumerate(tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue
            # Whole Word Masking means that if we mask all of the wordpieces
            # corresponding to an original word. When a word has been split into
            # WordPieces, the first token does not have any marker and any subsequence
            # tokens are prefixed with ##. So whenever we see the ## token, we
            # append it to the previous set of word indexes.
            #
            # Note that Whole Word Masking does *not* change the training code
            # at all -- we still predict each WordPiece independently, softmaxed
            # over the entire vocabulary.
            if (
                self.do_whole_word_mask
                and len(cand_indexes) >= 1
                and token.startswith("##")
            ):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)

        output_tokens = [
            t[2:] if len(self.whole_rec.findall(t)) > 0 else t for t in tokens
        ]  # 去掉"##"

        num_to_predict = min(
            self.max_predictions_per_seq,
            max(1, int(round(len(tokens) * self.masked_lm_prob))),
        )

        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)

                masked_token = None
                # 80% of the time, replace with [MASK]
                if random.random() < 0.8:
                    masked_token = "[MASK]"
                else:
                    # 10% of the time, keep original
                    if random.random() < 0.5:
                        masked_token = (
                            tokens[index][2:]
                            if len(self.whole_rec.findall(tokens[index])) > 0
                            else tokens[index]
                        )  # 去掉"##"
                    # 10% of the time, replace with random word
                    else:
                        masked_token = self.vocab_words[
                            random.randint(0, len(self.vocab_words) - 1)
                        ]

                output_tokens[index] = masked_token

                masked_lms.append(
                    MaskedLMInstance(
                        index=index,
                        label=tokens[index][2:]
                        if len(self.whole_rec.findall(tokens[index])) > 0
                        else tokens[index],
                    )
                )
        assert len(masked_lms) <= num_to_predict
        masked_lms = sorted(masked_lms, key=lambda x: x.index)
        masked_lm_output = [-1] * len(output_tokens)
        for p in masked_lms:
            masked_lm_output[p.index] = self.tokenizer.vocab[p.label]

        return (output_tokens, masked_lm_output)
