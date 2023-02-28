import time
import os
import re
import psutil
import h5py
import socket
import argparse
import numpy as np
import multiprocessing
from tqdm import tqdm
from random import shuffle
from transformers import AutoTokenizer
from get_mask import PreTrainingDatasetForIdiom


def read_idiom_corpus_examples(corpus_path):
    datas = []
    skip_num = 0
    with open(corpus_path, mode="rb") as f:
        for idx, data_str in tqdm(enumerate(f)):
            data = eval(data_str.decode("utf8"))
            context = data["content"]
            clean_context = ""
            pos_list = []
            idiom_list = []
            # 填充所有缺失的文本
            sent_list = context.split("#idiom#")
            if len(sent_list) != len(data["groundTruth"]) + 1:
                skip_num += 1
                continue
            assert len(sent_list) == len(data["groundTruth"]) + 1
            for i, sent in enumerate(sent_list):
                clean_context += sent
                if i == len(sent_list) - 1:
                    continue
                pos_list.append(len(clean_context))
                idiom = data["groundTruth"][i]
                idiom_list.append(idiom)
                clean_context += idiom

            # for i, (tag, idiom) in enumerate(
            #     zip(re.finditer("#idiom#", context), data["groundTruth"])
            # ):
            #     new_tag = idx * 20 + i
            #     # tag_str = "#idiom%06d#" % new_tag
            #     tmp_context = clean_context
            #     clean_context = "".join(
            #         (
            #             tmp_context[: tag.start(0)],
            #             idiom,
            #             tmp_context[tag.end(0) :],
            #         )
            #     )
            #     cur_pos = len(tmp_context[: tag.start(0)]) + len(idiom) - 1
            #     pos_list.append(cur_pos)
            #     idiom_list.append(idiom)
            # tmp_context = tmp_context.replace("#idiom#", "[UNK]")
            # tmp_context = tmp_context.replace(tag_str, "#idiom#")
            for pos, idiom in zip(pos_list, idiom_list):
                if len(idiom) != 4:
                    # skip_num += 1
                    continue
                assert clean_context[pos : pos + 4] == idiom
                datas.append({"context": clean_context, "idiom": idiom, "pos": pos})
                if idx < 5:
                    print(datas[-1])
    print("skip_num: ", skip_num)
    return datas


def get_idiom_instance(example, max_sequence_length=128):
    start_pos = example["pos"]
    end_pos = start_pos + 4
    before_part = example["context"][:start_pos]
    after_part = example["context"][end_pos:]
    assert example["context"][start_pos:end_pos] == example["idiom"]
    # parts = re.split("#idiom#", example.context)
    # assert len(parts) == 2
    before_part = (
        pretrain_data.tokenizer.tokenize(before_part) if len(before_part) > 0 else []
    )
    after_part = (
        pretrain_data.tokenizer.tokenize(after_part) if len(after_part) > 0 else []
    )
    idiom_part = pretrain_data.tokenizer.tokenize(example["idiom"])
    max_sequence_length = max_sequence_length - 2
    half_sequence_length = int((max_sequence_length - 4) / 2)
    cur_sequence_length = len(before_part) + len(idiom_part) + len(after_part)
    # 裁剪
    if cur_sequence_length > max_sequence_length:
        if len(before_part) <= half_sequence_length:
            after_part_length = max_sequence_length - 4 - len(before_part)
            after_part = after_part[:after_part_length]
        elif len(before_part) > half_sequence_length:
            before_part = before_part[-half_sequence_length:]
            if len(after_part) > half_sequence_length:
                after_part = after_part[:half_sequence_length]

    tokens = before_part + idiom_part + after_part
    idiom_start_pos = len(before_part)
    assert len(tokens) <= max_sequence_length
    return {
        "tokens": tokens,
        "idiom_start_pos": idiom_start_pos,
        "idiom": example["idiom"],
    }


def get_raw_instance(document, max_sequence_length=512):

    """
    获取初步的训练实例，将整段按照max_sequence_length切分成多个部分,并以多个处理好的实例的形式返回。
    :param document: 一整段
    :param max_sequence_length:
    :return: a list. each element is a sequence of text
    """
    # document = self.documents[index]
    max_sequence_length_allowed = max_sequence_length - 2
    # document = [seq for seq in document if len(seq)<max_sequence_length_allowed]
    sizes = [len(seq) for seq in document]

    result_list = []
    curr_seq = []  # 当前处理的序列
    sz_idx = 0
    while sz_idx < len(sizes):
        # 当前句子加上新的句子，如果长度小于最大限制，则合并当前句子和新句子；否则即超过了最大限制，那么做为一个新的序列加到目标列表中

        if (
            len(curr_seq) + sizes[sz_idx] <= max_sequence_length_allowed
        ):  # or len(curr_seq)==0:
            curr_seq += document[sz_idx]
            sz_idx += 1
        elif sizes[sz_idx] >= max_sequence_length_allowed:
            if len(curr_seq) > 0:
                result_list.append(curr_seq)
            curr_seq = []
            result_list.append(document[sz_idx][:max_sequence_length_allowed])
            sz_idx += 1
        else:
            result_list.append(curr_seq)
            curr_seq = []
    # 对最后一个序列进行处理，如果太短的话，丢弃掉。
    if len(curr_seq) > max_sequence_length_allowed / 2:  # /2
        result_list.append(curr_seq)

    # # 计算总共可以得到多少份
    # num_instance=int(len(big_list)/max_sequence_length_allowed)+1
    # print("num_instance:",num_instance)
    # # 切分成多份，添加到列表中
    # result_list=[]
    # for j in range(num_instance):
    #     index=j*max_sequence_length_allowed
    #     end_index=index+max_sequence_length_allowed if j!=num_instance-1 else -1
    #     result_list.append(big_list[index:end_index])
    return result_list


def split_numpy_chunk_pool(
    input_path, output_path, pretrain_data, worker, dupe_factor, seq_len, file_name
):

    if os.path.exists(os.path.join(output_path, f"{file_name}.h5")):
        print(f"{file_name}.h5 exists")
        return

    # documents = []
    instances = []

    s = time.time()
    examples = read_idiom_corpus_examples(corpus_path=input_path)
    print(f"read_file cost {time.time() - s}, length is {len(examples)}")

    ans = []
    s = time.time()
    # tokenize
    instances = []
    pool = multiprocessing.Pool(worker)
    encoded_doc = pool.imap_unordered(get_idiom_instance, examples, 100)
    for index, res in tqdm(
        enumerate(encoded_doc, start=1), total=len(examples), colour="cyan"
    ):
        instances.append(res)
    pool.close()
    # for example in tqdm(examples, colour="MAGENTA"):
    #     instance = get_idiom_instance(example, max_sequence_length=128)
    #     instances.append(instance)
    print((time.time() - s) / 60)
    del examples

    print("len instance", len(instances))

    # new_instances = []
    # for _ in range(dupe_factor):
    #     for ins in instances:
    #         new_instances.append(ins)

    shuffle(instances)
    # instances = new_instances
    print("after dupe_factor, len instance", len(instances))

    sentence_num = len(instances)
    sentence_num = sentence_num
    add_sentence_num = int(sentence_num * 0.2)
    input_ids = np.zeros([sentence_num + add_sentence_num, seq_len], dtype=np.int32)
    input_mask = np.zeros([sentence_num + add_sentence_num, seq_len], dtype=np.int32)
    segment_ids = np.zeros([sentence_num + add_sentence_num, seq_len], dtype=np.int32)
    masked_lm_output = np.zeros(
        [sentence_num + add_sentence_num, seq_len], dtype=np.int32
    )

    s = time.time()
    pool = multiprocessing.Pool(worker)
    encoded_docs = pool.imap_unordered(
        pretrain_data.create_training_instance_only_mask_idiom, instances, 32
    )
    add_encoded_docs = pool.imap_unordered(
        pretrain_data.create_training_instance, instances[:add_sentence_num], 32
    )

    for index, mask_dict in tqdm(
        enumerate(encoded_docs), total=len(instances), colour="blue"
    ):
        if index < 5:
            print(instances[index])
            print("input_ids: ", mask_dict[0])
            print("input_mask: ", mask_dict[1])
            print("segment_ids: ", mask_dict[2])
            print("masked_lm_output: ", mask_dict[3])
        input_ids[index] = mask_dict[0]
        input_mask[index] = mask_dict[1]
        segment_ids[index] = mask_dict[2]
        masked_lm_output[index] = mask_dict[3]
    for index, mask_dict in tqdm(
        enumerate(add_encoded_docs), total=add_sentence_num, colour="red"
    ):
        if index < 5:
            print(instances[index])
            print("input_ids: ", mask_dict[0])
            print("input_mask: ", mask_dict[1])
            print("segment_ids: ", mask_dict[2])
            print("masked_lm_output: ", mask_dict[3])
        input_ids[sentence_num + index] = mask_dict[0]
        input_mask[sentence_num + index] = mask_dict[1]
        segment_ids[sentence_num + index] = mask_dict[2]
        masked_lm_output[sentence_num + index] = mask_dict[3]
    pool.close()
    print((time.time() - s) / 60)

    with h5py.File(os.path.join(output_path, f"{file_name}.h5"), "w") as hf:
        hf.create_dataset("input_ids", data=input_ids)
        hf.create_dataset("input_mask", data=input_mask)
        hf.create_dataset("segment_ids", data=segment_ids)
        hf.create_dataset("masked_lm_positions", data=masked_lm_output)

    del instances


def split_numpy_chunk(path, tokenizer, pretrain_data, host):

    documents = []
    instances = []
    # read原始corpus, 读取成[[line1, line2, ...], [], []]
    s = time.time()
    with open(path, encoding="utf-8") as fd:
        document = []
        for i, line in enumerate(tqdm(fd)):
            line = line.strip()
            # document = line
            # if len(document.split("<sep>")) <= 3:
            #     continue
            if len(line) > 0 and line[:2] == "]]":  # This is end of document
                documents.append(document)
                document = []
            elif len(line) >= 2:
                document.append(line)
        if len(document) > 0:
            documents.append(document)
    print("read_file ", time.time() - s)

    # documents = [x for x in documents if x]
    # print(len(documents))
    # print(len(documents[0]))
    # print(documents[0][0:10])
    from typing import List
    import multiprocessing

    # 对文本进行tokenize
    ans = []
    for docs in tqdm(documents):
        ans.append(pretrain_data.tokenize(docs))
    print(time.time() - s)
    del documents

    # merge，句子合并成比较大的粒度，不超过512
    instances = []
    for a in tqdm(ans):
        raw_ins = get_raw_instance(a)
        instances.extend(raw_ins)
    del ans

    print("len instance", len(instances))

    sen_num = len(instances)
    # seq_len = 128  # corpus不超过128
    input_ids = np.zeros([sen_num, seq_len], dtype=np.int32)
    input_mask = np.zeros([sen_num, seq_len], dtype=np.int32)
    segment_ids = np.zeros([sen_num, seq_len], dtype=np.int32)
    masked_lm_output = np.zeros([sen_num, seq_len], dtype=np.int32)

    for index, ins in tqdm(enumerate(instances)):
        # 每一个instance创建得到
        mask_dict = pretrain_data.create_training_instance(ins)
        input_ids[index] = mask_dict[0]
        input_mask[index] = mask_dict[1]
        segment_ids[index] = mask_dict[2]
        masked_lm_output[index] = mask_dict[3]

    with h5py.File(f"./output/{host}.h5", "w") as hf:
        hf.create_dataset("input_ids", data=input_ids)
        hf.create_dataset("input_mask", data=input_ids)
        hf.create_dataset("segment_ids", data=segment_ids)
        hf.create_dataset("masked_lm_positions", data=masked_lm_output)

    del instances


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        default=10,
        help="path of tokenizer",
    )
    parser.add_argument("--seq_len", type=int, default=512, help="sequence length")
    parser.add_argument(
        "--max_predictions_per_seq",
        type=int,
        default=80,
        help="number of shards, e.g., 10, 50, or 100",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="input path of shard which has split sentence",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="output path of h5 contains token id",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="python",
        help="backend of mask token, python, c++, numpy respectively",
    )
    parser.add_argument(
        "--dupe_factor",
        type=int,
        default=1,
        help="specifies how many times the preprocessor repeats to create the input from the same article/document",
    )
    parser.add_argument("--worker", type=int, default=32, help="number of process")
    parser.add_argument("--server_num", type=int, default=10, help="number of servers")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    pretrain_data = PreTrainingDatasetForIdiom(
        tokenizer,
        args.seq_len,
        args.backend,
        max_predictions_per_seq=args.max_predictions_per_seq,
    )

    data_len = len(os.listdir(args.input_path))
    print(data_len)
    for i in range(data_len):
        if i > 1:
            break
        # input_path = os.path.join(args.input_path, f"{i}.txt")
        input_path = os.path.join(args.input_path, "pretrain_data_{}.txt".format(i))
        if os.path.exists(input_path):
            start = time.time()
            print(f"process {input_path}")
            split_numpy_chunk_pool(
                input_path,
                args.output_path,
                pretrain_data,
                args.worker,
                args.dupe_factor,
                args.seq_len,
                i,
            )
            end_ = time.time()
            print(
                "memory：%.4f GB"
                % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024)
            )
            print(f"has cost {(end_ - start) / 60}")
            print("-" * 100)
            print("")

    # if you have multiple server, you can use code below or modify code to openmpi

    # host = int(socket.gethostname().split('GPU')[-1])
    # for i in range(data_len // args.server_num + 1):
    #     h = args.server_num * i + host - 1
    #     input_path = os.path.join(args.input_path, f'{h}.txt')
    #     if os.path.exists(input_path):
    #         start = time.time()
    #         print(f'I am server {host}, process {input_path}')
    #         split_numpy_chunk_pool(input_path,
    #                                 args.output_path,
    #                                 pretrain_data,
    #                                 args.worker,
    #                                 args.dupe_factor,
    #                                 args.seq_len,
    #                                 h)
    #         end_ = time.time()
    #         print(u'memory：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
    #         print(f'has cost {(end_ - start) / 60}')
    #         print('-' * 100)
    #         print('')
