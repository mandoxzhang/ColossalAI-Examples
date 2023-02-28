import os
import math
import torch
from tqdm import tqdm
from utils.global_vars import get_timers, get_tensorboard_writer
from nvidia_bert_dataset_provider import NvidiaBertDatasetProvider
from dist_utils import reduce_value, distributed_concat_with_all_gather


def evaluate(student_model, teacher_model, distiller, args, logger, global_step):
    evaluate_dataset_provider = NvidiaBertDatasetProvider(args, evaluate=True)
    start_shard = 0

    student_model.eval()
    timers = get_timers()
    eval_step = 0
    eval_loss = 0
    eval_attn_loss = 0
    eval_rel_loss = 0
    eval_mlm_loss = 0

    cur_loss = 0
    world_size = torch.distributed.get_world_size()

    with torch.no_grad():

        for shard in range(start_shard, len(os.listdir(args.eval_data_path_prefix))):

            timers("eval_shard_time").start()

            dataset_iterator, total_length = evaluate_dataset_provider.get_shard(shard)
            # evaluate_dataset_provider.prefetch_shard(shard + 1)
            if torch.distributed.get_rank() == 0:
                iterator_data = tqdm(
                    enumerate(dataset_iterator),
                    total=(
                        total_length // args.eval_micro_batch_size_per_gpu // world_size
                    ),
                    colour="MAGENTA",
                    smoothing=1,
                )
            else:
                iterator_data = enumerate(dataset_iterator)

            for (
                step,
                batch_data,
            ) in (
                iterator_data
            ):  # tqdm(enumerate(dataset_iterator), total=(total_length // args.train_micro_batch_size_per_gpu // world_size), colour='cyan', smoothing=1):

                # batch_data = pretrain_dataset_provider.get_batch(batch_index)
                eval_step += 1
                input_ids = batch_data[0].cuda()
                attention_mask = batch_data[1].cuda()
                token_type_ids = batch_data[2].cuda()
                mlm_label = batch_data[3].cuda()
                # nsp_label = batch_data[5].cuda()
                distiller.clear_cache()
                student_output = student_model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    labels=mlm_label,
                )
                teacher_output = teacher_model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                )

                loss, loss_dict = distiller.compute_loss(
                    student_output, teacher_output, attention_mask
                )
                evaluate_dataset_provider.prefetch_batch()
                cur_all_loss = reduce_value(loss)
                # loss_cat = distributed_concat_with_all_gather(loss)
                # logger.info("cur_all_loss: {}".format(cur_all_loss))
                # logger.info("loss_cat: {}".format(loss_cat))
                # eval_mlm_loss += reduce_value(student_output.loss)
                # loss_cat = distributed_concat_with_all_gather(loss)
                # logger.info("cur_all_loss: {}".format(cur_all_loss))
                # logger.info("loss_cat: {}".format(loss_cat))
                eval_mlm_loss += reduce_value(student_output.loss)
                eval_loss += cur_all_loss
                cur_attn_loss = reduce_value(loss_dict["attention_loss"])
                eval_attn_loss += cur_attn_loss
                cur_rel_loss = reduce_value(loss_dict["value_rel_loss"])
                eval_rel_loss += cur_rel_loss

            cur_loss = eval_loss / eval_step
            eval_attn_loss = eval_attn_loss / eval_step
            eval_rel_loss = eval_rel_loss / eval_step
            eval_mlm_loss = eval_mlm_loss / eval_step
            elapsed_time = timers("eval_shard_time").elapsed()
            elapsed_time_per_iteration = elapsed_time / eval_step
            eval_ppl = math.exp(eval_mlm_loss)
            # ppl = math.exp(eval_mlm_loss / eval_step)

            if args.wandb and torch.distributed.get_rank() == 0:
                tensorboard_log = get_tensorboard_writer()
                tensorboard_log.log_eval(
                    {
                        "loss": cur_loss,
                        "attn_loss": eval_attn_loss,
                        "rel_loss": eval_rel_loss,
                        "mins_batch": elapsed_time_per_iteration,
                        "eval_ppl": eval_ppl,
                        "eval_mlm_loss": eval_mlm_loss,
                    },
                    global_step,
                )

            eval_log_str = (
                f"evaluation shard: {shard} | step: {eval_step} | elapsed_time: {elapsed_time / 60 :.3f} minutes "
                + f"| mins/batch: {elapsed_time_per_iteration :.3f} seconds | loss: {cur_loss:.7f} |  attn loss: {eval_attn_loss:.7f} | rel loss: {eval_rel_loss:.7f} | ppl loss: {eval_ppl:.7f}"
            )

            logger.info(eval_log_str)
            logger.info("-" * 100)
            logger.info("")

    evaluate_dataset_provider.release_shard()
    student_model.train()
    return cur_loss
