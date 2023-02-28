import math
import torch
from arguments import parse_args
from pretrain_utils import get_model, get_optimizer, get_lr_scheduler, save_ckpt
from utils.exp_util import get_tflops, get_mem_info, throughput_calculator, log_args
from utils.global_vars import set_global_variables, get_timers, get_tensorboard_writer
from utils.logger import Logger
from evaluation import evaluate

from nvidia_bert_dataset_provider import NvidiaBertDatasetProvider
from tqdm import tqdm
import os
import time
from functools import partial
from itertools import chain


from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

from distiller import BasicDisiller, MinilmGeneralDistiller
from distill_config import DistillConfig
from dist_utils import reduce_value, distributed_concat_with_all_gather


def main():

    args = parse_args()
    # init the distributed backend
    if args.local_rank == -1:
        raise ValueError("should use local/")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")

    launch_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    logger = Logger(
        os.path.join(args.log_path, launch_time),
        cuda=torch.cuda.is_available(),
        debug=args.vscode_debug,
    )

    log_args(logger, args)
    args.tokenizer = tokenizer
    args.logger = logger
    set_global_variables(launch_time, args.tensorboard_path)

    world_size = torch.distributed.get_world_size()
    # build model, optimizer and criterion
    student_config, student_model, student_model_numel = get_model(
        args,
        mlm_model_type=args.student_mlm_model_type,
        load_pretrain_model=args.student_load_pretrain_model,
        model_config=args.student_bert_config,
        logger=logger,
    )
    teacher_config, teacher_model, teacher_model_numel = get_model(
        args,
        mlm_model_type=args.teacher_mlm_model_type,
        load_pretrain_model=args.teacher_load_pretrain_model,
        model_config=args.teacher_bert_config,
        logger=logger,
    )
    logger.info(f"Student Model numel: {student_model_numel}")
    logger.info(f"Teacher Model numel: {teacher_model_numel}")

    # Init Distiller
    distill_config = DistillConfig()
    setattr(distill_config, "temperature", 1)
    setattr(distill_config, "is_init_from_teacher", False)
    # setattr(distill_config, "hard_target_weight", 1)
    # setattr(distill_config, "hard_target_weight", 1)
    # setattr(distill_config, "hard_target_weight", 1)

    distiller = MinilmGeneralDistiller(
        distill_config=distill_config,
        teacher_model_config=teacher_config,
        student_model_config=student_config,
        teacher_model=teacher_model,
        student_model=student_model,
    )

    # Register hook
    distiller.register_student_module_hook(student_model)
    distiller.register_teacher_module_hook(teacher_model)

    # ddp
    student_model.to(device)
    teacher_model.to(device)
    if args.local_rank != -1:
        student_model = DDP(
            student_model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
        )
        teacher_model = DDP(
            teacher_model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
        )
        if hasattr(distiller, "project_model"):
            # TODO(没完全支持project)
            logger.info("use project layer!")
            distiller.project_model = DDP(
                distiller.project_model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=False,
            )
    if torch.distributed.get_rank() == 0:
        os.mkdir(os.path.join(args.ckpt_path, launch_time))

    get_tflops_func = partial(
        get_tflops,
        student_model_numel,
        args.train_micro_batch_size_per_gpu,
        args.max_seq_length,
    )
    steps_per_epoch = (
        144003367
        // world_size
        // args.train_micro_batch_size_per_gpu
        // args.gradient_accumulation_steps
        // args.refresh_bucket_size
    )  # len(dataloader)
    total_steps = steps_per_epoch * args.epoch
    total_steps = 1000000  # follow minilm paper

    # build optimizer and lr_scheduler
    start_epoch = 0
    start_shard = 0
    global_step = 0
    if args.resume_train:
        assert os.path.exists(args.load_optimizer_lr)
        o_l_state_dict = torch.load(args.load_optimizer_lr, map_location="cpu")
        o_l_state_dict["lr_scheduler"]["last_epoch"] = (
            o_l_state_dict["lr_scheduler"]["last_epoch"] - 1
        )
        optimizer = get_optimizer(student_model, lr=args.lr)
        optimizer.load_state_dict(o_l_state_dict["optimizer"])
        lr_scheduler = get_lr_scheduler(
            optimizer,
            total_steps=total_steps,
            last_epoch=o_l_state_dict["lr_scheduler"]["last_epoch"],
        )  # o_l_state_dict['lr_scheduler']['last_epoch']
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda(f"cuda:{torch.cuda.current_device()}")
        # if you want delete the above three code, have to move the model to gpu, because in optimizer.step()
        lr_scheduler.load_state_dict(o_l_state_dict["lr_scheduler"])

        start_epoch = o_l_state_dict["epoch"]
        start_shard = o_l_state_dict["shard"] + 1
        # global_step = o_l_state_dict['global_step'] + 1
        logger.info(
            f"resume from epoch {start_epoch} shard {start_shard} step {lr_scheduler.last_epoch} lr {lr_scheduler.get_last_lr()[0]}"
        )
    else:
        optimizer = get_optimizer(student_model, lr=args.lr)
        lr_scheduler = get_lr_scheduler(
            optimizer, total_steps=total_steps, last_epoch=-1
        )

    # optimizer = gpc.config.optimizer.pop('type')(
    # model.parameters(), **gpc.config.optimizer)
    # optimizer = ShardedOptimizerV2(model, optimizer, initial_scale=2**5)

    # build dataloader
    pretrain_dataset_provider = NvidiaBertDatasetProvider(args)

    # engine, _, _, lr_scheduelr = colossalai.initialize(
    #     model=model, optimizer=optimizer, criterion=criterion, lr_scheduler=lr_scheduler
    # )

    logger.info(get_mem_info(prefix="After init model, "))

    best_loss = None
    train_loss = 0
    eval_loss = 0
    local_step = 0

    train_rel_loss = 0
    train_attn_loss = 0
    train_mlm_loss = 0
    eval_mlm_loss = 0
    eval_kd_loss = 0

    timers = get_timers()
    timers("interval_time").start()
    timers("epoch_time").start()
    timers("shard_time").start()

    # 混合精度
    scaler = torch.cuda.amp.GradScaler()
    teacher_model.eval()

    for epoch in range(start_epoch, args.epoch):

        for shard in range(start_shard, len(os.listdir(args.data_path_prefix))):

            dataset_iterator, total_length = pretrain_dataset_provider.get_shard(shard)
            dataset_iterator.sampler.set_epoch(epoch)
            # pretrain_dataset_provider.prefetch_shard(shard + 1) # may cause cpu memory overload
            if torch.distributed.get_rank() == 0:
                iterator_data = tqdm(
                    enumerate(dataset_iterator),
                    total=(
                        total_length
                        // args.train_micro_batch_size_per_gpu
                        // world_size
                    ),
                    colour="cyan",
                    smoothing=1,
                )
            else:
                iterator_data = enumerate(dataset_iterator)

            student_model.train()
            for step, batch_data in iterator_data:
                # batch_data = pretrain_dataset_provider.get_batch(batch_index)
                input_ids = batch_data[0].cuda(f"cuda:{torch.cuda.current_device()}")
                attention_mask = batch_data[1].cuda(
                    f"cuda:{torch.cuda.current_device()}"
                )
                token_type_ids = batch_data[2].cuda(
                    f"cuda:{torch.cuda.current_device()}"
                )
                mlm_label = batch_data[3].cuda(f"cuda:{torch.cuda.current_device()}")
                # nsp_label = batch_data[5].cuda()
                distiller.clear_cache()
                with torch.cuda.amp.autocast(enabled=args.use_amp):
                    student_output = student_model(
                        input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask,
                        labels=mlm_label,
                    )
                    with torch.no_grad():
                        teacher_output = teacher_model(
                            input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                        )
                    loss, loss_dict = distiller.compute_loss(
                        student_output, teacher_output, attention_mask
                    )

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # backward
                if args.use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                pretrain_dataset_provider.prefetch_batch()

                local_step += 1

                train_mlm_loss += (
                    reduce_value(student_output.loss) / args.gradient_accumulation_steps
                )
                cur_all_loss = reduce_value(loss)
                # loss_cat = distributed_concat_with_all_gather(loss)
                # logger.info("cur_all_loss: {}".format(cur_all_loss))
                # logger.info("loss_cat: {}".format(loss_cat))

                # loss_dict["attention_loss"] = loss
                # loss_dict["value_rel_loss"] = loss
                train_loss += cur_all_loss
                cur_attn_loss = reduce_value(loss_dict["attention_loss"])
                train_attn_loss += cur_attn_loss / args.gradient_accumulation_steps
                cur_rel_loss = reduce_value(loss_dict["value_rel_loss"])
                train_rel_loss += cur_rel_loss / args.gradient_accumulation_steps

                # logger.info("train_attn_loss: {}".format(cur_attn_loss))
                # logger.info(
                #     "train_attn_loss cat: {}".format(
                #         distributed_concat_with_all_gather(loss_dict["attention_loss"])
                #     )
                # )
                # logger.info("value_rel_loss: {}".format(cur_rel_loss))
                # logger.info(
                #     "value_rel_loss cat: {}".format(
                #         distributed_concat_with_all_gather(loss_dict["value_rel_loss"])
                #     )
                # )
                # logger.info("kd_loss: {}".format(reduce_value(loss_dict["kd_loss"])))
                # logger.info(
                #     "kd_loss cat: {}".format(
                #         distributed_concat_with_all_gather(loss_dict["kd_loss"])
                #     )
                # )
                # logger.info("keys: {}".format(loss_dict.keys()))
                # train_attn_loss += reduce_value(loss_dict["kd_loss"])
                # train_rel_loss += reduce_value(loss_dict["value_rel_loss"])
                if local_step % args.gradient_accumulation_steps == 0:
                    if args.use_amp:
                        scaler.unscale_(optimizer)
                    # clip param
                    if hasattr(distiller, "use_project") and distiller.use_project:
                        torch.nn.utils.clip_grad_norm_(
                            chain(
                                student_model.parameters(),
                                distiller.project.parameters(),
                            ),
                            args.max_grad_norm,
                            error_if_nonfinite=False,
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            student_model.parameters(), args.max_grad_norm
                        )
                    if args.use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if (
                        global_step % args.log_interval == 0
                        and global_step != 0
                        and torch.distributed.get_rank() == 0
                    ):
                        elapsed_time = timers("interval_time").elapsed(reset=False)
                        elapsed_time_per_iteration = elapsed_time / global_step
                        (
                            samples_per_sec,
                            tflops,
                            approx_parameters_in_billions,
                        ) = throughput_calculator(
                            student_model_numel,
                            args,
                            student_config,
                            elapsed_time,
                            global_step,
                            world_size,
                        )

                        cur_loss = train_loss / args.log_interval
                        current_lr = lr_scheduler.get_last_lr()[0]
                        rel_loss = train_rel_loss / args.log_interval
                        attn_loss = train_attn_loss / args.log_interval
                        cur_mlm_loss = train_mlm_loss / args.log_interval
                        ppl = math.exp(cur_mlm_loss)
                        log_str = (
                            f"| epoch: {epoch} | shard: {shard} | step: {global_step} | lr {current_lr:.7f} | elapsed_time: {elapsed_time / 60 :.3f} minutes "
                            + f"| mins/batch: {elapsed_time_per_iteration :.3f} seconds | loss: {cur_loss:.7f} |  attn loss: {attn_loss:.7f} | rel loss: {rel_loss:.7f} | ppl:{ppl:.7f}  | TFLOPS: {get_tflops_func(elapsed_time_per_iteration):.3f} or {tflops:.3f}"
                        )  # TODO(补充有效日志)
                        logger.info(log_str, print_=False)

                        if args.wandb:
                            tensorboard_log = get_tensorboard_writer()
                            tensorboard_log.log_train(
                                {
                                    "lr": current_lr,
                                    "loss": cur_loss,
                                    "rel_loss": rel_loss,
                                    "attn_loss": attn_loss,
                                    "ppl": ppl,
                                    "mins_batch": elapsed_time_per_iteration,
                                },
                                # "ppl": math.exp(train_mlm_loss / args.log_interval),
                                global_step,
                            )
                        train_loss = 0
                        train_attn_loss = 0
                        train_rel_loss = 0
                        train_mlm_loss = 0

            logger.info(
                f'epoch {epoch} shard {shard} has cost {timers("shard_time").elapsed() / 60 :.3f} mins'
            )
            logger.info("*" * 100)
            cur_eval_loss = evaluate(
                student_model, teacher_model, distiller, args, logger, global_step
            )
            eval_loss += cur_eval_loss
            # eval_mlm_loss += cur_eval_mlm_loss
            # eval_kd_loss += cur_eval_kd_loss
            save_ckpt(
                student_model,
                optimizer,
                lr_scheduler,
                os.path.join(
                    args.ckpt_path,
                    launch_time,
                    f"epoch-{epoch}_shard-{shard}_" + launch_time,
                ),
                epoch,
                shard,
                global_step,
            )

        eval_loss /= len(os.listdir(args.data_path_prefix))
        logger.info(
            f'epoch {epoch} | shard_length {len(os.listdir(args.data_path_prefix))} | elapsed_time: {timers("epoch_time").elapsed() / 60 :.3f} mins'
            + f"eval_loss: {eval_loss}"
        )
        logger.info("-" * 100)
        if args.wandb and torch.distributed.get_rank() == 0:
            tensorboard_log = get_tensorboard_writer()
            tensorboard_log.log_eval(
                {
                    "all_eval_shard_loss": eval_loss,
                },
                epoch,
            )
        start_shard = 0
        eval_loss = 0

    pretrain_dataset_provider.release_shard()

    logger.info("Congratulation, training has finished!!!")


if __name__ == "__main__":
    main()
