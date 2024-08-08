import logging
import sys
from typing import Literal
import transformers
import datasets
from transformers import TrainingArguments, Trainer
from datasets import load_from_disk, Dataset
from transformers import set_seed, AutoConfig, AutoModelForCausalLM
from utils import H4ArgumentParser
from dataclasses import dataclass

from TrapezoidLRScheduler import TrapezoidLRScheduler

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str
    dataset_name_or_path: str
    tokenizer_name_or_path: str
    seq_len: int
    attn_implementation: Literal['eager', 'flash_attention_2', 'sdpa']


class MorphPieceTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_optimizer_and_scheduler(self, num_training_steps):

        pct_start = (
            self.args.warmup_steps/num_training_steps if self.args.warmup_steps > 0 else self.args.warmup_ratio
        )
        
        if self.args.lr_scheduler_kwargs['step_cooldown']>0:
            assert self.args.lr_scheduler_kwargs['step_cooldown']<=num_training_steps, f"cooldown start {self.args.lr_scheduler_kwargs['step_cooldown']} cannot be greater than total training steps : {num_training_steps}"
        
        self.lr_scheduler = TrapezoidLRScheduler(
            optimizer=self.optimizer,
            max_lr=self.args.learning_rate,
            total_steps=num_training_steps,
            pct_start=pct_start,
            pct_cooldown=self.args.lr_scheduler_kwargs['pct_cooldown'],
            step_cooldown=self.args.lr_scheduler_kwargs['step_cooldown'],
            )

# def main(det_callback, tb_callback, model_args, extra_args, training_args):
def main(model_args, training_args):

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    ###########################
    # Setup model and tokenizer
    ###########################

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.n_positions = model_args.seq_len
    config.vocab_size = 50304
    config._attn_implementation = model_args.attn_implementation
    model = AutoModelForCausalLM.from_config(config)
    # model = GPT2LMHeadModel(config=config)
    # model = LlamaForCausalLM(config=config)
    # model.resize_token_embeddings(tokenizer.vocab_size,pad_to_multiple_of=128)

    ###############
    # Setup dataset
    ###############

    dataset = load_from_disk(
        f'/pfss/mlde/workspaces/mlde_wsp_MorphPiece/data/fineweb-edu/{model_args.dataset_name_or_path}')
    dataset.set_format('torch')

    if model_args.seq_len == 1024:  # Default dataset block_size is 2048
        dataset = Dataset.from_dict({'input_ids': dataset['input_ids'].reshape(-1, 1024),
                                    'labels': dataset['labels'].reshape(-1, 1024)})

    dataset = dataset.train_test_split(
        test_size=0.001, seed=training_args.seed)

    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in dataset.items()]}"
    )

    ##############################
    # Additional TrainingArguments
    ##############################

    training_args.set_push_to_hub(model_id=f'maveriq/morph_{model_args.model_name_or_path}',
                                  private_repo=True, strategy="all_checkpoints")

    #####################
    # Instantiate Trainer
    #####################

    trainer = MorphPieceTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        # data_collator=default_data_collator,
        callbacks=[],
        # compute_metrics=compute_metrics,
    )

    # trainer.add_callback(det_callback)
    # trainer.add_callback(tb_callback)

    ###############
    # Training loop
    ###############

    logger.info("*** Train ***")
    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")

    kwargs = {
        "language": "en",
        "license": "cc-by-nc-sa-4.0",
        "tags": training_args.to_dict(),
        "dataset": model_args.dataset_name_or_path,
    }

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)
    else:
        logger.warning(
            "Not pushing to hub. Set `push_to_hub` to True to push to hub.")

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type(
            "FULL_STATE_DICT")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process

    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    # info = det.get_cluster_info()
    # assert info
    # hparams = info.trial.hparams
    parser = H4ArgumentParser(
        (ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse()
    main(model_args, training_args)

    # if training_args.deepspeed:
    #     distributed = det.core.DistributedContext.from_deepspeed()
    # else:
    #     distributed = det.core.DistributedContext.from_torch_distributed()

    # with det.core.init(distributed=distributed) as core_context:
    #     user_data = {
    #         "trained_from_scratch": model_args.model_name_or_path,
    #         "tasks": "language-modeling",
    #         "dataset": model_args.dataset_name,
    #         "tags": ["language-modeling", "nlp"],
    #     }

    #     det_callback = DetCallback(
    #         core_context, training_args, user_data=user_data)

    #     tb_callback = TensorBoardCallback(
    #         tb_writer=SummaryWriter(core_context.train.get_tensorboard_path())
    #     )
    #     main(det_callback, tb_callback, model_args, extra_args, training_args)
