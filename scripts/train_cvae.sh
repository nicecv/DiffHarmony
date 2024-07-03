export PYTHONPATH=.:$PYTHONPATH
ACC_CONFIG_FILE="configs/acc_configs/multi_default.yaml"
export CUDA_VISIBLE_DEVICES="0,1"
NUM_PROCESSES=2
MASTER_PORT=29500

OUTPUT_DIR=""
mkdir -p $OUTPUT_DIR
cat "$0" >> $OUTPUT_DIR/run_script.sh

accelerate launch --config_file $ACC_CONFIG_FILE --num_processes $NUM_PROCESSES --main_process_port $MASTER_PORT \
scripts/train/cvae.py \
	--pretrained_vae_model_name_or_path checkpoints/sd-vae-ft-mse \
	--output_dir $OUTPUT_DIR \
	--seed= \
	--train_batch_size= \
	--eval_batch_size= \
	--dataloader_num_workers= \
	--num_train_epochs= \
	--gradient_accumulation_steps= \
	--learning_rate= \
	--lr_scheduler "" \
	--lr_warmup_ratio= \
	--use_ema \
    --adam_weight_decay= \
	--ema_decay= \
	--mixed_precision="fp16" \
	--checkpointing_epochs= \
	--checkpoints_total_limit= \
	--image_logging_epochs= \
	--dataset_root "data/iHarmony4" \
	--train_file "train.jsonl" \
	--test_file "test.jsonl" \
	--resolution=256 \
	--additional_in_channels=1 \
	--gradient_checkpointing