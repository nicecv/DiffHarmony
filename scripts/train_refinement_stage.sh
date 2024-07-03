export PYTHONPATH=.:$PYTHONPATH
ACC_CONFIG_FILE="configs/acc_configs/multi_default.yaml"
export CUDA_VISIBLE_DEVICES="0,1"
NUM_PROCESSES=2
MASTER_PORT=29500

OUTPUT_DIR=""
mkdir -p $OUTPUT_DIR
cat "$0" >> $OUTPUT_DIR/run_script.sh

accelerate launch --config_file $ACC_CONFIG_FILE --num_processes $NUM_PROCESSES --main_process_port $MASTER_PORT \
scripts/train/refinement_stage.py \
	--pipeline_path "checkpoints/stable-diffusion-inpainting" \
	--pretrained_vae_path "checkpoints/sd-vae-ft-mse" \
	--pretrained_unet_path "checkpoints/base/unet" \
	--model_path configs/stage2_configs/base.json \
	--output_dir $OUTPUT_DIR \
	--seed= \
	--dataloader_num_workers= \
	--train_batch_size= \
	--num_train_epochs= \
	--gradient_accumulation_steps= \
	--learning_rate= \
	--lr_scheduler "" \
	--lr_warmup_ratio= \
	--use_ema \
	--ema_decay= \
	--adam_weight_decay= \
	--mixed_precision="fp16" \
	--checkpointing_epochs= \
	--checkpoints_total_limit= \
	--infer_resolution=512 \
	--resolution=256 \
	--in_channels=7 \
	--dataset_root "data/iHarmony4" \
	--train_file "train.jsonl" \
	--gradient_checkpointing \
	--enable_xformers_memory_efficient_attention

	# --kl_div_weight=1e-8 \