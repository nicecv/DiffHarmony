export PYTHONPATH=.:$PYTHONPATH
ACC_CONFIG_FILE="configs/acc_configs/multi_default.yaml"
export CUDA_VISIBLE_DEVICES="0,1"
NUM_PROCESSES=2
MASTER_PORT=29500

DATASET_ROOT=
OUTPUT_DIR=$DATASET_ROOT/cand_composite_images
TEST_FILE="all_mask_metadata.jsonl"


accelerate launch --config_file $ACC_CONFIG_FILE --num_processes $NUM_PROCESSES --main_process_port $MASTER_PORT \
scripts/inference/inverse.py \
	--pretrained_model_name_or_path checkpoints/stable-diffusion-inpainting \
	--pretrained_vae_model_name_or_path  \
	--pretrained_unet_model_name_or_path  \
	--dataset_root $DATASET_ROOT \
	--test_file $TEST_FILE \
	--output_dir $OUTPUT_DIR \
	--seed=0 \
	--resolution=1024 \
	--output_resolution=1024 \
	--eval_batch_size= \
	--dataloader_num_workers= \
	--mixed_precision="fp16" \
	--rounds=10