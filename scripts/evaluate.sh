export PYTHONPATH=.:$PYTHONPATH

OUTPUT_DIR=""
DATA_DIR=data/iHarmony4
TEST_FILE=test.jsonl

python scripts/evaluate/main.py \
	--input_dir $OUTPUT_DIR \
	--output_dir $OUTPUT_DIR-evaluation \
	--data_dir $DATA_DIR \
	--json_file_path $TEST_FILE \
	--resolution=256 \
	--use_gt_bg \
	--num_processes=32