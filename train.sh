torchrun --nnodes=1 --nproc_per_node=8 open_flamingo/train/train.py \
  --lm_path anas-awadalla/mpt-1b-redpajama-200b \
  --tokenizer_path anas-awadalla/mpt-1b-redpajama-200b \
  --cross_attn_every_n_layers 1 \
  --dataset_resampled \
  --batch_size_mmc4 32 \
  --batch_size_laion 64 \
  --train_num_samples_mmc4 1250\
  --train_num_samples_laion 2500 \
  --loss_multiplier_laion 0.2 \
  --workers=8 \
  --run_name OpenFlamingo-3B-vitl-mpt1b \
  --num_epochs 480 \
  --warmup_steps  1875 \
  --mmc4_textsim_threshold 0.24 \
  --mmc4_shards "/scratch/shahils/open_flamingo/data/mmc4_data/mmc4/{0000..0004}.tar" \
  --laion_shards "/scratch/shahils/laion-dataset/laion2b-en-wds/{00000..00606}.tar" \
  --report_to_wandb

  # img2dataset   --url_list "/scratch/shahils/open_flamingo/part-001.snappy.parquet"   --input_format parquet   --url_col url   --caption_col text   --output_format webdataset   --output_folder "/scratch/shahils/open_flamingo/datasets"   --processes_count 8   --thread_count 32   --number_sample_per_shard 5000   --timeout 10   --retry_count 2   --resize_mode keep_ratio_largest   --image_size 256   --save_additional_columns '["license","nsfw","aesthetic_score","similarity","image_suffix","hash","punsafe","pwatermark"]'   --incremental_mode incremental   --log_every 1000

  # unzip -p datasets/mmc4-ff/mmc4_0_5000.zip | tar -cvf data/mmc4_data/5000.tar -T -
