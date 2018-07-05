#!/bin/sh

# Pre-train

python3 pretrain_predictor.py --data_dir tacred_20 --dataset labeled.json --save_dir ./saved_predictor --id 00 --seed 905 --log_step 100

python3 pretrain_selector.py --data_dir tacred_20 --dataset labeled.json --save_dir ./saved_selector --id 00 --seed 905 --log_step 100

# Sample from selector to update predictor

python3 sample.py --p_dir ./saved_predictor/00 --s_dir ./saved_selector/00 --data_dir tacred_20 --out ./tacred_20/extra_predictor.json --num 8000 --mode p

python3 update_predictor.py --data_dir tacred_20 --dataset extra_predictor.json --save_dir ./saved_predictor --id 00 --seed 905 --log_step 100

# Sample from predictor to update selector

python3 sample.py --p_dir ./saved_predictor/00 --s_dir ./saved_selector/00 --data_dir tacred_20 --out ./tacred_20/extra_selector.json --num 8000 --mode s

python3 update_selector.py --data_dir tacred_20 --dataset extra_selector.json --save_dir ./saved_selector --id 00 --seed 905 --log_step 100

# Sample from selector to update predictor

python3 sample.py --p_dir ./saved_predictor/00 --s_dir ./saved_selector/00 --data_dir tacred_20 --out ./tacred_20/extra_predictor.json --num 8000 --mode p

python3 update_predictor.py --data_dir tacred_20 --dataset extra_predictor.json --save_dir ./saved_predictor --id 00 --seed 905 --log_step 100

# Sample from predictor to update selector

python3 sample.py --p_dir ./saved_predictor/00 --s_dir ./saved_selector/00 --data_dir tacred_20 --out ./tacred_20/extra_selector.json --num 8000 --mode s

python3 update_selector.py --data_dir tacred_20 --dataset extra_selector.json --save_dir ./saved_selector --id 00 --seed 905 --log_step 100



