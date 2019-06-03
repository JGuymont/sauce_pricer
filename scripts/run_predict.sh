#!/usr/bin/env bash

# # PREDICT PRICES WITH ONLY NUMERICAL FEATURES

python -m sauce_pricer predict --model_definition config/regressor.json \
                                --model_state_path trained_models/model_state_mlp_regressor.pickle \
                                --data_path data/train_data.pkl

python -m sauce_pricer predict --model_definition config/regressor.json \
                                --model_state_path trained_models/model_state_mlp_regressor.pickle \
                                --data_path data/valid_data.pkl

# PREDICT MISSING VALUES

# python -m sauce_pricer predict --model_definition config/missing_imputer.json \
#                                --model_state_path trained_models/model_state_autoencoder.pickle \
#                                --data_path data/train_data_missing.pkl