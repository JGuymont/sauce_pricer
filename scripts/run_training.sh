#!/usr/bin/env bash

python -m sauce_pricer train --model_definition config/regressor.json
python -m sauce_pricer train --model_definition config/regressor_mix_features.json
python -m sauce_pricer train --model_definition config/autoencoder.json