#!/usr/bin/env bash

# Visualize losses
python -m option_pricer visualize --mse_loss \
                                  --writer_path results/writer_mlp_regressor.pickle \
                                  --output_filename mlp_regressor_loss.png \
                                  --output_dir results

# visualize predictions for MLP
python -m option_pricer visualize --price_prediction \
                                  --target_feature PX_LAST \
                                  --output_dir results \
                                  --target_data data/train_data.pkl \
                                  --pred results/predictions_mlp_regressor_train_data.pickle \
                                  --output_filename mlp_regressor_pred_train.png

python -m option_pricer visualize --price_prediction \
                                  --target_feature PX_LAST \
                                  --output_dir results \
                                  --target_data data/valid_data.pkl \
                                  --pred results/predictions_mlp_regressor_valid_data.pickle \
                                  --output_filename mlp_regressor_pred_valid.png
