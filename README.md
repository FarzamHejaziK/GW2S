# GW2S

Code for exploring generalization in deep learning for mmWave beam selection using sub-6 GHz channels.

This repository contains MATLAB experiment scripts for training and evaluating a neural network that predicts mmWave beam indices from sub-6 GHz channel measurements. The workflow is designed around wireless communication experiments where sub-6 GHz channel state information is transformed into an angle-delay profile and used as the input to a beam-selection classifier.

## Research Focus

The experiment studies whether a deep learning model trained on sub-6 GHz channel features can generalize to mmWave beam-selection tasks under different SNR and transmit-power settings.

The pipeline measures:

- Top-1, top-3, and top-5 beam prediction accuracy
- Average achievable rate for predicted beams
- Upper-bound achievable rate from the best beam
- Performance across multiple transmit-power and SNR values

## Repository Structure

| File | Purpose |
| --- | --- |
| `main.m` | Main script for training, testing, and saving experiment results |
| `dataPrep_train.m` | Loads, normalizes, shuffles, and labels training data |
| `dataPrep_test.m` | Prepares test data for evaluation |
| `buildNetconv4.m` | Builds the neural network used for beam classification |
| `CSI2ADP_theta_N.m` | Converts channel state information to angle-delay profile features |
| `UPA_codebook_generator.m` | Generates the beamforming codebook |
| `LICENSE` | Repository license |

## Expected Data Layout

The main script expects `.mat` channel datasets in this structure:

```text
Data/
  train/
    sub6Train_org_4_64.mat
    mmTrain_org_64_64.mat
  test/
    sub6test_LOSB_4_64.mat
    mmtest_LOSB_4_64.mat
```

Each dataset should include a `channel` variable. For NLOS cases, labels may also be required.

## Requirements

- MATLAB
- Deep Learning Toolbox
- Communications Toolbox, for `awgn`
- Wireless channel datasets matching the expected dimensions in `main.m`

## Running

From the MATLAB console:

```matlab
main
```

Before running, update the paths and experiment settings in `main.m` if your dataset names or dimensions differ from the defaults.

## Key Settings

Important configuration values live near the top of `main.m`:

- `tx_power`: transmit-power sweep
- `snr_db`: SNR sweep
- `num_ant`: number of sub-6 GHz antennas
- `num_ant_mm`: number of mmWave antennas
- `options.case`: channel case, such as `LOS`
- `options.top_n`: number of top beams used for top-N evaluation

## Notes

This is a research codebase, so the scripts assume a prepared dataset and MATLAB environment. For a new experiment, first verify data dimensions, codebook size, and the `options.inputSize` settings.
