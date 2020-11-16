# UniKER
<br>
We proposed UniKER to combine logical rule and KG embedding to conduct reasoning over KG.

## Quick Start
python run.py DATASET CUDA SAVE_MODEL_NAME BASIC_KGE_MODEL INTER NOISE_THRESHOLD TOP_K_THRESHOLD IS_INIT

e.g., python run.py create 2 create_model TransE 2 0.0 0.2 0
