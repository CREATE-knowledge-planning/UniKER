# UniKER
<br>
We proposed UniKER to combine logical rule and KG embedding to conduct reasoning over KG.

## Quick Start
python kge/run.py --do_train --do_valid --do_test --model TransE --data_path ./data/kinship/ -b 1024 -n 256 -d 100 -g 24 -a 1 -adv -lr 0.001 --max_steps 5000 --test_batch_size 16 -save ./models/TransE --train_path ./data/kinship/train.txt
