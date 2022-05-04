To start up a simulation, run

```
python3 example.py --model_class cVAE --checkpoint_path [CHECKPOINT_PATH]
```

how to run training example
python train.py --decode --model_class cVAE --enc_dims 10 10 10 --dec_dims 10 10 10 --keep_success --size_limit 100000

how to run without wandb
python train.py --no_wandb --decode --model_class cVAE --enc_dims 10 10 10 --dec_dims 10 10 10 --keep_success --size_limit 100000 --max_epochs 20


To exit a simulation, press back.
