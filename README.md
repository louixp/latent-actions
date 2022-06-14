this branch split from decoderWeightRegularizer.
it's goal is to add garbage value to context in random sequence to help the decoder use both context AND z vector to come up with final decision


To start up a simulation, run

```
python3 example.py --model_class cVAE --checkpoint_path [CHECKPOINT_PATH]
```

how to run training example
python train.py --decode --model_class cVAE --enc_dims 10 10 10 --dec_dims 10 10 10 --keep_success --size_limit 100000

how to run without wandb
python train.py --no_wandb --decode --model_class cVAE --enc_dims 10 10 10 --dec_dims 10 10 10 --keep_success --size_limit 100000 --max_epochs 20

python train.py --decode --model_class cVAE --enc_dims 10 10 10 --dec_dims 10 10 10 --keep_success --size_limit 100000 --max_epochs 20 --add_adversarial_pos

To exit a simulation, press back.
