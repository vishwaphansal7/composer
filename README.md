# Composer
It creates music using midi datasets dividing it into  dimensions and then using encoder and decoders Neural networks

# How to install
Install dependencies in python3 by running pip install -r requirements.txt.

# How to run

1) Find some dataset to train on.

2) Run preprocess_songs.py. This will load all midi files from your midi files folder data/raw/ into data/interim/samples.npy & lengths.npy. You can point the script to a location using the --data_folder flag one or multiple times to make it include more paths to index midi files.

3) Run train.py. This will train your network and store the progress from time to time (EPOCHS_TO_SAVE to change storing frequency). Only stops if you interrupt it or after 2000 epochs.

4) Run composer.py --model_path "e300" where "e300" indicates the folder the stored model is in. Look into the results/history folder to find your trained model folders.
