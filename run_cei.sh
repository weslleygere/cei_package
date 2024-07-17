#!/bin/bash

python src/compute_audio_emb.py "$1" --out "$3"
python src/compute_similarities.py "$3"/audio_emb/ --classes "$2" --out "$3"
python src/compute_CEI.py --classes "$2" --out "$3"
python src/visualize_CEI.py --classes "$2" --out "$3"
