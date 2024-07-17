import argparse, logging, os, sys
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np 
import librosa 
import soundfile as sf

from msclap import CLAP 
from re import split

# ----------------------------------------------------------------------------
def progress_bar(current_step, total_steps):
    assert total_steps>=current_step
    bar_length = 50
    progress = current_step / total_steps
    block = int(round(bar_length * progress))
    bar = '|' + '█' * block + ' ' * (bar_length - block) + '|'
    percentage = round(progress * 100, 2)
    print(f'\r{bar} {percentage}% ({current_step}/{total_steps}) Complete ', 
          end='', flush=True)

# ----------------------------------------------------------------------------
def main(argv):
    description = "CLAP - soundscape audio embeddings: python compute_audio_emb.py [audio_folder] --out [out_folder]"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("audio_folder")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--config", default="src/config.txt")
    parse_args = parser.parse_args()
    args = {} 
    
    conv = { 'int': int, 
             'float': float, 
             'bool': lambda x: x=='True', 
             'str': str }
             
    assert os.path.exists(parse_args.config)
    with open(parse_args.config,'r') as config:
        lines=config.readlines() 
        for line in lines:
            if len(line) and line!='\n' and line[0]!='#':
                while "\n" in line:
                    line = line.replace("\n",'')
                p_name, p_value, p_type = split(',', line)
                args[p_name] = conv[p_type](p_value)

    args["audio_folder"] = parse_args.audio_folder
    args["out"] = parse_args.out
    return args 

# ----------------------------------------------------------------------------
if __name__ == '__main__':
    # Collect parameters
    args = main(sys.argv[1:])

    # Output folder management
    out = os.path.join(args["out"], "audio_emb")
    if not os.path.exists(out):
        if not os.path.exists(args["out"]):
            os.system("mkdir "+args["out"])
        os.system(f"mkdir {out}")

    # Temporary folder
    tmp_audio = args["tmp_audio"]
    if os.path.exists(tmp_audio):
        os.system(f"rm -r {tmp_audio}")
    os.system(f"mkdir {tmp_audio}")
    
    # Audio data management  
    emb_dur = args["duration"]
    audio_path = args["audio_folder"]
    assert os.path.exists(audio_path)
    audio_list = [ os.path.join(audio_path, audio) for audio in os.listdir(audio_path)]
    audio_list.sort()
    assert len(audio_list)

    # CLAP model 
    clap_model = CLAP(version=args["clap_version"], use_cuda=args["clap_cuda"])

    # Iterate over all audios per batch
    N_BATCHES=len(audio_list)//args["batch_size"] 
    for i in range(1+N_BATCHES):
        # Retrieve unprocessed audios
        progress_bar(1+i, 1+N_BATCHES)
        if os.path.exists(f"{out}/batch_{i}.npz"):
            continue 
        
        if i==N_BATCHES:
            audios_i = audio_list[i*args["batch_size"]:]
        else:
            audios_i = audio_list[i*args["batch_size"]:(i+1)*args["batch_size"]]

        # Excerpts of short duration
        for audio in audios_i:
            audio_name = split('/', audio)[-1][:-4]
            try:
                y, sr = librosa.load(audio, sr=None)
            except Exception as e:
                print(f"An exception {e} occurred for the audio: {audio}. Ignored.")
                continue 
            n_emb = round((len(y)/sr)/emb_dur)
            for j in range(n_emb):
                # Temporary short audio files
                end_j = min(len(y), int(((j+1)*emb_dur)*sr))
                y_j = y[int((j*emb_dur)*sr):end_j]
                sf.write(f"{tmp_audio}{audio_name}_{j+1}_{n_emb}.wav", y_j, sr)

                # Keep track of the sub-divisions per file
                with open(f"{out}/batch_{i}.txt", "a") as f:
                    f.write(audio+'\n')

        the_audios_i = [ tmp_audio+f for f in os.listdir(tmp_audio) ]
        the_audios_i.sort()
        if not len(the_audios_i):
            print("Empty audio batch. Ignored.")
            continue

        # Extract audio embeddings
        audio_embeddings = clap_model.get_audio_embeddings(the_audios_i)

        # Delete all small audio excerpts
        os.system(f"rm {tmp_audio}*")

        # Save the batch embeddings
        np.savez_compressed(f"{out}/batch_{i}.npz",
                            array=audio_embeddings)

    os.system(f"rm -r {tmp_audio}")
