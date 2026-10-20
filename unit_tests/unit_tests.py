# ----------------------------------------------------------------------------
try:
    import os
except Exception as e:
    raise Exception("/!\ os package could not be imported!")
try:
    import librosa 
except Exception as e:
    raise Exception("/!\ librosa package could not be imported!")
try:
    import numpy as np 
except Exception as e:
    raise Exception("/!\ numpy package could not be imported!")
try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise Exception("/!\ matplotlib.pyplot package could not be imported!")
try:
    import pandas as pd 
except Exception as e:
    raise Exception("/!\ pandas package could not be imported!")
try:
    import soundfile as sf
except Exception as e:
    raise Exception("/!\ soundfile package could not be imported!")
try:
    import torch 
except Exception as e:
    raise Exception("/!\ torch package could not be imported!")

try:
    from msclap import CLAP 
except Exception as e:
    raise Exception("/!\ CLAP package could not be imported!")
try:
    from re import split,match
except Exception as e:
    raise Exception("/!\ re package could not be imported!")
try:
    from unidecode import unidecode
except Exception as e:
    raise Exception("/!\ unidecode package could not be imported!")

# ----------------------------------------------------------------------------
audio_path = '../data/raw'
texts = '../txt/classes.txt'
config_file = "../src/config.txt"

if not os.path.exists(audio_path):
    raise Exception("/!\ Audio path does not exist!")
if not os.path.exists(texts):
    raise Exception("/!\ Text description file does not exist!")
if not os.path.exists(config_file):
    raise Exception("/!\ Configuration file does not exist!")

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
# Configuration hyperparameters
try:
    conv = { 'int': int, 
            'float': float, 
            'bool': lambda x: x=='True', 
            'str': str }
    args = {}
    with open(config_file,'r') as config:
        lines=config.readlines() 
        for line in lines:
            if len(line) and line!='\n' and line[0]!='#':
                while "\n" in line:
                    line = line.replace("\n",'')
                p_name, p_value, p_type = split(',', line)
                args[p_name] = conv[p_type](p_value)
except Exception as e:
    raise Exception("/!\ Could not load hyperparameter well (see config file).")

# ----------------------------------------------------------------------------
try:
    eps = args['eps']
    facecolor = args['facecolor']
    grey_color = args['grey_color']
    axbackcolor = args['axbackcolor']
except Exception as e:
    raise Exception("/!\ Could not read some hyperparameters (see config file).")

# ----------------------------------------------------------------------------
try:
    clap_model = CLAP(version=args["clap_version"], use_cuda=args["clap_cuda"])
except Exception as e:
    raise Exception("/!\ CLAP model could be loaded!")
assert clap_model

# ----------------------------------------------------------------------------
# Select the audio data
audio_list = [ os.path.join(audio_path, audio) for audio in os.listdir(audio_path)]
audio_list.sort()
if not len(audio_list):
    raise Exception("/!\ No audio files detected. Audio folder looks empty!")

# ----------------------------------------------------------------------------
# Temporary folder for short audio segments
try:
    tmp_audio = args["tmp_audio"]
except Exception as e:
    raise Exception("/!\ Argument tmp_audio not found in hyperparameters!")
if os.path.exists(tmp_audio):
    os.system(f"rm -r {tmp_audio}")
os.system(f"mkdir {tmp_audio}")

# ----------------------------------------------------------------------------
# Extract audio embeddings
try:
    sub_audios = []
    emb_dur = args["duration"]
    N_BATCHES=len(audio_list)//args["batch_size"] 
    audio_embeddings = None
except:
    raise Exception("/!\ Hyperparameters not found (duration, batch_size) - see config.")

# Iterate over all audios per batch
for i in range(1+N_BATCHES):
    # Retrieve unprocessed audios
    progress_bar(1+i, 1+N_BATCHES)
    
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
            try:
                sf.write(f"{tmp_audio}{audio_name}_{j+1}_{n_emb}.wav", y_j, sr)
            except Exception as e:
                print(f"An exception {e} occurred when writing sub-excerpt for audio: {audio}. Ignored.")
                continue 

            # Keep track of the sub-divisions per file
            sub_audios.append(audio)

    the_audios_i = [ tmp_audio+f for f in os.listdir(tmp_audio) ]
    the_audios_i.sort()
    if not len(the_audios_i):
        print("Empty audio batch. Ignored.")
        continue

    # Extract audio embeddings
    try:
        audio_emb_i = clap_model.get_audio_embeddings(the_audios_i)
    except Exception as e:
        raise Exception("/!\ CLAP cannot compute audio embeddings!")
    
    try:
        if audio_embeddings is None:
            audio_embeddings = audio_emb_i 
        else:
            audio_embeddings = torch.cat((audio_embeddings, audio_emb_i), dim=0)
    except Exception as e:
        raise Exception("/!\ Issue encountered when concatenating audio embeddings!")

    # Delete all small audio excerpts
    os.system(f"rm {tmp_audio}*")

if os.path.exists(tmp_audio):
    os.system(f"rm -r {tmp_audio}")

# ----------------------------------------------------------------------------
# Select text data
try:
    all_classes = open(texts, "r").read()
    raw_classes = [ unidecode(c.lower().replace(' - ', ' '))
                    for c in split('\n', all_classes) if len(c) ]
    classes = [ unidecode(split(' - ', c.lower())[-1])
                for c in split('\n', all_classes) if len(c) ]
except Exception as e:
    raise Exception("/!\ Could not read text descriptions!")

# ----------------------------------------------------------------------------
# Extract text embeddings
try:
    text_embeddings = clap_model.get_text_embeddings(classes)
except Exception as e:
    raise Exception("/!\ CLAP cannot compute text embeddings!")

# ----------------------------------------------------------------------------
# Compute similarity between audio and text embeddings 
try:
    similarities = clap_model.compute_similarity(audio_embeddings, 
                                                text_embeddings)
    similarities = similarities.detach().numpy()
except Exception as e:
    raise Exception("/!\ CLAP cannot compute audio-text similarities!")

# ----------------------------------------------------------------------------
# Global categories
try:
    biophony = np.array([ ("biophony" in c) for c in raw_classes ])
    geophony = np.array([ ("geophony" in c) for c in raw_classes ])
    antphony = np.array([ ("anthropophony" in c) for c in raw_classes ])
    tecphony = np.array([ ("technophony" in c) for c in raw_classes ])
except Exception as e:
    raise Exception(f"/!\ Exception {e} when aggregating classes into high-level categories!")

# ----------------------------------------------------------------------------
# Activation matrix
try:
    activation_matrix = (similarities>args["thr_sim"])*similarities
    activation_matrix = np.maximum([eps], np.minimum([1.], activation_matrix/args["max_sim"]))
except Exception as e:
    raise Exception(f"/!\ Could not derive activation matrix from audio-text similarities!")

if not activation_matrix.shape[0]==len(sub_audios):
    raise Exception("/!\ Shape of activation matrix does not match!")

try:
    df_am = pd.DataFrame(activation_matrix)
    df_am.columns = classes
    df_am.index = sub_audios
except Exception as e:
    raise Exception("/!\ Problem when defining pandas DataFrame from activation matrix!")

# ----------------------------------------------------------------------------
#  Per-time activation of the different classes
try:
    alphas = activation_matrix > args["thr_activity"]

    alphas_b = alphas[:,biophony].sum(axis=-1)
    alphas_b = alphas_b/(eps+alphas_b.max())

    alphas_g = alphas[:,geophony].sum(axis=-1)
    alphas_g = alphas_g/(eps+alphas_g.max())

    alphas_a = alphas[:,antphony].sum(axis=-1)
    alphas_a = alphas_a/(eps+alphas_a.max())

    alphas_t = alphas[:,tecphony].sum(axis=-1)
    alphas_t = alphas_t/(eps+alphas_t.max())
except Exception as e:
    raise Exception("/!\ Issue when defining temporal activation of the classes!")

# ----------------------------------------------------------------------------
# High-level temporal curves per categories
try:
    bio_m = activation_matrix[:,biophony].mean(axis=-1)
    geo_m = activation_matrix[:,geophony].mean(axis=-1)
    ant_m = activation_matrix[:,antphony].mean(axis=-1)
    tec_m = activation_matrix[:,tecphony].mean(axis=-1)
    bio = np.ma.masked_array(activation_matrix[:,biophony], mask=~alphas[:,biophony]).mean(axis=-1).filled(0.)
    geo = np.ma.masked_array(activation_matrix[:,geophony], mask=~alphas[:,geophony]).mean(axis=-1).filled(0.)
    ant = np.ma.masked_array(activation_matrix[:,antphony], mask=~alphas[:,antphony]).mean(axis=-1).filled(0.)
    tec = np.ma.masked_array(activation_matrix[:,tecphony], mask=~alphas[:,tecphony]).mean(axis=-1).filled(0.)
except Exception as e:
    raise Exception("/!\ Issue when aggregating the classes into high-level categories!")

# ----------------------------------------------------------------------------
# Temporal curves per categories (for display only)
try:
    bio = np.maximum(bio_m, np.minimum([1.], (1+alphas_b)*bio))
    geo = np.maximum(geo_m, np.minimum([1.], (1+alphas_g)*geo))
    ant = np.maximum(ant_m, np.minimum([1.], (1+alphas_a)*ant))
    tec = np.maximum(tec_m, np.minimum([1.], (1+alphas_t)*tec))
except Exception as e:
    raise Exception("/!\ Issue when finalizing categorical curves!")

# ----------------------------------------------------------------------------
# Data frames
try:
    df = pd.DataFrame([sub_audios, bio, geo, ant, tec]).T
    df.columns = ['filenames', 'biophony', 'geophony', 'anthropophony', 'technophony']
    df = df.groupby('filenames').mean() 
except Exception as e:
    raise Exception("/!\ Could not define DataFrame from categorical curves!")

# ----------------------------------------------------------------------------
# Contrastive Ecoacoustic Indices (CEI)
try:
    cei = df.median()
except Exception as e:
    raise Exception("/!\ Could not extract median, hence CEI!")

# ----------------------------------------------------------------------------
# Reproducibility: max 0.5% relative error 
tolerance = 0.005
cei_unit_test = "../notebook/cei_unit_test.csv"
if not os.path.exists(cei_unit_test):
    raise Exception("/!\ Reference CEI for unit test are missing!")
try:
    ref = pd.read_csv(cei_unit_test)
except Exception as e:
    raise Exception("/!\ Could not read reference CEI csv file!")

try:
    relative_error = cei.values-ref.to_numpy()[:,-1]
except Exception as e:
    raise Exception("/!\ Could not process the CEI unit test!")

if not np.all(relative_error <= tolerance):
    raise Exception("[UNIT TEST FAILED] Relative error tolerance failed!")

# ----------------------------------------------------------------------------
print("\n 🥳🥳🥳 Everything seems to work as excepted. You are ready to go. 🥳🥳🥳")
