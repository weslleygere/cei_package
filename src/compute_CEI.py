import argparse, logging, os, sys
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np 
import pandas as pd 

from re import split, match
from unidecode import unidecode

# ----------------------------------------------------------------------------
def main(argv):
    description = "Contrastive Ecoacoustic Indices: python compute_CEI.py --classes [classes] --out [out_folder]"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--classes", default="classes/classes.txt")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--config", default="config.txt")
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

    args["classes"] = parse_args.classes
    args["out"] = parse_args.out
    return args 

# ----------------------------------------------------------------------------
def sort_by_index(file_name):
    # Extract the index from the file name
    index = int(match(r'.*?(\d+)\.npz', file_name).group(1))
    return index

# ----------------------------------------------------------------------------
if __name__ == '__main__':
    # Collect parameters
    args = main(sys.argv[1:])
    class_names = split('/',args["classes"][:-4])[-1]
    out = os.path.join(args["out"], "sim_"+class_names)
    eps = args["eps"]
    assert os.path.exists(out)

    # Output folder management
    save = os.path.join(args["out"], "out_"+class_names)
    if not os.path.exists(save):
        os.system(f"mkdir {save}")

    # Iterate over all similarity batches
    npz_path = [ os.path.join(out, npz) for npz in os.listdir(out) ]
    assert len(npz_path)
    for i in range(len(npz_path)):
        data_i = np.load(npz_path[i])['array']
        sim_matrix = data_i if i==0 else np.concatenate([sim_matrix, data_i], axis=0)

    # Text data management
    if not os.path.exists(args["classes"]):
        print("Please provide a correct set of classes")
    classes = open(args["classes"], "r").read()
    classes = [ unidecode(c.lower().replace(' - ', ' '))
                for c in split('\n', classes) if len(c) ]

    # Global categories
    biophony = np.array([ ("biophony" in c) for c in classes ])
    geophony = np.array([ ("geophony" in c) for c in classes ])
    antphony = np.array([ ("anthropophony" in c) for c in classes ])
    tecphony = np.array([ ("technophony" in c) for c in classes ])

    # List of audio files
    site_audios = []
    audio_emb_dir = os.path.join(args["out"], "audio_emb")
    list_audios = [ os.path.join(audio_emb_dir,d) for d in os.listdir(audio_emb_dir) if d.endswith('.txt') ]
    for l_audios in list_audios:
        site_audios += split('\n',open(l_audios,'r').read())
    site_audios = [ aud for aud in site_audios if len(aud) ]
    
    # Activation matrix
    activation_matrix = (sim_matrix>args["thr_sim"])*sim_matrix
    activation_matrix = np.maximum([eps], np.minimum([1.], activation_matrix/args["max_sim"]))
    assert activation_matrix.shape[0]==len(site_audios)
    df_am = pd.DataFrame(activation_matrix)
    df_am.columns = classes
    df_am.index = site_audios
    df_am.to_csv(f"{save}/activation_matrix.csv")

    #  Per-time activation of the different classes
    alphas = activation_matrix > args["thr_activity"]
    alphas_b = alphas[:,biophony].sum(axis=-1); alphas_b = alphas_b/(eps+alphas_b.max())
    alphas_g = alphas[:,geophony].sum(axis=-1); alphas_g = alphas_g/(eps+alphas_g.max())
    alphas_a = alphas[:,antphony].sum(axis=-1); alphas_a = alphas_a/(eps+alphas_a.max())
    alphas_t = alphas[:,tecphony].sum(axis=-1); alphas_t = alphas_t/(eps+alphas_t.max())

    # High-level temporal curves per categories
    bio_m = activation_matrix[:,biophony].mean(axis=-1)
    geo_m = activation_matrix[:,geophony].mean(axis=-1)
    ant_m = activation_matrix[:,antphony].mean(axis=-1)
    tec_m = activation_matrix[:,tecphony].mean(axis=-1)
    bio = np.ma.masked_array(activation_matrix[:,biophony], mask=~alphas[:,biophony]).mean(axis=-1).filled(0.)
    geo = np.ma.masked_array(activation_matrix[:,geophony], mask=~alphas[:,geophony]).mean(axis=-1).filled(0.)
    ant = np.ma.masked_array(activation_matrix[:,antphony], mask=~alphas[:,antphony]).mean(axis=-1).filled(0.)
    tec = np.ma.masked_array(activation_matrix[:,tecphony], mask=~alphas[:,tecphony]).mean(axis=-1).filled(0.)

    # Temporal curves per categories (for display only)
    bio = np.maximum(bio_m, np.minimum([1.], (1+alphas_b)*bio))
    geo = np.maximum(geo_m, np.minimum([1.], (1+alphas_g)*geo))
    ant = np.maximum(ant_m, np.minimum([1.], (1+alphas_a)*ant))
    tec = np.maximum(tec_m, np.minimum([1.], (1+alphas_t)*tec))
    
    # Data frames
    df = pd.DataFrame([site_audios, bio, geo, ant, tec]).T
    df.columns = ['filenames', 'biophony', 'geophony', 'anthropophony', 'technophony']
    df = df.groupby('filenames').mean() 
    df.to_csv(f"{save}/curves.csv")
    
    # Contrastive Ecoacoustic Indices (CEI)
    cei = df.median()
    cei.to_csv(f"{save}/CEI.csv", header=False)
    print(cei)
