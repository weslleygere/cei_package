import argparse, logging, os, sys
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np 
import torch 

from msclap import CLAP 
from re import split
from unidecode import unidecode

# ----------------------------------------------------------------------------
def main(argv):
    description = "CLAP - soundscape-text similarities: python compute_similarities.py [audio_emb_folder] --classes [classes] --out [out_folder]"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("audio_emb_folder")
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

    args["audio_emb_folder"] = parse_args.audio_emb_folder 
    args["classes"] = parse_args.classes
    args["out"] = parse_args.out
    return args

# ----------------------------------------------------------------------------
if __name__ == '__main__':
    # Collect parameters
    args = main(sys.argv[1:])
    out_classes = split('/', args["classes"][:-4])[-1]
    out = os.path.join(args["out"], "sim_"+out_classes)

    # Output folder management
    if not os.path.exists(out):
        os.system(f"mkdir {out}")
    
    # Audio embeddings management  
    emb_path = args['audio_emb_folder']
    assert os.path.exists(emb_path)
    emb_list = [ os.path.join(emb_path, emb) 
                 for emb in os.listdir(emb_path) 
                 if emb.endswith(".npz") ]
    emb_list.sort() 

    # Text data management
    if not os.path.exists(args["classes"]):
        print("Please provide a correct set of classes")
    classes = open(args["classes"], "r").read()
    classes = [ unidecode(split(' - ', c.lower())[-1]) # only the finest sub-class
                for c in split('\n', classes) if len(c) ]

    # CLAP model 
    clap_model = CLAP(version=args["clap_version"], use_cuda=args["clap_cuda"])
    
    # Extract text embeddings
    text_embeddings = clap_model.get_text_embeddings(classes)
    
    # Iterate over all embedding batches 
    for emb in emb_list:
        # Extract audio embeddings
        audio_embeddings = torch.from_numpy(np.load(emb)['array'])

        # Compute similarity between audio and text embeddings 
        similarities = clap_model.compute_similarity(audio_embeddings, 
                                                     text_embeddings)
        similarities = similarities.detach().numpy()

        # Save the batch similarities
        np.savez_compressed(emb.replace(emb_path[:-1], out), 
                            array=similarities)
