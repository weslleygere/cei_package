import argparse, logging, os, sys
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

from re import split
from unidecode import unidecode
    
# ----------------------------------------------------------------------------
def main(argv):
    description = "Contrastive Ecoacoustic Indices: python visualize_CEI.py --classes [classes] --out [out_folder]"
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
if __name__ == '__main__':
    # Collect parameters
    args = main(sys.argv[1:])
    class_names = split('/',args["classes"][:-4])[-1]
    out = os.path.join(args["out"], "sim_"+class_names)
    eps = args["eps"]
    facecolor = args['facecolor']
    grey_color = args['grey_color']
    axbackcolor = args['axbackcolor']
    assert os.path.exists(out)

    # Output folder management
    save = os.path.join(args["out"], "out_"+class_names)
    assert os.path.exists(save)
    
    # Text data management
    if not os.path.exists(args["classes"]):
        print("Please provide a correct set of classes")
    classes = open(args["classes"], "r").read()
    classes = [ unidecode(c.lower().replace(' - ', ' '))[:50]
                for c in split('\n', classes) if len(c) ]
    
    # Activation matrix
    df_am = pd.read_csv(f"{save}/activation_matrix.csv")
    
    plt.close('all')
    interp = 'nearest' 
    plt.figure(figsize=(10,6), facecolor=facecolor)
    plt.gca().xaxis.label.set_color(grey_color)
    plt.gca().yaxis.label.set_color(grey_color)
    plt.gca().tick_params(axis='x', colors=grey_color)
    plt.gca().tick_params(axis='y', colors=grey_color)
    plt.gca().title.set_color(grey_color)
    plt.gca().spines['top'].set_color(grey_color)
    plt.gca().spines['bottom'].set_color(grey_color)
    plt.gca().spines['left'].set_color(grey_color)
    plt.gca().spines['right'].set_color(grey_color)
    plt.imshow(df_am.T.to_numpy()[1:].astype(float), cmap='magma', aspect='auto', interpolation=interp)
    cbar = plt.colorbar()
    cbar.ax.yaxis.set_ticklabels([str(v/10.) for v in range(0,11,2) ], color=grey_color)
    cbar.ax.yaxis.set_tick_params(color=grey_color)
    plt.clim([0,1])
    plt.yticks(np.arange(len(classes)), classes, ha='right', fontsize=4)
    plt.xticks([]) 
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('Classes', fontsize=10)
    plt.title(f'Soundscape - raw contrastive activations', fontsize=10)
    plt.savefig(f"{save}/raw_activations.png", dpi=200, bbox_inches='tight')
    plt.close('all')
    
    # Contrastive curves
    df = pd.read_csv(f"{save}/curves.csv")

    plt.close('all')
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), facecolor=facecolor)
    raw_colors = ['green', 'blue', 'orange', 'red']
    colors = ["#66CC66", "#88B4E0", "#FFAA66", "#FF6666"]
    values = [ np.array(df[col]) for col in df.columns[1:] ]
    ceis = [ np.median(v) for v in values ]
    titles = ['Biophony', 'Geophony', 'Anthropophony', 'Technophony']
    for i in range(len(axes)):
        x = np.arange(values[i].shape[0])
        axes[i].plot(x, values[i], color=colors[i])
        ci = np.std(values[i]) #1.96*np.std(values[i])/np.sqrt(len(x))
        axes[i].fill_between(x, (values[i]-ci), (values[i]+ci), color=colors[i], alpha=.1)
        axes[i].spines['top'].set_color(grey_color)
        axes[i].spines['bottom'].set_color(grey_color)
        axes[i].spines['left'].set_color(grey_color)
        axes[i].spines['right'].set_color(grey_color)
        axes[i].set_facecolor(axbackcolor)
        axes[i].set_title(titles[i], color=colors[i], fontsize=10)
        axes[i].tick_params(axis='x', colors=grey_color)
        axes[i].tick_params(axis='y', colors=grey_color)
        axes[i].xaxis.set_visible(False)
        axes[i].set_ylim(-0.05, 1.05)
        axes[i].axhline(y=0.5, color=colors[i], linestyle=':', alpha=0.8)
        axes[i].axhline(y=ceis[i], color=raw_colors[i], linestyle='--', alpha=0.7)
    fig.text(0.5, 0.025, "Time", ha='center', color=grey_color, fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(f'Soundscape contrastive curve \n [%] bio={np.round(ceis[0], 2)} geo={np.round(ceis[1], 2)} ant={np.round(ceis[2], 2)} tec={np.round(ceis[3], 2)}', color=grey_color, fontsize=10)
    plt.savefig(f"{save}/contrastive_curves.png", dpi=200, bbox_inches='tight')
    plt.close('all')
