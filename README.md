# Contrastive Ecoacoustic Indices (CEI)

Welcome on the GitHub page of the `cei_package` developed by the [EAR](https://ear.cnrs.fr/) team!

## Introduction

This project introduces novel acoustic indices derived from contrastive inference with [CLAP](https://github.com/microsoft/CLAP),
which we call Contrastive Ecoacoutic Indices (CEI), that allow to globally characterize 
large amount of collected soundscape data into four categorical constituents:
* __Biophony__, *i.e.*, sounds produced by living organisms (excluding humans) in the environment;
* __Geophony__, *i.e.*, sounds produced by all abiotic elements in the environment;
* __Anthropophony__, *i.e.*, sounds produced directly by humans via their phonation or body;
* __Technophony__, *i.e.*, sounds produced by all machines and technologies from human activity.

This GitHub provides the complete supporting code to extract audio embeddings, compute 
audio-text similarities and obtain the CEI. Results can be saved into `csv` files and plotted as `png` figures. 

A complete executable `run_cei.sh`, encompassing all above-mentioned steps, as well as a Jupyter notebook, showcasing an example of usage, are also included.

## Quote / Citation 

If you find this work useful and use it in your research, industrial projects, practical applications, etc. please cite the following paper:

``` 
Soon.
``` 

and take the time to acknowledge this github page.

## Installation

`cei_package` has been developed with `Python==3.11.0`.

`cei_package` relies on various classical Python packages such as [`numpy`](http://www.numpy.org/), 
[`soundfile`](https://pysoundfile.readthedocs.io/), etc. and [CLAP](https://github.com/microsoft/CLAP).
These are listed in the `requirements.txt` file and are all easily installable via, *e.g.*, `pip install`. 
Any reasonably recent version of these packages should work. 

For convenience, we recommend creating a new `conda` environment (with chosen `<my-env>`) by running the 
next commands in a Terminal:
```
conda create --name <my-env> python==3.11.0
conda activate <my-env>
``` 

and then installing all necessary packages in `<my-env>` as follows:

```
pip install -r requirements.txt
```

## Unit tests 

To check that everything is correctly installed and should work fine, please run:
```
cd unit_tests
python unit_tests.py
```

If the Terminal output does not throw any errors and ends with
```
🥳🥳🥳 Everything seems to work as excepted. You are ready to go. 🥳🥳🥳
```
then you are ready to proceed.

## Usage

`cei_package` can be used in two ways:

* __As a whole executable__ taking as inputs (1) a folder of (relevantly time-ordered) audios, 
(2) a list of soundscape text descriptions and (3) an output folder, *i.e.*,

```
./run_cei.sh <audio-folder> <text-file> <output-folder> 
```

Toy example:
```
./run_cei.sh data txt/classes.txt out 
```

* __As part of your own code__: explore the `notebook` to see how each processing step can be integrated into a pipeline, and how the different functions can be independently used.

## Codes

The `cei_package` code layout is as follows.

- `data` folder contains 96 1-min audio files for examples and unit tests. These audios 
feature  a whole day of soundscape recordings in the Risoux Forest, Jura, France.

- `notebook` folder contains a notebook showing how to calculate the CEI and integrate the various steps in your own pipeline with your own sound data.

- `src` folder contains source code, notably:
    * `compute_audio_emb.py` - extract of the audio embeddings by CLAP and save them as `npz`.
    * `compute_similarities.py` - compute CLAP similiarities between audio and classes and save them as `npz`.
    * `compute_CEI.py` - derive Contrastive Ecoacoustic Indices from audio-text similarities and save outputs as `csv`. 
    * `visualize_CEI.py` - plot figures from previous steps as `png`.
    * `config.txt` - key hyperparameters.

```
python compute_audio_emb.py <audio-folder> --out <output-folder>
python compute_similarities.py <audio-embedding-folder> --classes <text-file> --out <output-folder>
python compute_CEI.py --classes <text-file> --out <output-folder>
python visualize_CEI.py --classes <text-file> --out <output-folder>
```

- `txt` folder contains possible lists of elements of interest to investigate soundscapes.  

- `run_cei.sh`: an executable file encompassing the complete CEI computation pipeline.  

```
./run_cei.sh <audio-folder> <text-file> <output-folder> 
```

## I/O configuration 

### Configuration file

The configuration file found in `src/config.txt` lists important hyperparameters used 
in our CEI pipeline, two of which may require tuning according to your dataset of interest.

The `batch_size` parameter (default: 64) is the number of audio files read, processed and saved at the same time by the CLAP model. 
Depending on the typical duration of your audio recordings and your hardware, you may have sufficient memory to increase this parameter (for faster extraction of all audio embeddings) or you may lack sufficient available memory and thus would need to reduce this parameter.
By default, we have set `batch_size=64`. For information, on a recent machine (`Dell Inc. Precision 3581, 13th Gen Intel® Core™ i7-13700H × 20, 64.0 GiB memory`) with audio recordings of 1-min duration, a `batch_size` of `128` can have been used. 

The duration parameter (default: 6.0 seconds) represents the typical length into which longer audio files are segmented to compute embeddings locally. For instance, an audio file of 1 minute would be subdivided into 10 segments of 6 seconds each. This approach allows for more reliable application of CLAP, as local events that might be omitted when processing the whole audio can be detected by breaking it down into smaller, uniform segments.

You can specify another configuration file (*e.g.*, `<new-config>`) to all `src` python codes with the optional parameter `--config <new-config>`.

### Input: Audio recordings

We ask the user to specify a folder `<audio-folder>` containing raw audio files in `.wav` or `.mp3` format. 

Audio recordings are turned into audio embeddings by the [CLAP](https://github.com/microsoft/CLAP) model. Their processing thus follows the original CLAP pipeline (see [1]).

For information, in our Passive Acoustic Monitoring (PAM) framework, we collected `wav` audio data with 16-bit precision and 44.1kHz sampling rate (but 48kHz would naturally be possible).

### Input: Text descriptions

As CLAP allows knowing to what extent any text description describes well or badly the content of an audio, it offers a practical flexibility to investigate the presence/absence/recognition of various elements (as text prompts) within the recordings. 

By default, we have defined a list of events/classes of interest for soundscape analysis at `txt/classes.txt`. However, for many practical reasons, one could need to modify and/or remplace these classes to investigate other specific aspect of a soundscape. 

It is thus completely possible to define a new list of classes `<new-text-file>` and specify their usage to all `src` python codes with the optional parameter `--classes <new-text-file>`. 

As a reminder, the CEI are derived from the high-level aggregation of classes into four global categories: biophony, geophony, anthropophony, and technophony. To enable this aggregation, each text description (corresponding to one line in the `.txt` file) must include one of these four categories, regardless of case sensitivity. A separator ` - ` can be used to investigate a precise element while indicating its affiliation with broader group(s) (the broader group are excluded in this case). 

We encourage users to take inspiration from `txt/classes.txt` to know how to design such a file of text descriptions. For example, a valid file may look like:

```
Biophony - Domestic animals - Cats
Biophony - Non-domestic animal - Birds
Geophony - Water - Rain falling
Geophony - Wind 
Anthropophony - Hand clapping
Anthropophony - People talking
Technophony - Objects - Bells ringing
Technophony - Vehicles - Tractor, mower, lawnmower running, and other machines used for farming or gardening
Technophony - Vehicles - Airplanes
Technophony - Vehicles - Cars
```

Keep in mind that CLAP text embeddings have been trained to encode linguistic semantics so one can please with the text description provided, including for instance perceptual aspect or notion of distance (see [1]). 

__[IMPORTANT] Note, however, that changing our original proposal of classes modifies the whole pipeline and ultimately generated different CEI that are not identical to the ones we have defined and cannot be directly compared to ours.__ 

### Output: 

Running 
```
./run_cei.sh <audio-folder> <text-file> <output-folder> 
```
would result in the creation of three output folders with the following content:
* `<output-folder>/audio_emb/`: batches of audio embeddings saved in `npz` format (`numpy` compressed arrays) and the list of processed audio in each batch saved in `txt` format.

* `<output-folder>/sim_<text-file>/`: batches of audio-text similarities saved in `npz` format (`numpy` compressed arrays).

* `<output-folder>/out_<text-file>/`:
   - `activation_matrix.csv`: the raw audio-text activation matrix.
   - `curves.csv`: the temporal curves indicating biophony, geophony, anthropophony and technophony levels in the soundscape.
   - `CEI.csv`: the four CEI values.
   
   - `raw_activations.png`: plot of the `activation_matrix.csv`.
   - `contrastive_curves.png`: plot of the `curves.csv`.

## References

[1] [CLAP](https://github.com/microsoft/CLAP) - Elizalde, B., Deshmukh, S., Al Ismail, M., & Wang, H. (2023, June). Clap: learning audio concepts from natural language supervision. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 1-5). IEEE. [https://arxiv.org/pdf/2206.04769](https://arxiv.org/pdf/2206.04769) 

