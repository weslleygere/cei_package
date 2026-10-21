#!/usr/bin/env python3
"""Compare three 6-second CEI aggregation formulas on the 60 annotated clips.

Run from the repository root:
    python src/compare_cei_methods.py

Inputs (produced by select_cei_review.py):
    outputs/cei_review/annotations.csv
    outputs/cei_review/selection_key.csv
    outputs/cei_review/audio/clip_001.wav ... clip_060.wav
    txt/classes.txt

Requires numpy, pandas, matplotlib, scipy, msclap, torch, unidecode.
On first run CLAP is executed once and the raw similarity matrix is cached.
Subsequent runs reuse the cache. No existing pipeline files are overwritten.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from unidecode import unidecode

CATEGORIES = ("bio", "geo", "antro")
METHODS = ("original_60", "paper_eq7", "continuous_top2")
DISPLAY = {
    "original_60": "GitHub (60 clips)",
    "paper_eq7": "Equation 7 (+clip)",
    "continuous_top2": "Continuous + Top-2",
}
EPS = 1e-6


def load_annotation_data(review: Path):
    annotations = pd.read_csv(review / "annotations.csv")
    key = pd.read_csv(review / "selection_key.csv")
    required_ann = {"clip_id", "bio_0_3", "geo_0_3", "antro_0_3"}
    required_key = {"clip_id", "bio", "geo", "antro"}
    for name, df, required in (
        ("annotations.csv", annotations, required_ann),
        ("selection_key.csv", key, required_key),
    ):
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")
        if df.clip_id.isna().any() or df.clip_id.duplicated().any():
            raise ValueError(f"{name}: missing or duplicate clip_id")
    if set(annotations.clip_id) != set(key.clip_id):
        raise ValueError("Annotation clips and selection-key clips do not match")

    data = annotations.merge(key, on="clip_id", validate="one_to_one").sort_values("clip_id")
    data = data.reset_index(drop=True)
    for category in CATEGORIES:
        data[f"{category}_0_3"] = pd.to_numeric(data[f"{category}_0_3"], errors="raise")
        if not data[f"{category}_0_3"].isin([0, 1, 2, 3]).all():
            raise ValueError(f"Invalid human score for {category}; expected 0, 1, 2 or 3")
        data[category] = pd.to_numeric(data[category], errors="raise")
    return data


def load_classes(path: Path):
    classes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    prompts = [unidecode(line.lower().split(" - ")[-1]).strip() for line in classes]
    labels = [unidecode(line.lower()) for line in classes]
    masks = {
        "bio": np.array(["biophony" in s for s in labels], dtype=bool),
        "geo": np.array(["geophony" in s for s in labels], dtype=bool),
        "antro": np.array([("anthropophony" in s or "technophony" in s) for s in labels], dtype=bool),
    }
    for name, mask in masks.items():
        if mask.sum() < 2:
            raise ValueError(f"Category {name} needs at least two prompts for Top-2 pooling")
    return prompts, masks


def get_similarities(review, data, prompts, batch_size, cpu, recompute):
    """Return similarities aligned exactly to data.clip_id, caching CLAP outputs."""
    cache = review / "clap_similarities_60.npz"
    clip_ids = data["clip_id"].astype(str).tolist()
    if cache.exists() and not recompute:
        with np.load(cache, allow_pickle=False) as saved:
            stored_ids = saved["clip_ids"].astype(str).tolist()
            stored_prompts = saved["prompts"].astype(str).tolist()
            scores = saved["similarities"]
        if stored_ids != clip_ids or stored_prompts != prompts:
            raise ValueError(
                "Cached CLAP similarities do not match the current clip list/prompts. "
                "Use --recompute to regenerate them."
            )
        if scores.shape != (len(clip_ids), len(prompts)):
            raise ValueError(f"Invalid cached score shape {scores.shape}")
        print(f"Reusing CLAP scores: {cache}")
        return scores

    wav_paths = [review / "audio" / f"{clip_id}.wav" for clip_id in clip_ids]
    missing = [p for p in wav_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Could not find {len(missing)} audio clips, e.g. {missing[0]}. "
            "Place WAVs in outputs/cei_review/audio or use a valid saved cache."
        )

    import torch
    from msclap import CLAP

    use_cuda = (not cpu) and torch.cuda.is_available()
    print(f"Computing CLAP similarities once (CUDA: {use_cuda})")
    model = CLAP(version="2023", use_cuda=use_cuda)
    text_embeddings = model.get_text_embeddings(prompts)
    pieces = []
    for start in range(0, len(wav_paths), batch_size):
        chunk = wav_paths[start:start + batch_size]
        audio_embeddings = model.get_audio_embeddings([str(p) for p in chunk])
        if isinstance(audio_embeddings, torch.Tensor):
            audio_embeddings = audio_embeddings.detach().to(text_embeddings.device)
        else:
            audio_embeddings = torch.as_tensor(
                np.asarray(audio_embeddings), device=text_embeddings.device
            )
        values = model.compute_similarity(audio_embeddings, text_embeddings)
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        values = np.asarray(values)
        if values.shape != (len(chunk), len(prompts)):
            raise ValueError(f"CLAP returned unexpected shape {values.shape}")
        pieces.append(values)
        print(f"  {min(start + len(chunk),len(wav_paths))}/{len(wav_paths)} clips")
    scores = np.concatenate(pieces, axis=0)
    if not np.isfinite(scores).all():
        raise ValueError("CLAP scores contain NaN or infinity")
    np.savez_compressed(
        cache,
        similarities=scores,
        clip_ids=np.asarray(clip_ids),
        prompts=np.asarray(prompts),
    )
    print(f"Cached raw CLAP scores at {cache}")
    return scores


def compute_methods(sim, masks):
    """Reproduce original Github formula, paper Equation 7, and continuous alternative.

    For methods A and B, abundance alpha is normalized by the maximum
    active-class count across THESE 60 clips, not the full original 10k.
    Equation 7 is capped to [0,1], as explicitly reported in output labels.
    """
    original_activation = np.clip(
        np.where(sim > 5.0, sim, 0.0) / 20.0, EPS, 1.0
    )
    activity = original_activation > 0.1
    continuous_activation = np.clip((sim - 5.0) / 15.0, 0.0, 1.0)

    outputs = {}
    for category, mask in masks.items():
        x = original_activation[:, mask]
        valid = activity[:, mask]
        n_active = valid.sum(axis=1)
        alpha = n_active / (EPS + n_active.max())
        avg = x.mean(axis=1)
        active_avg = np.divide(
            (x * valid).sum(axis=1), n_active,
            out=np.zeros(len(n_active), dtype=float), where=n_active > 0,
        )
        # A: exactly as in the released compute_CEI.py / our generate script.
        outputs[("original_60", category)] = np.maximum(
            avg, np.minimum(1.0, (1.0 + alpha) * active_avg)
        )
        # B: paper's Equation 7, clipped at 1 for a bounded comparison.
        outputs[("paper_eq7", category)] = np.clip(
            (1.0 + alpha) * avg, 0.0, 1.0
        )
        # C: threshold-shifted continuous normalization plus Top-2 pooling.
        # 0.5*max + 0.5*mean(top2) = 0.75*top1 + 0.25*top2.
        top2 = np.sort(continuous_activation[:, mask], axis=1)[:, -2:]
        outputs[("continuous_top2", category)] = (
            0.75 * top2[:, -1] + 0.25 * top2[:, -2]
        )
    return outputs


def auc_roc(y_positive, scores):
    """ROC-AUC from average ranks (ties properly handled); no sklearn needed."""
    pos = y_positive.astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(scores, method="average")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def summarize(data, outputs):
    rows = []
    for category in CATEGORIES:
        human = data[f"{category}_0_3"].to_numpy(dtype=float)
        positive = human > 0
        for method in (*METHODS, "original_10k_reference"):
            # The 10k reference is the actual precomputed teacher label in selection_key.csv;
            # it helps distinguish changing the formula from changing the alpha denominator.
            values = (data[category].to_numpy(dtype=float) if method == "original_10k_reference"
                      else np.asarray(outputs[(method, category)], dtype=float))
            rho = spearmanr(human, values).statistic if (np.unique(values).size > 1 and np.unique(human).size > 1) else np.nan
            rows.append({
                "category": category, "method": method, "n": len(human),
                "n_present": int(positive.sum()), "spearman_rho": rho,
                "roc_auc_presence": auc_roc(positive, values),
                "pct_zero": 100 * float(np.mean(values <= 1e-5)),
                "pct_one": 100 * float(np.mean(values >= 1 - 1e-6)),
                "median_index": float(np.median(values)),
                "iqr_index": float(np.quantile(values, .75) - np.quantile(values, .25)),
            })
    return pd.DataFrame(rows)


def plot_distributions(data, outputs, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    bins = np.linspace(0, 1, 26)
    for ax, category in zip(axes, CATEGORIES):
        for method in METHODS:
            ax.hist(outputs[(method, category)], bins=bins, alpha=0.48, label=DISPLAY[method])
        ax.set(title=category.upper(), xlabel="Index (0–1)", xlim=(0, 1))
        ax.grid(axis="y", alpha=.18)
    axes[0].set_ylabel("Number of 6-second clips")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("CEI formula comparison — 60 annotated clips", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "distributions_comparison.png", dpi=180)
    plt.close(fig)


def plot_agreement(data, outputs, out):
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    shifts = [-0.16, 0, .16]
    for ax, category in zip(axes, CATEGORIES):
        annotation = data[f"{category}_0_3"].to_numpy(dtype=float)
        for method, shift in zip(METHODS, shifts):
            x = annotation + shift + rng.normal(0, .018, size=len(annotation))
            ax.scatter(x, outputs[(method, category)], alpha=.52, s=23, label=DISPLAY[method])
        ax.set(title=category.upper(), xlabel="Human annotation (0 absent – 3 intense)",
               xticks=[0, 1, 2, 3], xlim=(-.36, 3.36), ylim=(-.03, 1.04))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Computed index")
    axes[2].legend(loc="lower right", fontsize=8)
    fig.suptitle("Human annotation versus computed CEI", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "agreement_comparison.png", dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--review", default="outputs/cei_review",
                    help="Folder with audio/, annotations.csv, selection_key.csv")
    ap.add_argument("--classes", default="txt/classes.txt")
    ap.add_argument("--out", default=None, help="Output folder (default: <review>/comparison)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cpu", action="store_true", help="Force CLAP to run on CPU")
    ap.add_argument("--recompute", action="store_true", help="Ignore and overwrite saved similarity cache")
    args = ap.parse_args()
    if args.batch_size < 1:
        ap.error("--batch-size must be >= 1")
    review = Path(args.review)
    out = Path(args.out) if args.out else review / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    data = load_annotation_data(review)
    prompts, masks = load_classes(Path(args.classes))
    sim = get_similarities(review, data, prompts, args.batch_size, args.cpu, args.recompute)
    outputs = compute_methods(sim, masks)
    for category in CATEGORIES:
        for method in METHODS:
            data[f"{method}_{category}"] = outputs[(method, category)]
    metrics = summarize(data, outputs)
    data.to_csv(out / "comparison_by_clip.csv", index=False)
    metrics.to_csv(out / "metrics.csv", index=False)
    plot_distributions(data, outputs, out)
    plot_agreement(data, outputs, out)
    print("\nMetrics (descriptive: intentionally non-random, 60-clip evaluation sample):")
    print(metrics[["category", "method", "spearman_rho", "roc_auc_presence", "pct_zero", "pct_one"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nNOTE: original_60 recomputes the Github formula using the max count among these 60 clips.")
    print("The 'bio', 'geo', 'antro' columns in selection_key.csv were normalized over the original ~10,000 segments.")
    for category in CATEGORIES:
        d = np.abs(data[f"original_60_{category}"] - data[category])
        print(f"  {category}: original_60 vs saved full-dataset label: median abs diff={d.median():.4f}; max abs diff={d.max():.4f}")
    print("\nEquation 7 output is explicitly clipped at 1; continuous Top-2 uses fixed weights (0.75,0.25).")
    print(f"\nSaved results under {out}/")


if __name__ == "__main__":
    main()
