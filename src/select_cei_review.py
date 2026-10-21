"""Seleciona 60 janelas de 6 s para anotação às cegas dos CEI.

Entrada: CEI do generate_cei_3class_6s.py e CSV de metadados da amostra.
Saída: WAVs de 6 s, annotations.csv sem CEI, e selection_key.csv com a seleção.

Execute na raiz do repositório; os WAVs originais devem estar acessíveis no servidor.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

QUOTAS = {
    "geo_saturada": 15,      # Geo ~= 1
    "geo_intermediaria": 15,  # 0.4 <= Geo <= 0.7
    "geo_zero": 10,          # Geo ~= 0
    "antro_alta": 10,        # Antro >= percentil 90 da amostra
    "controle": 10,          # Sorteio aleatório dos demais
}


def diversify(frame, rng):
    """Mistura aleatória e intercala locais para evitar concentração em um local."""
    if "site_name" not in frame or frame.site_name.isna().all():
        return frame.iloc[rng.permutation(len(frame))]
    buckets = []
    for _, group in frame.groupby("site_name", dropna=False):
        buckets.append(group.iloc[rng.permutation(len(group))].reset_index(drop=True))
    rng.shuffle(buckets)
    ordered = []
    for i in range(max(len(b) for b in buckets)):
        for group in buckets:
            if i < len(group):
                ordered.append(group.iloc[i])
    return pd.DataFrame(ordered).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="outputs/cei_3class_6s.csv")
    ap.add_argument("--sample", default="data/Caatinga_samples/amostra_1000_terrestres.csv")
    ap.add_argument("--out", default="outputs/cei_review")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manifest-only", action="store_true", help="Seleciona sem acessar os arquivos WAV")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cei = pd.read_csv(args.csv)
    required = {"audio_id", "segment_index", "start_s", "end_s", "bio", "geo", "antro"}
    missing = required - set(cei.columns)
    if missing:
        ap.error(f"Colunas ausentes no CSV de CEI: {sorted(missing)}")
    cei["audio_id"] = cei["audio_id"].astype(int)
    if cei.duplicated(["audio_id", "segment_index"]).any():
        ap.error("Há segmentos duplicados (audio_id, segment_index)")
    for col in ("bio", "geo", "antro"):
        cei[col] = pd.to_numeric(cei[col], errors="raise")
    if (cei["end_s"] - cei["start_s"]).sub(6.0).abs().max() > 1e-3:
        ap.error("Esperadas janelas de exatamente 6 segundos")

    meta = pd.read_csv(args.sample)
    if meta["audio_id"].duplicated().any():
        ap.error("Há audio_id duplicados na amostra original")
    meta["audio_id"] = meta["audio_id"].astype(int)
    meta = meta.drop(columns=["source_path"], errors="ignore")  # caminho do CEI prevalece
    df = cei.merge(meta, how="left", on="audio_id", validate="many_to_one")
    if "source_path" not in df.columns or df["source_path"].isna().any():
        ap.error("Coluna source_path ausente ou incompleta no CEI")
    # Não incluir resultados NaN ou fora do intervalo.
    for col in ("bio", "geo", "antro"):
        df = df.loc[df[col].between(0, 1, inclusive="both")].copy()
    if len(df) < sum(QUOTAS.values()):
        ap.error("Há menos de 60 janelas CEI válidas")

    p90 = df["antro"].quantile(.90)
    group_masks = {
        "geo_saturada": df["geo"] >= 1 - 1e-6,
        "geo_intermediaria": df["geo"].between(.4, .7),
        "geo_zero": df["geo"] <= 1e-4,
        "antro_alta": df["antro"] >= p90,
        "controle": pd.Series(True, index=df.index),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    audio_dir = out / "audio"
    if not args.manifest_only:
        audio_dir.mkdir(exist_ok=True)
    chosen = []
    used_audio = set()  # preferimos gravações independentes
    used_segment = set()
    failures = 0

    def save_segment(row, clip_path):
        src = Path(str(row.source_path))
        with sf.SoundFile(src) as wav:
            sr = wav.samplerate
            start_frame = round(float(row.start_s) * sr)
            length = round((float(row.end_s) - float(row.start_s)) * sr)
            if start_frame < 0 or start_frame + length > len(wav):
                raise ValueError("janela fora da duração do WAV")
            wav.seek(start_frame)
            frames = wav.read(length, dtype="float32", always_2d=True)
            if len(frames) != length:
                raise ValueError("leitura incompleta")
        mono = frames.mean(axis=1)  # mesmo downmix utilizado no pipeline CLAP
        sf.write(clip_path, mono, sr, subtype="PCM_16")

    for group, quota in QUOTAS.items():
        candidates = diversify(df.loc[group_masks[group]], rng)
        added = 0
        for row in candidates.itertuples(index=False):
            segkey = (int(row.audio_id), int(row.segment_index))
            if segkey in used_segment or row.audio_id in used_audio:
                continue
            clip_id = f"clip_{len(chosen)+1:03d}"
            filename = audio_dir / f"{clip_id}.wav"
            if not args.manifest_only:
                try:
                    save_segment(row, filename)
                except (OSError, ValueError, RuntimeError) as exc:
                    failures += 1
                    if failures <= 8:
                        print(f"WAV indisponível {row.audio_id}: {exc}")
                    continue
            chosen.append({**row._asdict(), "grupo_selecao": group, "clip_id": clip_id})
            used_segment.add(segkey)
            used_audio.add(row.audio_id)
            added += 1
            if added == quota:
                break
        if added < quota:
            raise RuntimeError(
                f"Grupo '{group}': só {added}/{quota} áudios disponíveis. "
                "Veja caminhos ou critérios de seleção. Nenhum CSV final foi gravado."
            )

    selected = pd.DataFrame(chosen)
    # Embaralhar nomes de clips para a sequência de audição não revelar o grupo.
    order = rng.permutation(len(selected))
    # Renomeamos WAVs de forma segura em duas fases.
    if not args.manifest_only:
        for old_idx, new_idx in enumerate(order):
            old = audio_dir / f"clip_{old_idx+1:03d}.wav"
            tmp = audio_dir / f"tmp_{new_idx+1:03d}.wav"
            old.rename(tmp)
        for i in range(len(selected)):
            (audio_dir / f"tmp_{i+1:03d}.wav").rename(audio_dir / f"clip_{i+1:03d}.wav")
    selected["clip_id"] = [f"clip_{i+1:03d}" for i in order]
    selected = selected.sort_values("clip_id").reset_index(drop=True)
    selected.to_csv(out / "selection_key.csv", index=False)
    annotation = pd.DataFrame({
        "clip_id": selected["clip_id"],
        "bio_0_3": [""] * len(selected),
        "geo_0_3": [""] * len(selected),
        "antro_0_3": [""] * len(selected),
        "observacoes": [""] * len(selected),
    })
    annotation.to_csv(out / "annotations.csv", index=False)
    print(f"Selecionados: {len(selected)} segmentos, {len(used_audio)} gravações distintas")
    print("Grupos:", selected["grupo_selecao"].value_counts().to_dict())
    if failures:
        print(f"Tentativas de leitura sem sucesso: {failures}")
    print(f"Planilha às cegas: {out / 'annotations.csv'}")
    print(f"Chave com labels CLAP (NÃO abrir durante anotação): {out / 'selection_key.csv'}")
    if args.manifest_only:
        print("--manifest-only: WAVs não foram extraídos")
    else:
        print(f"Áudios: {audio_dir}")


if __name__ == "__main__":
    main()
