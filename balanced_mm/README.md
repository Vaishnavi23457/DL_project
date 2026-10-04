# Balanced Multimodal Learning — OPM / OGM-GE (Text + Image)

Paper: **Wei, Hu, Du, Wen — "On-the-fly Modulation for Balanced Multimodal Learning" (IEEE TPAMI 2024)**
(+ CVPR 2022 "OGM-GE"). Ye repo us paper ka **ready-to-use PyTorch implementation** hai — Text + Image
late-fusion model ke saath, lekin core module (`bml/modulation.py`) **kisi bhi modality / encoder / M ≥ 2** ke liye kaam karta hai.

```
balanced_mm/
├── bml/
│   ├── modulation.py        ★ CORE: Eq 6/7 (discrepancy), OPM (Eq 8), OGM-GE (Eq 11/12/16/17), BalancedModulator
│   ├── models.py            ResNet-18 image encoder, Transformer/HF text encoder, VectorMLPEncoder (audio feats), LateFusionModel (block-wise logits)
│   ├── data.py              CSV dataset (image,text,label; audio column as .npy or packed audio.npz::key), tokenizer, synthetic imbalance dataset, collate
│   ├── engine.py            train_one_epoch / evaluate / paper-style linear probing
│   ├── xai.py               XAI detection: Integrated Gradients, grad×input tokens, modality attribution shares
│   ├── metrics.py           confusion matrix, per-class acc, macro-F1, calibration error (ECE)
│   ├── analysis.py          runs → results/ (mean±std tables + plots, auto report)
│   └── utils.py
├── train.py                 CLI training  (--modulation none|opm|ogm|both)
├── eval.py                  evaluate checkpoint (+ --probe)
├── compare.py               runs ka side-by-side table
├── plot_history.py          Fig-1 / Fig-5 style curves
├── plug_in_example.py       ★ existing training loop me 3 line me integrate karne ka example
├── notebooks/
│   └── 01_walkthrough.ipynb   executed demo: none vs OGM-GE + curves + XAI attribution (GitHub Preview me)
├── scripts/
│   ├── run_synthetic_compare.sh   uni-modal baselines + none/opm/ogm/both ek saath
│   ├── run_seed_sweep.sh          3-seed sweep (reproducibility, mean±std)
│   ├── run_food101.sh             Food-101 subset pe poora comparison (image + prompt-text)
│   ├── run_meld.sh                MELD pe poora comparison (text + audio embeddings)
│   ├── make_food101_csv.py        Food-101 -> (image, CLIP-style prompt text, label) CSVs
│   ├── make_meld_csv.py           MELD official features -> packed data/meld/audio.npz + (audio,text,label) CSVs
│   ├── make_report.py             runs → results/RESULTS.md + plots
│   ├── build_walkthrough_notebook.py   notebook ko execute karke commit karne wala builder
│   └── make_demo_csv.py           CSV format ka demo dataset
├── results/                 ★ auto-generated report: RESULTS.md + bar/curve/rho plots
├── tests/                   32 unit tests (modulation, models, data, xai, metrics, analysis)
└── requirements.txt
```

---

## 0. Kya problem solve ho rahi hai (1 min recap)

Joint training me `logits = W¹φ¹ + W²φ² + b` — jo modality zyada discriminative hai wo loss pehle gira deti hai,
`∂ℓ/∂f = softmax(f) − onehot(y) ≈ 0` ho jaata hai, aur **weak modality ko gradient milna band** ho jaata hai
(under-optimized). Fix:

| | Kahan | Kya karta hai | Eq |
|---|---|---|---|
| **Discrepancy ratio ρᵐ** | har mini-batch | har modality ka apna score `sᵐ = softmax(Wᵐφᵐ + b/M)[y]`, ratio `ρᵐ = mean_j (Σsᵐ / Σsʲ)`; ρ>1 → dominant | 6, 7 |
| **OPM** | forward | dominant modality ka feature probability `q = q_base(1+λ·tanh(ρ−1))` se drop | 8 |
| **OGM** | backward | dominant encoder ka gradient `k = 1 − α·tanh(ρ−1)` se scale | 11, 12 |
| **GE** | backward | gradient me `N(0, std(grad))` noise add → SGD-noise recover `(k²+1)×` | 16, 17 |

---

## 1. Install

```bash
cd balanced_mm
python -m venv .venv && source .venv/bin/activate        # (optional)
pip install -r requirements.txt
# GPU: apne CUDA ke hisaab se torch install karo -> https://pytorch.org/get-started/locally/
python -m pytest tests/                                # 32 tests, sab pass aane chahiye
```

---

## 2. Quick start — 5 minute me synthetic data pe (koi data nahi chahiye)

Synthetic dataset me har modality sirf 70% samples me informative hai (independently), isliye
**uni-modal ceiling ≈ 0.75, fused ceiling ≈ 0.92** — fused model tabhi 0.9 cross karega jab **dono** encoders acche se seekhein.

```bash
# ek run (CPU pe ~15 s/epoch)
python train.py --data synthetic --image_encoder smallcnn --img_size 64 --epochs 15 \
                --syn_train_n 2400 --lr 0.01 --modulation ogm --out_dir runs/syn_ogm

# poora comparison: image-only, text-only, none, opm, ogm, both  (+ table)
bash scripts/run_synthetic_compare.sh 15 2400 smallcnn 3
python plot_history.py runs/syn_none runs/syn_opm runs/syn_ogm runs/syn_both --out runs/curves.png
```

Log me har epoch ye dikhega:

```
[ep 005] train loss=0.28 acc=0.99 | rho_image=2.25 rho_text=0.45 | k_image=0.58 q_image=0.00 k_text=1.00 q_text=0.00 || val acc=0.52 uni_image=0.41 uni_text=0.64
```
* `rho_image=2.25` → image dominant hai (ρ>1), text weak.
* `k_image=0.58` → OGM ne image encoder ka gradient 58% kar diya; `k_text=1.00` untouched.
* OPM me `q_image≈0.5–0.7` dikhega (image feature drop probability), `q_text=0`.
* `uni_*` = block-logits se cheap uni-modal accuracy (paper ka Fig 1 wala idea; exact protocol ke liye `--probe`).

---

## 3. Apne data pe (Text + Image)

### 3.1 CSV format
```
image,text,label
images/000.jpg,"a red car parked near the beach",car
images/001.jpg,"golden retriever playing with ball",dog
```
* `image` = path (relative to `--img_root` ya absolute), `text` = string, `label` = koi bhi string/int.
* Column names alag hain? → `--col_image img_path --col_text caption --col_label category`
* `--val_csv` nahi diya toh train ka 10% hold-out ho jaata hai.
* Format dekhne/test karne ke liye demo: `python scripts/make_demo_csv.py --out data/demo`

### 3.2 Train (paper default: ResNet-18 scratch + SGD 1e-3, batch 32)
```bash
python train.py --data csv --train_csv data/train.csv --val_csv data/val.csv --img_root data/images \
       --image_encoder resnet18 --img_size 224 --text_encoder transformer --max_len 64 \
       --modulation ogm --alpha 0.5 --epochs 60 --batch_size 32 --lr 1e-3 --sched cos \
       --out_dir runs/ogm --probe
```
Variants:
```bash
--modulation none                       # baseline (compare ke liye zaroor chalao)
--modulation opm --q_base 0.5 --lam 0.5 # OPM
--modulation both                       # OPM + OGM
--pretrained_image                      # ImageNet ResNet-18
--text_encoder hf --hf_model distilbert-base-uncased --lr_text 2e-5   # pretrained text (pip install transformers)
--head mlp --score_mode zero_out        # multi-layer classifier (paper Sec 4.3.7)
--only image  /  --only text            # uni-modal reference lines (paper Fig 1)
--amp                                   # mixed precision (GPU)
```

### 3.3 Evaluate / compare / plot
```bash
python eval.py --run_dir runs/ogm --test_csv data/test.csv --img_root data/images --probe
python compare.py runs/none runs/opm runs/ogm runs/both
python plot_history.py runs/none runs/opm runs/ogm runs/both --out curves.png
```
`--probe` = paper ka exact protocol (footnote 1): encoder freeze karke naya linear classifier fit → encoder ki asli quality.

Output folder me: `best.pt`, `last.pt`, `history.json` (har epoch ka ρ/k/q/acc), `summary.json`, `config.json`, `vocab.json`, `label_map.json`, `train.log`.

---

## 4. Existing project me plug-in (sirf `bml/modulation.py` copy karo)

```python
from bml.modulation import BalancedModulator, unimodal_logits_zero_out

mod = BalancedModulator(num_modalities=2, mode="both", q_base=0.5, lam=0.5, alpha=0.5, ge=True)

def unimodal_logits(feats):                       # Eq 6: W^m φ^m + b/M  (linear head ke weight blocks)
    W, b = head.weight, head.bias
    return [F.linear(feats[0], W[:, :D1]) + b/2, F.linear(feats[1], W[:, D1:]) + b/2]
    # MLP head:  return unimodal_logits_zero_out(lambda fs: head(torch.cat(fs, 1)), feats)

for epoch in range(E):
    for x_img, x_txt, y in loader:
        opt.zero_grad()
        feats = [img_enc(x_img), txt_enc(x_txt)]                              # 1. features (fusion se pehle)
        feats, rho = mod.before_fusion(feats, y, unimodal_logits, epoch)      # 2. ρ + OPM drop
        loss = F.cross_entropy(head(torch.cat(feats, 1)), y)
        loss.backward()
        # scaler.unscale_(opt)   <- agar AMP use kar rahe ho, modulation se PEHLE
        mod.after_backward([img_enc.parameters(), txt_enc.parameters()], epoch)  # 3. OGM-GE
        opt.step()
```
Poora runnable example: `python plug_in_example.py`. Low-level classes bhi available hain:
`OPM(...).apply/.update`, `OGMGE(...).coefficients/.apply_`, `discrepancy_ratios`, `unimodal_scores_from_logits`.

---

## 5. Equation → code mapping

| Paper | Code |
|---|---|
| Eq 2  `f = Σ Wᵐφᵐ + b` | `LateFusionModel.fuse` / `unimodal_logits(mode="linear")` (weight slicing) |
| Eq 6  `sᵢᵐ` | `modulation.unimodal_scores_from_logits` |
| Eq 7  `ρₜᵐ` | `modulation.discrepancy_ratios` |
| Eq 8  `qᵐ` | `OPM.update`; drop = `OPM.apply` (per-sample Bernoulli mask, weak modality q=0) |
| Eq 11 `kᵐ` | `OGMGE.coefficients` |
| Eq 12/16 update | `OGMGE.apply_` → `grad = k·grad + N(0, std(grad))` on encoder params |
| Eq 17 `(k²+1)` noise | verified in `tests/test_modulation.py::test_ge_noise_variance_recovers` |
| Sec 4.3.7 zero-out | `modulation.unimodal_logits_zero_out` / `--score_mode zero_out` |
| Alg 1 / Alg 2 order | `engine.train_one_epoch` |
| Footnote 1 probing | `engine.probe_unimodal_encoders` (`--probe`) |

---

## 6. Hyper-parameters & practical tips

| Param | Default | Paper range | Note |
|---|---|---|---|
| `--alpha` (OGM) | 0.5 | 0.2–0.6 | bada = dominant ko zyada slow |
| `--q_base` (OPM) | 0.5 | 0.2–0.6 | drop probability ka base |
| `--lam` (OPM) | 0.5 | 0.2–0.6 | discrepancy ke saath q kitna badhe |
| `--z` | tanh | tanh / sigmoid | dono chalte hain (Table 15) |
| `--mod_start/--mod_end` | 0 / ∞ | — | modulation sirf ek epoch window me (official code me bhi option hai) |
| `--ge_scope` | all | — | `all` = Eq 17 sab modalities pe (official code); `modulated` = sirf k<1 wali pe |

**Zaroori baatein**
1. **OGM ke liye SGD (+momentum) use karo** — paper bhi. Adam/AdamW gradient ko `√v` se normalize karta hai, isliye gradient ko `k` se scale karne ka effect kaafi kam ho jaata hai. Agar Adam use karna hi hai → **OPM prefer karo** (wo optimizer-independent hai).
2. **Baseline (`--modulation none`) same seed/config pe zaroor chalao** — improvement tabhi dikhegi.
3. **`ρ` ko dekho**: agar poore training me ρ≈1 hai toh imbalance hai hi nahi → modulation kuch nahi karegi (sahi behaviour). Agar ρ ≫ 1 aur `uni_acc` of weak modality baseline se upar aa raha hai → kaam kar raha hai. ρ 1 tak nahi pahunchega (paper Fig 5) — normal hai.
4. **Train loss thoda higher rahega** modulation ke saath (paper Fig 4) — ye expected hai, over-memorization kam hoti hai.
5. **Shuru me val acc baseline se kam** ho sakti hai, baad me upar (paper Sec 4.4.1) — patience rakho.
6. **AMP**: `scaler.unscale_(optimizer)` modulation se pehle (train.py me already hai). **Grad clipping** modulation ke *baad*.
7. **Batch size / LR**: SGD noise ∝ lr/batch (Table 1) — chhota batch (32) + lr 1e-3 paper setting hai; bahut bada batch generalization girata hai.
8. **Pretrained text encoder** (HF) → `--lr_text 2e-5` warna diverge/forget hoga.
9. **Multi-layer head** → `--score_mode zero_out` (auto ho jaata hai `--head mlp` ke saath).
10. **OPM + OGM** combine kabhi better, kabhi nahi (paper Table 12) — dono alag-alag bhi try karo.

---

## 7. Extend karna

* **3+ modalities** (e.g. text + image + audio): `LateFusionModel({"text":..., "image":..., "audio":...})` — `modulation.py` automatically M-way ρ compute karta hai (Eq 7 ka `1/(M−1) Σ_j`).
* **Audio (spectrogram)**: `ImageEncoderResNet18(in_channels=1)` — paper ke audio branch jaisa.
* **Apna encoder**: koi bhi `nn.Module` jisme `.out_dim` attribute ho aur `forward(inputs[name])` chale.
* **Missing modality at test time**: OPM-trained model zyada robust hai (paper Fig 6); eval me feature 0 kar do.
* **Sirf module chahiye**: `bml/modulation.py` standalone hai — koi internal import nahi.

---

## 8. Troubleshooting

| Symptom | Reason / fix |
|---|---|
| `rho` hamesha ≈1, k=1, q=0 | Modalities balanced hain ya dono abhi seekh hi nahi rahe (early epochs). Normal. |
| Weak modality ka `uni_acc` random-level pe stuck | LR bahut kam / encoder bahut weak; `--only text` chala ke dekho wo akele kitna seekhta hai. |
| OGM se koi fark nahi | Adam use kar rahe ho? → SGD ya OPM. `--alpha` badhao (0.6–0.8). |
| Training unstable with OPM | `--q_base 0.3`, ya `--mod_start 2` (pehle 2 epoch warm-up). |
| `assert block-wise logits need a linear head` | `--score_mode zero_out` |
| CUDA OOM with `--probe` | `probe_unimodal_encoders` features CPU pe rakhta hai; batch chhota karo. |

---

## 9. Reference results (synthetic, CPU, smallcnn, 15 epochs, 2400 train / 600 val)

### 9.1 Reproducibility — mean ± std over 3 seeds {0, 1, 2}

`bash scripts/run_seed_sweep.sh` (seeds 1, 2) + `runs/syn_*` (seed 0) → `python scripts/make_report.py`
se auto-report banti hai: **`results/RESULTS.md`** (+ `results/*.png`).

| modulation | best val acc | last-3-epoch avg | ρ_image (final) |
|---|---|---|---|
| none | 0.887 ± 0.022 | 0.884 | 1.344 |
| OGM-GE | 0.898 ± 0.017 | 0.894 | 1.188 |
| OPM | 0.904 ± 0.023 | 0.901 | 1.140 |
| **OPM + OGM-GE** | **0.906 ± 0.025** | 0.903 | **1.121** |

![best val acc](results/bar_val_acc.png)
![rho curves](results/rho_curves.png)
![acc curves](results/acc_curves.png)

Har seed me ordering same: fusion > none, modulation se ρ_image 1.34 → 1.12 (niiche wali table seed 0 ki detail hai).

### 9.2 Seed-0 detail (`bash scripts/run_synthetic_compare.sh 15 2400 smallcnn 3`, `runs/compare_log.txt`)

| run | modulation | best val acc (fused) | uni image | uni text | ρ_image (train, last) |
|---|---|---|---|---|---|
| syn_only_image | image-only baseline | 0.715 | 0.715 | – | – |
| syn_only_text | text-only baseline | 0.738 | – | 0.738 | – |
| syn_none | joint, **none** | 0.885 | 0.705 | 0.725 | **1.45** |
| syn_ogm | joint, **OGM-GE** | 0.898 | 0.717 | 0.728 | 1.25 |
| syn_opm | joint, **OPM** | 0.905 | 0.715 | 0.732 | 1.14 |
| syn_both | joint, **OPM+OGM** | **0.908** | **0.730** | 0.727 | 1.13 |

Last-3-epoch average (noise kam): none 0.884 → OGM 0.893 → OPM 0.901 → both 0.906.
Curves (`runs/curves.png`): discrepancy ratio modulation ke saath saaf girta hai (paper Fig 5 jaisa), train loss
thoda higher rehta hai (paper Fig 4 jaisa), fused val acc end me upar.

Severe-imbalance stress test: `--syn_synonyms 6` (text bahut slow seekhta hai) → baseline me ρ_image ≈ 4.8 tak
jaata hai, OPM use ≈ 2.3 pe le aata hai aur weak text ka uni-acc 0.25 → 0.31 (10 epochs). Zyada epochs chahiye.

*(Synthetic ceilings: uni-modal ≈ 0.75, fused ≈ 0.92. Numbers seed/CPU pe thode alag aa sakte hain; asli
dataset pe paper jaise 40–100 epochs chalao.)*

---

## 10. Real datasets — Food-101 & MELD

Do real benchmarks bhi built-in hain (CPU pe chalane ke liye subset-scale, scripts ready):

### 10.1 Food-101 — image + prompt-text (10 classes)

Food-101 me koi text nahi hota, isliye **CLIP-style prompt** text modality banaate hain: probability
`--text_p 0.7` pe prompt me asli class name (`"a photo of sushi"`), baaki time distractor
(`"a photo of food"`) — matlab **image strong, text imperfect = real modality imbalance**.

```bash
# 1) dataset download + extract (https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz)
#    tar -xzf food-101.tar.gz -C data/food101            (ya sirf 10 classes ka subset)
# 2) CSVs (2,500 train / 1,000 val, 10 classes)
python scripts/make_food101_csv.py --root data/food101
# 3) poora comparison: image-only, text-only + none/opm/ogm/both × seeds {0,1,2}
bash scripts/run_food101.sh
python scripts/make_report.py        # -> results/RESULTS.md (Food-101 section + plots)
```

### 10.2 MELD — text + audio (7 emotions, official splits)

Official release (declare-lab) ke **per-utterance 300-d audio embeddings** + raw dialogue text →
`VectorMLPEncoder` (MLP) + Transformer text encoder. Splits official: 9,989 / 1,109 / 2,610.

```bash
# 1) features tarball: https://huggingface.co/datasets/declare-lab/MELD
#    wget .../MELD.Features.Models.tar.gz && tar -xzf MELD.Features.Models.tar.gz
# 2) CSVs + packed audio.npz (13,708 x 300, ek hi file)
python scripts/make_meld_csv.py --features_dir <audio_emotion.pkl+data_emotion.p folder> --out data/meld
# 3) poora comparison: text-only, audio-only + none/opm/ogm/both × seeds {0,1,2}
bash scripts/run_meld.sh
python scripts/make_report.py
```

MELD me aksar **text dominant** nikalta hai (dialogue ka sentiment words se clearly milta hai,
audio embedding zyada noisy) → `ρ_text > 1` dikhega aur modulation **text ko slow** karega
(`k_text < 1`). Point yahi hai: **jo bhi modality dominant ho, modulation usi ko rokti hai**
— synthetic/Food-101 me image pe, MELD me text pe.

CSV format dono ka apne aap detect ho jaata hai:
* Food-101: `image,text,label` (classic)
* MELD: `audio,text,label` — image column hi nahi hai → image encoder skip, `--col_audio` (default `audio`) se `.npy` ya `audio.npz::<key>` load

### 10.3 Results (3 seeds each — auto-report: `results/RESULTS.md`)

**Food-101** (2,500 train / 1,000 val, 12 epochs, smallcnn@64):

| modulation | best val acc | ρ_text (final) | ρ_image (final) |
|---|---|---|---|
| none | 0.812 ± 0.004 | 2.73 | 0.38 |
| OGM-GE | 0.814 ± 0.005 | 2.69 | 0.39 |
| **OPM** | **0.845 ± 0.000** | **2.09** | 0.49 |
| OPM + OGM-GE | 0.842 ± 0.005 | 2.01 | 0.51 |

Uni-modal: image **0.554**, text **0.739** → fusion (none) +7.3 pts. Yahan **text dominant** nikla
(prompt me class name aata hai) — OPM ne ρ_text 2.51 → 1.91 kiya aur **uni-image acc 0.42 → 0.57**
(seed 0) le aaya: jo modality dab rahi thi wahi bachayi. Acc +3.1 pts.

**MELD** (9,989 train / 1,109 val, official splits, 12 epochs):

| modulation | best val acc | ρ_text (final) | ρ_audio (final) |
|---|---|---|---|
| none | **0.578 ± 0.002** | 2.22 | 0.46 |
| OGM-GE | 0.573 ± 0.002 | 1.60 | 0.63 |
| OPM | 0.565 ± 0.004 | 1.24 | 0.82 |
| OPM + OGM-GE | 0.561 ± 0.006 | 1.22 | 0.83 |

Uni-modal: text **0.560**, audio **0.469**. Balance **sabse dramatic yahin hua**: ρ_text 2.22 → 1.22,
ρ_audio 0.46 → 0.83 (gap lagbhag band). Lekin acc −0.5 se −1.7 pts — **honest trade-off**: MELD me
text alone ≈ fused (0.560) hai aur audio weak (0.469), isliye balance karne par strong modality ko
neeche aana pada zyada. (Agar sir yeh poochhein toh seedha jawab: mechanism sahi chal raha hai;
hyper-parameter tuning `--alpha/--q_base` + zyada epochs next step hai — paper me bhi har dataset
pe har method best nahi chalta.)

---

## Citation
```
@article{wei2024onthefly, title={On-the-fly Modulation for Balanced Multimodal Learning},
  author={Wei, Yake and Hu, Di and Du, Henghui and Wen, Ji-Rong}, journal={IEEE TPAMI}, year={2024}}
@inproceedings{peng2022balanced, title={Balanced Multimodal Learning via On-the-fly Gradient Modulation},
  author={Peng, Xiaokang and Wei, Yake and Deng, Andong and Wang, Dong and Hu, Di}, booktitle={CVPR}, year={2022}}
```
