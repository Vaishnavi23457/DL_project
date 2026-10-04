# Project Review: Questions and Answers (Hinglish)

Modality bias, OPM aur OGM-GE ke concepts, implementation aur experiments ke review notes.
Equation numbers `docs/Wei2024_On-the-fly_Modulation_TPAMI.pdf` ke according hain.
Detailed derivations: `docs/OGM_OPM_Explanation.pdf`.

Code paths neeche `balanced_mm/` ke relative hain, jab tak full repository path na diya ho.

---

## A. Modality Bias: Concepts

**Q1. Modality bias / imbalance samjhao.**
Joint training me fusion logit **sum** hota hai: `f = W¹φ¹ + W²φ² + b` (Eq 2). Jo modality
zyada discriminative hoti hai, uska contribution bada → usi ka gradient `∂ℓ/∂f = softmax(f) − 1[y]`
(Eq 5) seedha usi encoder ko push karta hai → **rich-get-richer**: ek modality dominate karne
lagti hai, doosri underfit. Paper ka observation: gradient ka direction ek hi modality decide
karta hai.
*Code:* `bml/modulation.py` (Eq 6/7 ρ), `bml/engine.py` (Alg 1/2 order).

**Q2. Bias hai ya phir wahi hota hai ki ek modality strong hai?**
Strong hona alag baat hai — **imbalance** tab hai jab training me *consistently* ek hi ka
contribution badhta jaye. Measure karte hain **discrepancy ratio ρ** (Eq 6/7): 1 = balanced,
>1 = pehli modality dominant. Hamare synthetic runs me `none` pe ρ_image ≈ 1.3–1.4 rehta hai,
modulation ke baad ≈ 1.1–1.2.
*Output:* `results/RESULTS.md`, `balanced_mm/runs/curves.png`.

**Q3. Bias ko detect kaise karo? (XAI wala part)**
Hamare proposal ke hisaab se teen tarike, repo me teeno implemented:
1. **Uni-modal probe** — frozen encoder pe linear classifier; jiska acc kam uska encoder
   weak (`train.py --probe`, paper footnote 1).
2. **Attention / Integrated Gradients** — model *kahan dekh raha hai* (`bml/xai.py`:
   `integrated_gradients_image`, `token_importance_text`, `modality_attribution`).
3. **ρ ratio** — optimization-side signal, bina kisi extra model ke (Eq 6/7).

**Q4. Uni-modal baselines ka point?**
`--only image` / `--only text` se pata chalta hai fusion **actually** kar kya raha hai.
Hamare results: image-only 0.715, text-only 0.738 → fusion 0.885 (≈ +15 pts). Matlab fusion
sach me dono use kar raha hai; modulation usko aur balanced banata hai.
*Output:* `results/RESULTS.md` → "Uni-modal baselines" table.

---

## B. Mathematical Formulation

**Q5. Eq 6/7 — score kaise banta hai?**
Per block: `s^m_i = softmax(W^m φ^m_i + b/M)_{y_i}` — i.e. **sirf modality m ke contribution**
ka true-class probability (bias b/M share hota hai).
`ρ^m = (1/(M−1)) Σ_{j≠m} (Σ_i s^m_i) / (Σ_i s^j_i)` — dominant modality ka score doosre se
kitna bara (batch-averaged). ρ=1 → balance.
*Code:* `discrepancy_ratios()` + `unimodal_scores_from_logits()` in `bml/modulation.py` (no_grad).

**Q6. ρ se modulation factor kaise milta hai?**
`z = tanh(ρ − 1)` — saturating, sign rakhta hai (ρ>1 → z>0).

- **OPM (Eq 8):** dominant modality ke samples ko **randomly drop** karta hai training forward me:
  `q^m = q_base·(1 + λ·z)` probability se feature zero (sirf tab jab ρ>1; drop = `q/2` rate se
  Bernoulli, no rescale). Doosre modality ka gradient ≈ free pass milta hai.
- **OGM-GE (Eq 11/12):** dominant modality ke **encoder parameters** ke gradients ko scale:
  `k^m = 1 − α·z` (jab ρ>1, doosre modality ka k=1). Gradient badhta nahi — sirf *kam* hota hai
  (k ∈ [0,1]) → train stability.
- **GE (Eq 16/17):** OGM ke baad un sabhi parameters me Gaussian noise add jo `std(grad)` se aata
  hai → dono modalities ke gradients ka **ratio** preserve rehta hai + regularization milta hai.
  `(k²+1) × noise` factor paper me derive hota hai.
*Code:* `OPM`, `OGMGE`, `BalancedModulator` in `bml/modulation.py`; hooks `before_fusion` /
`after_backward` in `bml/engine.py`.

**Q7. Defaults / hyper-parameters?**
`q_base = λ = α = 0.5`, `z = tanh`, SGD momentum 0.9 (paper recommendation: OGM ko momentum
SGD chahiye; OPM Adam/SGD dono chalta hai). Sab flags se changeable:
`train.py --q_base --lam --alpha --z --mod_start --mod_end`.

**Q8. Overhead kitna hai?**
ρ nikalne ke liye **sirf ek no-grad pass** over block logits (linear head me exact, cheap);
OPM forward me ek Bernoulli mask; OGM backward me element-wise grad multiply + optional noise.
FLOPs pe ~0 overhead, memory me batch ka ek extra tuple (`last_unimodal_logits`).
Paper me bhi inference pe **zero cost** (modulation training-time only).

**Q9. ρ>1 wali condition kyu?**
Agar kisi ki bhi ρ<1 (wo dominant nahi) to usse **kuch nahi karna** — sirf dominant modality ko
rokna hai. Isliye `if ρ>1: apply else: q=0 / k=1`. Agle step ρ phir measure hota hai (Alg 1/2).

---

## C. Implementation

**Q10. Implementation ki main files ka kya role hai?**
| File | Kaam |
|---|---|
| `bml/modulation.py` | Eq 6/7 ρ + OPM (Eq 8) + OGM-GE (Eq 11/17) + `BalancedModulator` — modulation module |
| `bml/engine.py` | training/eval loops — Alg 1/2 ka exact order (encode → before_fusion → CE → after_backward → step) |
| `bml/models.py` | ResNet-18 / SmallCNN image encoder, Transformer/HF text, late-fusion head + block logits |
| `bml/data.py` | CSV pipeline (`image,text,label`) + synthetic imbalance dataset |
| `bml/xai.py` | **detection side**: IG, grad×input tokens, modality attribution shares |
| `bml/metrics.py` | confusion matrix, per-class acc, macro-F1, ECE |
| `bml/analysis.py` + `scripts/make_report.py` | runs → `results/RESULTS.md` + plots (auto report) |
| `train.py / eval.py / compare.py` | CLI training, checkpoint eval, multi-run table |
| `tests/` | Unit tests (`python -m pytest tests/`) |
| `notebooks/01_walkthrough.ipynb` | 4-epoch demo + plots + XAI attribution — saved code outputs |

**Q11. Modulation lagane ka exact order?**
```
feats = model.encode(inputs)                    # ek baar forward
feats, rho = modulator.before_fusion(...)       # ρ (Eq 6/7), OPM drop, ρ update
logits = model.fuse(feats); loss.backward()
modulator.after_backward(model.modality_parameters(), epoch)   # OGM: grad × k (+ GE noise)
optimizer.step()
```
*Implementation reference:* `bml/engine.py` docstring + `train_one_epoch()`.

**Q12. Drop ka matlab output zero ho gayi to agla layer?**
OPM drop **training forward** me hota hai (batch ke kuch samples me), aur linear head ke paas
bias `b/M` hota hi hai → NaN/undefined nahi. Eval/val me drop **nahi** hota (modulator eval
mode me pass-through) — isliye val acc inflated nahi hota.

**Q13. OPM aur OGM me difference?**
| | OPM | OGM-GE |
|---|---|---|
| kya badalta hai | forward — dominant samples ka **feature drop** | backward — dominant encoder ka **grad scale** |
| granularity | per-sample, random | poore modality parameters par constant factor |
| noise | nahi | GE Gaussian (Eq 17) |
| combo dono chalta hai? | haan — `--modulation both` (humne 4 configs chalayi hain) | |

---

## D. Experiments and Results

**Q14. Synthetic data kaisa hai?**
6 classes; har sample ka text **prob 0.7** pe class-keyword (uske baad synonyms) rakhta hai,
image **prob 0.7** pe class-patch rakhti hai (baaki noise). Modality-wise informative
probability hi **imbalance ka source** (aur `--syn_synonyms 6` se severe). 2400 train / 600 val,
64×64 images, SmallCNN + 2-layer Transformer, SGD lr 0.01, 15 epochs, seed sweep {0,1,2}.

**Q15. Main result?**
`results/RESULTS.md` (auto-generated, mean ± std over seeds {0,1,2}, **teen datasets**):

- **synthetic**: fusion **none 0.887 ± 0.022 → OGM 0.898 → OPM 0.904 → both 0.906 ± 0.025**
  (har seed me same ordering); ρ_image 1.34 → 1.12; uni 0.715 / 0.738
- **Food-101** (image + prompt-text): none 0.826 → **OPM 0.857** (+3.1 pts), ρ_text 2.51 → 1.91,
  uni-image 0.42 → 0.57 (seed 0) — OPM ne dominated image encoder ko bachaya. Uni 0.591 / 0.739.
- **MELD** (text + audio): none 0.578 → OGM 0.573 / OPM 0.565 / both 0.561 — balance **sabse
  strong yahin** (ρ_text 2.22 → 1.22, ρ_audio 0.46 → 0.83) lekin acc −0.5..−1.7 pts. Uni 0.560 / 0.469.
  *MELD par accuracy trade-off:* MELD me text alone ≈ fused (0.560) hai aur audio
  weak (0.469) — balance karne par strong ko neeche aana pada zyada. Mechanism sahi chal raha hai;
  `--alpha/--q_base` tuning + zyada epochs next step. (Paper me bhi har dataset pe har method
  best nahi chalta.)
- stress (`--syn_synonyms 6`): none ρ 4.8 → OPM 2.3
Paper (real datasets, Concat→OPM/OGM/both): CREMA-D 66.9→75.1/74.6/76.7 etc. — hamara code
wahi method, Text+Image / Text+Audio settings me.

**Q16. Experiment results aur run records kahan hain?**

- `balanced_mm/results/RESULTS.md` aur `balanced_mm/results/*.png`: synthetic, Food-101 aur MELD ke result tables aur plots.
- `balanced_mm/runs/`: training logs, epoch histories, run summaries aur comparison logs.
- `balanced_mm/notebooks/01_walkthrough.ipynb`: example workflow aur saved outputs.
- `balanced_mm/tests/`: implementation ke unit tests. Tests chalane ke liye `balanced_mm/` se `python -m pytest tests/` use karein.

**Q17. Multiple seeds se results kaise evaluate kiye gaye?**
Har dataset pe 3 seeds {0,1,2} × 4 methods = **12 fusion runs × 3 datasets = 36** (+6 uni-modal),
sab `runs/` me logs ke saath; tables me mean ± std. Scripts: `run_seed_sweep.sh`,
`run_food101.sh`, `run_meld.sh` → `python scripts/make_report.py`.

**Q18. Current limitations aur next steps kya hain?**
- abhi synthetic benchmark (imbalance controlled) ke saath **Food-101** (image+prompt-text, 10 classes)
  aur **MELD** (text+audio, official splits) pe bhi runs hain — dono real datasets hain, scale CPU ke
  hisaab se (subset / official features) rakha gaya; **next step**: poora dataset + ResNet-18/HF encoders
  (`--pretrained_image --text_encoder hf`)
- modulation mitigation ka part hai; **detection/XAI** side hamare proposal ka MBS + IG hai
  (`bml/xai.py`) — dono milke "detect → mitigate" pipeline banate hain.

**Q18b. Food-101 me text inputs kaise banaye gaye?**
Food-101 sirf images ka dataset hai — humne **CLIP-style prompt text** banaya: 70% samples me prompt
me class name, 30% distractor (`scripts/make_food101_csv.py --text_p 0.7`). Ye methodology ka hissa
hai (CLIP ne bhi prompts use kiye) aur imbalance ka clean source hai: image reliable, text noisy.
Training images 96×96 px hain (repo compact rakne ke liye) — full-res pe yehi protocol ~1.4 pts
better tha (none 0.826, OPM 0.857) lekin ranking/ conclusions same: OPM > both > OGM > none.

**Q18c. MELD me audio kaise?**
Raw MELD (mp4) ~11 GB hai — humne **official MELD features release** (declare-lab) use kiya:
per-utterance 300-d audio embeddings + asli text (`scripts/make_meld_csv.py`). Ye wahi features hain
jinse MELD ke official baselines train hote hain. Model: text Transformer + audio MLP (VectorMLPEncoder).

---

## E. Relation to the Project Proposal

**Q19. Hamare XAI proposal se ye paper kaise relate hota hai?**
Proposal: *"XAI-Driven Modality Bias **Detection** and Mitigation"*.
- **Detection:** attention heat-maps (Colab notebook) + Integrated Gradients + MBS score →
  *where* model dekh raha hai; ρ + probe → *how strongly* ek modality dominate.
- **Mitigation:** training-time modulation (OPM/OGM-GE) — TPAMI 2024 ka strong baseline jo
  humne implement kiya; aage "Balanced Modal Attention" (proposal ka apna idea) isi ke
  comparison me aayega.
*Flow:* detect (XAI) → quantify (ρ) → mitigate (modulation) → verify (probe, curves).

**Q20. Project mein PaliGemma 3B ka kya role hai?**
`multimodal (2).ipynb` me loading + prompt demo + attention heat-map hai (Colab GPU).
End-to-end modulation 3B pe Colab free tier me possible nahi isliye **method ko chhota karke**
Text+Image late-fusion pe validate kiya (repo ka `balanced_mm/`) — method same, compute feasible.
Aage ka plan: detection scores (MBS/IG) ko modulation ke saath jodna.

---

## Key Concepts

1. `f = Σ W^m φ^m + b` → softmax − y gradient → strong modality ko zyada push
2. ρ (Eq 6/7) = score ratio, 1 = balanced
3. `z = tanh(ρ−1)`; OPM = drop, OGM = grad×k, GE = noise (Eq 17); performance dataset aur configuration par depend karti hai
4. Repo map: modulation.py / engine.py / xai.py / analysis.py / tests / notebook
5. Numbers: none 0.885 → both ≈ 0.908; ρ 1.45 → 1.13; uni 0.715/0.738; 3 seeds
6. Current limitations: dataset scale, compute budget aur detection-mitigation integration
