# Journal Fit Analysis — BrainHybridNet manuscript

Date: 2026-10-25 · Source: `Main_Manuscript_NCA_BrainHybridNet.md`

## 1. Readiness verdict: not submittable yet

The manuscript has **no results**. Abstract numbers, Tables 1–4, calibration, external validation, ablations,
discussion and conclusion are all `[TBD]`, and the repository contains no BraTS experiment output (`runs/` holds only
the 2D web-app evaluations). Any journal would desk-reject it in this state. Both folders contain fully formatted
papers with every missing number marked in red, so the only remaining work is running the experiments and copying
the measured numbers in. No number was invented.

Experiments needed (commands in the repository README, "3D experiments"):

| Experiment | Runs |
| --- | --- |
| BrainHybridNet pre-training + fine-tuning | 5 folds |
| 5 baselines (3D U-Net, SegResNet, UNETR, SwinUNETR, nnU-Net v2) | 5 folds each |
| 5 ablations + masking-ratio / depth / loss-weight sweeps (3 seeds) | 5 folds each |
| Test set, MC-Dropout calibration, UCSF-PDGM external test and grading | 1 each |

This needs a CUDA GPU (3D 128³ crops, 300 epochs); this PC is CPU-only with ~1 GB free RAM and cannot run it.
UCSF-PDGM must be requested from The Cancer Imaging Archive.

## 2. Springer Nature

| Journal | Fit | Comment |
| --- | --- | --- |
| **Neural Computing and Applications** (recommended, already the target) | Good | Scope is applied neural computing; a hybrid CNN–transformer with self-supervised pre-training and Bayesian-style uncertainty applied to MRI fits directly. The journal has published closely related work (D-Former, cited). |
| Journal of Imaging Informatics in Medicine | Good | Choose this if the paper leans on clinical deployment (efficiency, uncertainty for review). |
| Scientific Reports | Fair | Judges soundness more than novelty; open access with an article processing charge. Option if NCA rejects for novelty. |

Risks with reviewers
- The topic is crowded (TransBTS, UNETR, SwinUNETR, nnU-Net). Novelty rests on the combination (multi-task + MIM
  pre-training + calibrated uncertainty + small model) and on the re-trained, identical-split comparison — the
  results must show either better accuracy or similar accuracy at clearly lower cost.
- MAE self pre-training on BraTS already exists (Zhou et al., ISBI 2023). The original text claimed the opposite;
  the contribution is now worded as "masking before the CNN stem of a hybrid encoder" and the paper is cited.
- nnU-Net is very hard to beat on BraTS; if BrainHybridNet loses, frame the result around efficiency, calibration
  and multi-task output, and report it honestly.

Format applied (`1_SpringerNature_NCA/`): `sn-jnl` template, numbered references, structured abstract (179 words
now), 6 keywords, Springer "Declarations" section (funding, competing interests, ethics, consent, data, materials,
code, author contributions). The NCA guideline page could not be fetched automatically (login redirect) — confirm
reference style and any length limit on it before submitting.

## 3. IEEE

| Journal | Fit | Comment |
| --- | --- | --- |
| **IEEE Journal of Biomedical and Health Informatics (J-BHI)** (recommended) | Good | Health-informatics angle (compact model, uncertainty to guide review, multi-task output) matches the journal. |
| IEEE Access | Fair–good | Broad, faster review, open access with a charge. Fallback. |
| IEEE Transactions on Medical Imaging | Weak | Expects stronger methodological novelty than a hybrid CNN–ViT combination; not recommended unless results are exceptional. |

J-BHI rules (EMBS author page, checked 2026-10-25): abstract ≤ 250 words with no abbreviations, references or
equations, ending with a significance sentence; 14-page hard limit; mandatory charges for pages 9–10 ($250 each)
and 11+ ($350 each); single-anonymous review; ORCID and institutional e-mail for every author; signed consent form.

Format applied (`2_IEEE_JBHI/`): `IEEEtran` journal, two columns, condensed to target ≈ 8 pages (related work
merged into one section, methods tightened, same tables), abstract rewritten without abbreviations (212 words),
Index Terms, cover-letter text for the submission box.

## 4. Errors found in the original manuscript and fixed in both versions

| Issue | Fix |
| --- | --- |
| "~300,000 new cases globally" cited to CBTRUS (a US-only report) | Global figure (≈308,000 in 2020) cited to GLOBOCAN 2020 (Sung et al. 2021) |
| Ref [23] D-Former listed as "Zhang Y, et al." | Authors corrected: Wu Y, Liao K, Chen J, et al. |
| Ref [27] (Zhou H-Y, "Preservational learning") described as "MAE on 3D CT" — it is not | Replaced by Zhou L et al., MAE self pre-training (ISBI 2023), which includes BraTS |
| Contribution 2 implied MIM on brain MRI was new | Reworded; prior work cited |
| Wrong years: TransBTS (2021, not 2022), Roy et al. (2018, not 2019) | Corrected |
| MoCo-CXR arXiv id 2101.02149 | Corrected to 2010.05352 |
| SegResNet and Tversky loss cited as preprints/workshop only | Cited to their Springer LNCS proceedings |
| Author "Debobrato [Last Name]", contribution initials "D.L." | Debobrato Biswas, "D.B." (co-authors still [TBD]) |
| Cover letter claimed ~8,500 words, 8 figures, 60 references | Manuscript has 1 figure (2 more planned), 4 tables (IEEE: 3), 47 references (IEEE: 45); counts corrected |
| ASCII architecture diagram | Real vector Figure 1 (`make_figure1.py`) |
| WHO "Grade IV" | WHO CNS5 uses Arabic numerals: "grade 4" |

Entries marked `VERIFY` in `references.bib` (TransBTS, MedT, Roy et al., D-Former pages) should be checked
against their DOI pages; the rest were checked against the original list or primary sources.

## 5. Order of work

1. Run the experiments (GPU), fill `[TBD]` in `1_SpringerNature_NCA/manuscript.tex` first.
2. Submit to Neural Computing and Applications.
3. Port the same numbers to `2_IEEE_JBHI/manuscript.tex` only if the paper is rejected or withdrawn — the same
   work must not be under review at two journals at the same time.

## 6. Alternative with results available today

The repository already has measured, reproducible results for a different study: the 2D tumor segmentation /
type model (`MODEL_CARD.md`, `runs/seg2d_eval`) — Dice 0.75 on unseen figshare patients, 99.8% detection and
98.9% 4-class accuracy on the Kaggle test split, with size and side estimation. That could become a separate paper
(e.g. IEEE Access or Scientific Reports) now, but its limitations (public overlapping datasets, no external
validation, one MRI contrast) must be stated. It is not the BrainHybridNet paper and should not be mixed into it.

## 7. Word and PDF versions

`paper_submission/_build/` regenerates both `.docx` files from `content_springer.js` / `content_ieee.js`
(text identical to the `.tex` files) and exports PDFs with Microsoft Word:

```
cd paper_submission/_build
npm install                     # docx 9.5.1
node build_docx.js springer ../1_SpringerNature_NCA/BrainHybridNet_SpringerNature_NCA.docx
node build_docx.js ieee ../2_IEEE_JBHI/BrainHybridNet_IEEE_JBHI.docx
powershell -ExecutionPolicy Bypass -File to_pdf.ps1 <in.docx> <out.pdf>
```

When results are filled in, edit the `.tex` file and the matching `content_*.js` together, then rebuild.
Equations in the Word files are typeset as text (Unicode); for the final submission the LaTeX version is preferred.

## 8. Update 2026-10-26 — submission-ready 2D paper

The alternative in Section 6 is now written: `3_SpringerNature_BMC_MedImaging/` contains the full paper
(two-column PDF and DOCX, LaTeX, figures, cover letter) with every number taken from
`runs/seg2d_paper/metrics.json`, and `JOURNAL_FIT_REPORT.md` recommends **BMC Medical Imaging** first,
then Journal of Imaging Informatics in Medicine, then Scientific Reports.
