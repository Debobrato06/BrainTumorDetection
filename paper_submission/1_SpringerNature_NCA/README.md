# Springer Nature — Neural Computing and Applications (NCA)

| File | Purpose |
| --- | --- |
| `manuscript.tex` | Full paper in the Springer Nature LaTeX template (`sn-jnl`, numbered references `sn-mathphys-num`) |
| `references.bib` | 48 entries (47 cited); errors in the original list fixed (marked `FIXED`), a few marked `VERIFY` |
| `figures/fig1_architecture.pdf/.png` | Figure 1, made by `../make_figure1.py` |
| `cover_letter.md` | Cover letter (corrected manuscript counts) |
| `BrainHybridNet_SpringerNature_NCA.docx` / `.pdf` | Word version (A4, single column, Times 11 pt, numbered references) and its PDF (11 pages) — for a Word submission or sharing. The LaTeX file stays the master copy |

## Compile (no LaTeX on this PC — use Overleaf)
1. Overleaf → New Project → Templates → search **"Springer Nature LaTeX Template"** → Open as template.
2. Upload `manuscript.tex`, `references.bib` and the `figures/` folder (keep the template's `sn-jnl.cls` and `.bst` files).
3. Set `manuscript.tex` as the main document, compile with pdfLaTeX. Red **[TBD]** marks show what is missing.

## Status: NOT ready to submit
Every result (Tables 1–4, abstract numbers, calibration, external validation) is `[TBD]` because the BraTS
experiments have not been run (`runs/` has no BraTS output). A paper without results will be desk-rejected.
Run the commands in the repository README ("3D experiments"), then fill each `[TBD]` from `src/evaluate.py` /
`src/stats.py` output only.

## Submission checklist
- [ ] All `[TBD]` filled from experiment outputs (abstract, Tables 1–4, Sections 5–7, cover letter)
- [ ] Figures 2 (reliability diagram) and 3 (qualitative examples) created — the text refers to them
- [ ] Author names, affiliations, institutional e-mails, ORCID for every author
- [ ] Abstract 150–250 words (current draft 179 words with placeholders — recount after filling)
- [ ] 4–6 keywords (6 given)
- [ ] Declarations complete (funding, competing interests, ethics, data/code availability, author contributions)
- [ ] Code repository public, with a release tag or Zenodo DOI cited in "Code availability"
- [ ] References marked `VERIFY` in `references.bib` checked against their DOI pages
- [ ] CLAIM checklist filled as supplementary material
- [ ] Check the current NCA "Submission guidelines" page for reference style and any length limit
      (the page could not be fetched automatically on 2026-10-25)
