# 2D brain tumor paper — two-column version, Springer Nature journal template (target: BMC Medical Imaging)

This is the typeset-style, two-column version. The file to upload to BMC Medical Imaging is the one-column,
double-spaced manuscript in `../4_BMC_MedicalImaging_Submission/` (the journal requires that layout).

Template structure followed (Springer Nature LaTeX template, `sn-jnl` with `iicol`, sample `sn-article.tex`):
centred title, authors with affiliation numbers and `*` for the corresponding author, structured affiliation
(`\orgdiv`, `\orgname`, `\orgaddress`), "*Corresponding author(s). E-mail(s):", abstract and keywords,
numbered sections, `Fig. n` / `Table n` captions with table footnotes, running-title header, and back matter in
the template's order — Supplementary information, Acknowledgements, Declarations (Funding, Competing interests,
Ethics approval and consent to participate, Consent for publication, Data availability, Materials availability,
Code availability, Author contribution) — then numbered Vancouver references.

| File | Purpose |
| --- | --- |
| `BrainTumor2D_SpringerNature.pdf` | Final paper, two columns (A4), exported from Word |
| `BrainTumor2D_SpringerNature.docx` | Same paper as an editable Word file (two columns) |
| `manuscript.tex` + `references.bib` + `figures/` | LaTeX version (Springer Nature template `sn-jnl`, options `sn-vancouver-num,iicol`) |
| `figures/` | Figures 1–6 as 300-dpi PNG and vector PDF |
| `metrics.json` | Every number in the paper, produced by `src/paper_metrics_seg2d.py` |
| `cover_letter.md` | Cover letter for BMC Medical Imaging |
| `JOURNAL_FIT_REPORT.md` | Which Springer Nature journal fits and why |
| `supplementary/` | Additional files 1–4 cited in the paper |

## Where the numbers come from
All results are read automatically from the evaluation outputs (`runs/seg2d_paper/metrics.json`,
`runs/seg2d_paper/unet_baseline_test.json`, `runs/seg2d/split.json`) by `../_build/content_bmc2d.js`;
nothing is typed in by hand.

```
python src/paper_metrics_seg2d.py                       # metrics + evaluation figures
python src/train_unet_baseline.py --epochs 12           # optional U-Net baseline (CPU: ~3 h, ~2.5 GB RAM);
                                                        # the paper includes it automatically once it has run
python paper_submission/make_figures_2d.py              # Fig. 1, Fig. 6, final figure set
cd paper_submission/_build
node build_docx.js bmc ../3_SpringerNature_BMC_MedImaging/BrainTumor2D_SpringerNature.docx
node build_tex.js ../3_SpringerNature_BMC_MedImaging/manuscript.tex
powershell -ExecutionPolicy Bypass -File to_pdf.ps1 <docx> <pdf>
```

## LaTeX
No LaTeX installation on this PC; the `.tex` was checked for balanced braces/environments and that every
citation and label resolves. Compile on Overleaf with the "Springer Nature LaTeX Template" (keep its
`sn-jnl.cls` and `.bst` files). If `sn-vancouver-num` is not available in the template version, use
`sn-mathphys-num`. For BMC submission a single-column file is also accepted — remove `iicol`.
