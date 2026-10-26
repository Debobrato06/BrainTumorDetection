# BMC Medical Imaging — submission package

Upload in the submission system:

| Upload as | File |
| --- | --- |
| Manuscript (Word route) | `Manuscript_BMC_MedicalImaging.docx` — title page, abstract, text, tables, declarations, references, figure titles and legends, additional-files list; double spacing, line and page numbers |
| *or* Manuscript (LaTeX route) | `manuscript.tex` + `references.bib` + `figures/` (Springer Nature template `sn-jnl`; BMC compiles with pdfLaTeX) |
| Figures 1–6 | `figure_files/Figure1.pdf` … `Figure6.pdf` (PNG versions also provided) |
| Additional file 1 | `additional_files/Additional_file_1.pdf` (CLAIM checklist, Tables S1–S6; editable `.docx` included) |
| Additional files 2–4 | `additional_files/Additional_file_2.csv`, `_3.csv`, `_4.csv` |
| Cover letter | `Cover_letter.md` (paste into the cover-letter field) |

For reading only (do not upload): `Manuscript_BMC_MedicalImaging.pdf` (PDF of the Word file) and
`Manuscript_with_figures_preview.pdf` (same, with figures placed after their legends).

`BMC_GUIDELINE_COMPLIANCE.md` lists every journal rule and where it is met, plus the few items only the author can
complete (suggested reviewers, repository licence/archive DOI, funding of the APC).

Rebuild after any change (all numbers are read from `runs/seg2d_paper/metrics.json`):

```
cd paper_submission/_build
node build_docx.js bmcsub ../4_BMC_MedicalImaging_Submission/Manuscript_BMC_MedicalImaging.docx
node build_tex.js ../4_BMC_MedicalImaging_Submission/manuscript.tex sub
node build_additional.js ../4_BMC_MedicalImaging_Submission/additional_files/Additional_file_1.docx
powershell -ExecutionPolicy Bypass -File to_pdf.ps1 <docx> <pdf>
```
