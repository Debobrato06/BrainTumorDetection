# BMC Medical Imaging — guideline compliance check

Manuscript: *Segmentation, type classification and measurement of brain tumors on single 2D MRI slices: a
retrospective development and leakage-controlled internal validation study* · Article type: Research article ·
Checked 2026-10-26 against the journal's submission guidelines (text supplied by the author) and the BMC author
instructions for research articles.

## 1. Fit with the journal

| Journal criterion | This manuscript |
| --- | --- |
| Scope: development, evaluation and use of imaging techniques and image-processing tools to diagnose and manage disease | A segmentation, classification and measurement tool for brain tumors on MRI, evaluated against expert masks |
| Topic list includes **Neuroimaging** | Brain MRI |
| Decisions on scientific validity, not perceived impact: clear question, sound methods and analysis, community standards | Explicit research question (Background); patient-level splits, duplicate removal, patient-level bootstrap CIs, Bland–Altman, calibration, ablations; CLAIM checklist (Additional file 1) |

**Verdict: within scope; the paper's strength (careful, leakage-controlled evaluation) matches the journal's
soundness-based policy.** Main reviewer risks: no external test set and no segmentation baseline (both stated as
limitations).

## 2. Formatting rules

| Rule | Status | Where |
| --- | --- | --- |
| Double-line spacing | Done | Word: line spacing 2.0; LaTeX: `\doublespacing` |
| Line and page numbering | Done | Continuous line numbers; page numbers in footer |
| No page breaks | Done | Title page, abstract and text flow continuously |
| Editable file (DOC/DOCX/RTF/TeX) | Done | `Manuscript_BMC_MedicalImaging.docx`; `manuscript.tex` + `references.bib` + `figures/` |
| LaTeX: Springer Nature template, pdfLaTeX, TeX Live 2021 | Done | `\documentclass[pdflatex,sn-vancouver-num]{sn-jnl}`; only standard packages (lineno, setspace) |
| Special characters embedded | Done | Unicode in Word (Times New Roman); LaTeX uses ASCII commands only |
| SI units | Done | mm, mL, s |
| Title page: title (with study design), full names, institutional addresses, e-mail of all authors, corresponding author | Done | Start of manuscript |
| Structured abstract (Background, Methods, Results, Conclusions), ≤ 350 words, no references/figures/tables, few abbreviations | Done | 318 words; abbreviations defined (2D, MRI) |
| Keywords | Done | 7 |
| Sections: Background, Methods, Results, Discussion, Conclusions | Done | Unnumbered headings |
| List of abbreviations | Done | After Conclusions |
| Declarations: ethics and consent; consent for publication; availability of data and materials; competing interests; funding (study and publication fees); authors' contributions (initials); acknowledgements | Done | Declarations section |
| Data availability in the required wording, datasets cited with DOI in the reference list | Done | figshare DOI 10.6084/m9.figshare.1512427 [8]; Kaggle DOI 10.34740/KAGGLE/DSV/2645886 [9] |
| Software block (name, home page, archive, OS, language, requirements, licence, restrictions) | **2 items for the author** | Archive DOI and licence are marked [TBD] in red |
| Web links as numbered references with access date | Done | GitHub repository [32] with "Accessed 26 Oct 2026" |
| Vancouver style, numbers in square brackets, MEDLINE journal abbreviations | Done | 32 references, numbered in order of first citation |
| Figures numbered in order of first mention; composite multi-panel files | Done | Figures 1–6 |
| Figure titles ≤ 15 words and legends ≤ 300 words in the manuscript, not in the graphic | Done | "Figure titles and legends" section; longest title 11 words |
| Figure files: PDF (vector) or PNG, ≤ 10 MB, closely cropped, ~300 dpi at 170 mm | Done | `figure_files/Figure1–6.pdf/.png`; largest 1.4 MB; PNG ≥ 2000 px wide |
| Tables: numbered, cited in sequence, title above (≤ 15 words), legend below, no colour/shading, no commas in numbers, made with the table tool | Done | Tables 1–3 |
| Additional files named "Additional file N", cited in sequence in the text, listed with file name, format, title and description; tabular data as CSV | Done | Additional files 1–4 (PDF, CSV, CSV, CSV); "Additional files" section |
| No individual participant details in additional files | Done | Only public dataset identifiers |

## 3. Cover letter (required contents)

| Item | Status |
| --- | --- |
| Why the manuscript should be published in BMC Medical Imaging | Done |
| Issues relating to journal policies | Done (none; public de-identified data) |
| Competing interests declaration | Done |
| All authors approved the submission | Done (sole author) |
| Not published or submitted elsewhere | Done |
| Suggested reviewers with institutional e-mail / ORCID or Scopus ID | **For the author to add** |

## 4. Fees and licence (from the guidelines)

- APC: £2490 / $3290 / €2790, set at acceptance; the journal is in Springer Nature's **country-tiered APC
  pricing pilot**, which replaces the low-income waiver list — check the price for Bangladesh on the
  journal's pricing page.
- Any discretionary waiver must be requested **at submission**; it cannot be requested later.
- Licence: CC BY or CC BY-NC-ND. Transparent peer review: reviewer reports are published (reviewers anonymous).

## 5. What the author still has to do before pressing "submit"

1. Add 3–4 suggested reviewers (cover letter).
2. Choose a licence for the GitHub repository (e.g. MIT) and, ideally, archive a release on Zenodo; replace the two
   red [TBD] markers in the software block (in `../_build/content_bmc2d.js`, then rebuild).
3. Confirm the Funding statement ("No funding was received for this study or for the publication fees") — it must
   name whoever pays the APC.
4. Confirm the e-mail address (institutional addresses are preferred by many editors) and add co-authors or a
   supervisor if any.
5. Optional but strongly recommended: train the segmentation baseline (`python src/train_unet_baseline.py`), which
   the build adds to the paper automatically, and test on an external dataset.
6. Optional: language editing (not required by the journal).
