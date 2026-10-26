# IEEE — Journal of Biomedical and Health Informatics (J-BHI)

| File | Purpose |
| --- | --- |
| `manuscript.tex` | Condensed two-column paper in `IEEEtran` (journal mode), target ≈ 8 pages |
| `references.bib` | Same corrected references as the Springer version (IEEEtran style) |
| `figures/fig1_architecture.pdf/.png` | Figure 1 |
| `cover_letter.md` | Text for the submission system's cover-letter box |
| `BrainHybridNet_IEEE_JBHI.docx` / `.pdf` | Word version (US Letter, two columns, Times 10 pt, IEEE references) and its PDF (4 pages now; grows when results are added) |

## Compile (Overleaf)
New Project → Blank Project → upload all files and `figures/` → compile `manuscript.tex` with pdfLaTeX
(`IEEEtran.cls` and `IEEEtran.bst` are built into Overleaf).

## J-BHI rules checked on 2026-10-25 (embs.org/jbhi/prepare-and-submit-your-manuscript)
- Abstract ≤ 250 words, **no abbreviations, references or equations**; state objective, methods, results,
  conclusion and one sentence of significance. The draft follows this (212 words, no abbreviations —
  recount after filling).
- 14 pages max for a regular paper including supplementary material. Overlength charges are mandatory:
  pages 9–10 cost $250 each, page 11 onward $350 each. This version is condensed for ≈ 8 pages.
- Single-anonymous review, ≥ 3 reviewers. Names stay on the manuscript.
- Every author needs an ORCID and an institutional or IEEE e-mail; multi-author papers need a signed
  author consent form.
- Public datasets must be cited (done for BraTS 2021 and UCSF-PDGM).
- 4 suggested reviewers from other institutions are encouraged.

## Status: NOT ready to submit
Same blocker as the Springer version: all results are `[TBD]`; the BraTS experiments have not been run.

## Checklist
- [ ] All `[TBD]` filled from experiment outputs; recount abstract words (≤ 250, no abbreviations)
- [ ] Compiled PDF ≤ 8 pages (or accept overlength charges), ≤ 14 pages hard limit
- [ ] Author list, affiliations, institutional e-mails, ORCIDs; consent form signed by all authors
- [ ] Optional: author biographies/photos are only needed at final acceptance (check Author Center)
- [ ] Code repository public with a release tag
- [ ] References marked `VERIFY` in `references.bib` checked
