# Two-column final version (JIIM paper)

`BrainTumor2D_JIIM_TwoColumn.pdf` / `.docx` contain the same text, numbers, references and figures as
the JIIM submission in `../5_JIIM_Submission`, laid out as a two-column Springer Nature journal article
(A4, 8 pages). The figures sit in the text, and there is a running title with page numbers.

Use it for reading, sharing or a preprint. **Do not submit it to JIIM.** The journal asks for a
single-column Word file and a separate blinded manuscript, so upload the files in `../5_JIIM_Submission`.

Rebuild after any change to `_build/content_jiim.js`:

```
cd paper_submission/_build
node build_docx.js jiim2col ../6_JIIM_TwoColumn/BrainTumor2D_JIIM_TwoColumn.docx
powershell -File to_pdf.ps1 ../6_JIIM_TwoColumn/BrainTumor2D_JIIM_TwoColumn.docx ../6_JIIM_TwoColumn/BrainTumor2D_JIIM_TwoColumn.pdf
```
