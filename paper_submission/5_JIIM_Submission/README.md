# Journal of Imaging Informatics in Medicine (JIIM): submission package

Springer, hybrid journal. Choose the **subscription (non-open-access) route** to publish without an
article processing charge. Do **not** select "Open Choice" after acceptance.

Submission site: Editorial Manager, linked from https://link.springer.com/journal/10278

## What to upload

| Upload item (Editorial Manager) | File |
|---|---|
| Title page | `1_Title_Page.docx` |
| Blinded manuscript (main document) | `2_Manuscript_Blinded.docx` |
| Manuscript with author details, if the system asks for it | `3_Manuscript_Full.docx` |
| Figures 1–6 | `figures/Fig1.tif` … `Fig6.tif` |
| Electronic supplementary material | `online_resources/ESM_1.pdf`, `ESM_2.xlsx`, `ESM_3.xlsx`, `ESM_4.xlsx` |
| Cover letter | paste or upload `Cover_Letter.md` |

`Preview_with_figures.pdf` is for checking only. It is the full manuscript with each figure under its
caption. Do not upload it.

## Author guideline compliance

| Guideline | How the package meets it |
|---|---|
| Article type: hypothesis-driven research | Abstract, Introduction, Materials and Methods, Results, Discussion, Conclusions; Acknowledgements on the title page |
| Abstract 150–250 words, no undefined abbreviations | Unstructured abstract of 232 words |
| 4–6 keywords | 6 keywords |
| Word file, plain font such as 10 pt Times | Word, 10 pt Times New Roman, one column, automatic page numbers. Double spacing and line numbers are added for reviewers. |
| At most three heading levels | Two levels are used |
| Double-blind review | Blinded manuscript has no names, affiliation, repository link or self-citation. The Word file author is "Anonymous". The online resources carry no author details either. |
| Title page: authors, affiliation with address, e-mail, ORCID | `1_Title_Page.docx` |
| Statements and Declarations on the title page | Funding, competing interests, author contributions, ethics approval, consent to participate, consent to publish, data availability, code availability |
| Generative AI use described in Methods | One paragraph in Materials and Methods, Study design |
| Reporting guideline (EQUATOR) | CLAIM checklist (Online Resource 1, Table S1); TRIPOD+AI cited |
| References: numbered by first citation, JIIM style | e.g. "Dice LR: Measures of the amount of ecologic association between species. Ecology 26:297-302, 1945. https://doi.org/…" |
| Tables in the text, numbered, captions, footnotes below the table | 3 tables |
| Figures: separate files named Fig1 …; 600 dpi for combination art; RGB; 174 mm wide; Arial lettering 8–12 pt; panels (a), (b); no title inside the figure | `figures/`: TIFF with LZW compression, 600 dpi, 174 mm wide, RGB |
| Captions: bold "Fig. n", no punctuation after the number or at the end of the caption | "Figure Captions" section after the references |
| Expert and model outlines must stay distinguishable without colour | In Fig. 2 the expert outline is thicker than the model outline |
| Supplementary files: "Online Resource n", named ESM_n, text as PDF, tables as .xlsx, with article title and journal name | `online_resources/`. Each .xlsx has an "About" sheet. |
| Software for readers: include a user manual | The GitHub repository README serves as the user guide. Check that it explains installation and use. |

## You still need to fill in (red `[TBD]` in the title page and the full manuscript)

1. **ORCID.** Create one free at https://orcid.org.
2. **Zenodo DOI** of a GitHub release. Connect the repository to Zenodo, then publish a release.
3. **License** of the repository, for example MIT. Add a `LICENSE` file to the repository.
4. **Address.** Check the affiliation address on the title page (Daffodil Smart City, Birulia, Savar,
   Dhaka 1216). It will be published.
5. **ICMJE disclosure form.** JIIM asks for it together with the competing-interests statement.
6. **Generative AI paragraph.** Check that the paragraph in Materials and Methods describes your use
   accurately, and edit it if needed.

After you edit the content, regenerate the files instead of editing the Word files by hand (see below).

## Rebuilding

```
cd paper_submission/_build
node build_docx.js jiimtitle ../5_JIIM_Submission/1_Title_Page.docx
node build_docx.js jiimblind ../5_JIIM_Submission/2_Manuscript_Blinded.docx
node build_docx.js jiim      ../5_JIIM_Submission/3_Manuscript_Full.docx
PREVIEW=1 node build_docx.js jiim ../5_JIIM_Submission/Preview_with_figures.docx   # then export to PDF
node build_additional.js ../5_JIIM_Submission/online_resources/ESM_1.docx jiim     # then export to PDF
python make_esm_xlsx.py
python ../make_figures_jiim.py            # Fig1-Fig6; figures 2 and 6 run the model
powershell -File to_pdf.ps1 <in.docx> <out.pdf>
```

Text and numbers come from `_build/content_jiim.js`, which builds on `content_bmc2d.js`. All results
are read from `runs/seg2d_paper/metrics.json`.
