# Journal Fit Report — 2D brain tumor paper (Springer Nature)

Paper: *Brain tumor segmentation, type classification and size and laterality estimation from single 2D
MRI slices: a leakage-controlled, patient-level evaluation* · Date: 2026-10-25

## 1. What the paper offers a journal (honest profile)

| Aspect | Assessment |
| --- | --- |
| Method novelty | **Low–moderate.** A U-Net with a pretrained ResNet34 encoder is a standard design. |
| Evaluation rigor | **Strong for this data.** Patient-level splits, duplicate removal across the two public datasets, a separately reported "no copies" subset, patient-bootstrap CIs, Bland–Altman, calibration, ablations, failure cases shown. |
| Practical contribution | **Moderate–good.** Turns a mask into diameters, side and a low-confidence flag; evaluates size and side against expert masks (rare in work on these datasets). |
| Main message | Near-perfect accuracy on the Kaggle split is inflated by duplicates; localization and measurement are the better benchmark. Useful to the field. |
| Weaknesses reviewers will raise | No external test set; public, overlapping datasets; one contrast (T1-CE); 24 test patients (3 pituitary); single seed; **no segmentation baseline trained on the same split** (a classic U-Net run was stopped by the system for low memory after 1 of 12 epochs); mm scale estimated for JPEG input. |

The paper is therefore best matched to journals that judge **methodological soundness and usefulness**
rather than methodological novelty.

## 2. Springer Nature journals compared

| Rank | Journal | Fit | Why | Main risk |
| --- | --- | --- | --- | --- |
| **1** | **BMC Medical Imaging** (BMC, Springer Nature) | **Best** | Scope covers image-processing tools for diagnosing and managing disease, including the nervous system; editorial decisions are stated not to depend on perceived interest or impact, which suits a sound evaluation paper. Format (structured abstract, Declarations, Vancouver references) is already applied. | Open-access article processing charge (country-tiered; a third-party copy of the journal page lists £2,290 / $3,090 / €2,590 — confirm the amount and any waiver for Bangladesh on the official page). |
| 2 | Journal of Imaging Informatics in Medicine (Springer, SIIM; formerly Journal of Digital Imaging) | Good | Covers AI, deep learning and computer-aided diagnosis in imaging; the measurement/report pipeline fits imaging informatics. Subscription route possible without a charge. | Reviewers expect clinical relevance and, usually, external or clinical-workflow validation. |
| 3 | Scientific Reports (Nature Portfolio) | Fair–good | Accepts technically sound work regardless of novelty; broad readership. | High charge; many similar submissions; external validation strongly preferred. |
| 4 | Medical & Biological Engineering & Computing (Springer) | Fair | Biomedical engineering journal that publishes image-analysis methods. | Scope page not verified; method novelty expected. |
| 5 | SN Computer Science (Springer) | Fair (fallback) | Broad applied computer science. | Lower visibility in medical imaging; scope page not verified. |
| — | Neural Computing and Applications | **Not recommended** | Expects a new neural-computing method; this paper contributes evaluation and measurement, not a new network. | High desk-reject risk. |

**Recommendation: submit to BMC Medical Imaging first; if rejected, Journal of Imaging Informatics in
Medicine; then Scientific Reports.** Submit to one journal at a time.

## 3. Readiness for BMC Medical Imaging

| Requirement (BMC research article) | Status |
| --- | --- |
| Structured abstract: Background, Methods, Results, Conclusions | Done (322 words; BMC research articles allow up to 350 — confirm on the official page) |
| Keywords | Done (7) |
| Sections: Background, Methods, Results, Discussion, Conclusions | Done |
| Abbreviations list | Done |
| Declarations: ethics, consent for publication, data and materials, competing interests, funding, authors' contributions, acknowledgements | Done — **confirm "no funding" and "no competing interests" are true** |
| Data availability with public links | Done (figshare, Kaggle, GitHub — repository is public) |
| Reporting checklist | CLAIM followed; attach a filled CLAIM checklist as supplementary file (not yet filled) |
| Figures as separate files | `figures/*.pdf` and `*.png` (300 dpi) |
| Cover letter | `cover_letter.md` |

Before submitting:
1. Check the official BMC Medical Imaging pages (submission guidelines, fees, waiver eligibility) — they
   could not be fetched automatically (login redirect).
2. Decide on the corresponding e-mail (the paper uses debobrato095@gmail.com; an institutional address is
   often preferred) and add co-authors/supervisor if any.
3. Fill the CLAIM checklist as a supplementary file.
4. Check the references marked `VERIFY` in `references.bib` (Kaggle dataset DOI, ANSUR report number).
5. **Strongly recommended before submission:** train a segmentation baseline on the same split
   (`python src/train_unet_baseline.py --epochs 12`, about 3 h on this CPU; close other programs first —
   it needs ~2.5 GB free RAM). When `runs/seg2d_paper/unet_baseline_test.json` exists, rebuilding the paper
   adds the baseline column, sentences and abstract text automatically.
6. Optional but valuable: an external test set with masks (e.g. slices from a different center), and nnU-Net.

Sources: [BMC Medical Imaging — aims and scope](https://bmcmedimaging.biomedcentral.com/submission-guidelines/aims-and-scope),
[BMC Medical Imaging about page (mirror)](https://biomedcentral-bmcmedimaging.publicaciones.saludcastillayleon.es/about),
[Journal of Imaging Informatics in Medicine — SIIM](https://siim.org/resources/publications/journal-of-imaging-informatics),
[Springer journal 10278](https://www.springer.com/10278),
[Medical & Biological Engineering & Computing (Editage listing)](https://editage.com/journal/medical-biological-engineering-computing).
