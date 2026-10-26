Debobrato Biswas
Department of Computer Science and Engineering, Daffodil International University, Dhaka, Bangladesh
debobrato095@gmail.com

To the Editor, *BMC Medical Imaging*

Dear Editor,

I submit the research article **"Segmentation, type classification and measurement of brain tumors on single 2D MRI slices: a retrospective development and leakage-controlled internal validation study"** for consideration in *BMC Medical Imaging*.

**Why the manuscript suits BMC Medical Imaging.** The study develops and evaluates an image-processing tool for the diagnosis and management of brain tumors (neuroimaging), which lies within the journal's scope. Public two-dimensional brain MRI collections are widely used to train tumor classifiers, and near-perfect accuracies are often reported; such results rarely say where a tumor is or how large it is, and the two most used collections share images. The manuscript evaluates one model that outlines the tumor, names its type and reports its diameters and side, using patient-level splits after duplicated images were removed. Main findings:

- On 24 unseen patients, mean Dice was 0.750 (95% confidence interval 0.657–0.838): 0.942 for meningioma, 0.801 for pituitary tumor and 0.634 for glioma.
- The longest tumor diameter had a median absolute error of 3.6 mm, and the side of the brain agreed with the expert mask whenever the outline overlapped the tumor.
- Classification accuracy on the public Kaggle test split was 98.9%, but 621 of its 1311 images copy a training image; the manuscript reports both figures and argues that localization and measurement are the more informative benchmarks.

As the journal assesses scientific validity rather than perceived impact, I believe a carefully controlled evaluation of this kind, including its failure cases and limitations, will be useful to its readers.

**Journal policies.** The study used only publicly available, de-identified datasets released for research (figshare brain tumor dataset; Kaggle Brain Tumor MRI Dataset), so ethics approval and consent were not required; both datasets are cited with persistent identifiers. Code is publicly available, and the patient-level split and all per-image results are provided as additional files. Reporting follows the CLAIM checklist, supplied as Additional file 1. I am not aware of any other policy issues.

**Competing interests.** The author declares no competing interests.

**Approval and originality.** I am the sole author and have approved the manuscript for submission. Its content has not been published and is not submitted for publication elsewhere.

**Suggested peer reviewers.** [To be added by the author: 3–4 researchers in brain MRI segmentation from other institutions, with institutional e-mail addresses and ORCID or Scopus IDs.]

**Reviewers to exclude.** None.

Yours sincerely,
Debobrato Biswas
