Debobrato Biswas
Department of Computer Science and Engineering, Daffodil International University, Dhaka, Bangladesh
debobrato095@gmail.com

To the Editor-in-Chief, *Journal of Imaging Informatics in Medicine*

Dear Editor,

I submit the original research article **"Segmentation, type classification and measurement of brain tumors on single 2D MRI slices: a retrospective development and leakage-controlled internal validation study"** for consideration in the *Journal of Imaging Informatics in Medicine*.

**Fit with the journal.** The manuscript develops and evaluates an image-analysis tool that turns a single brain MRI slice into a structured result: the tumor outline, its type, its diameters and its side. Public two-dimensional brain MRI collections are widely used to train tumor classifiers, and near-perfect accuracies are often reported. Such results rarely say where a tumor is or how large it is, and the two most used collections share images. The study therefore addresses both the informatics question of producing a usable, quantitative output and the methodological question of evaluating it without data leakage. Main findings:

- On 24 unseen patients, mean Dice was 0.750 (95% confidence interval 0.657–0.838): 0.942 for meningioma, 0.801 for pituitary tumor and 0.634 for glioma.
- The longest tumor diameter had a median absolute error of 3.6 mm, and the side of the brain agreed with the expert mask whenever the outline overlapped the tumor.
- Classification accuracy on the public Kaggle test split was 98.9%, but 621 of its 1311 images copy a training image. The manuscript reports both figures and argues that localization and measurement are the more informative benchmarks.

**Policies.** The study used only publicly available, de-identified datasets released for research (figshare brain tumor dataset; Kaggle Brain Tumor MRI Dataset), so ethics approval and consent were not required. Both datasets are cited with persistent identifiers. Reporting follows the CLAIM checklist, supplied as Online Resource 1. The source code is publicly available with a user guide, and the patient-level split and all per-image results are provided as Online Resources 2–4. The use of a large language model during code writing and manuscript drafting is described in Materials and Methods. A blinded manuscript and a separate title page are provided for double-blind review.

**Competing interests.** The author has no relevant financial or non-financial interests to disclose.

**Originality.** I am the sole author and have approved the manuscript for submission. Its content has not been published and is not under consideration for publication elsewhere.

Yours sincerely,
Debobrato Biswas
