Dear Editor,

I am submitting the research article **"Brain tumor segmentation, type classification and size and laterality estimation from single 2D MRI slices: a leakage-controlled, patient-level evaluation"** for consideration in *BMC Medical Imaging*.

Public two-dimensional brain MRI collections are widely used to train tumor classifiers, and many studies report near-perfect accuracy. These results rarely say where the tumor is or how large it is, and the two most used collections share images, so random splits can place copies of test images in training. This study builds one model that outlines the tumor, names its type and reports its diameters and side, and evaluates it with patient-level splits after removing duplicated images.

Main findings:
- On unseen patients, mean Dice was 0.750 (95% CI 0.657–0.838); meningiomas 0.942, pituitary tumors 0.801, gliomas 0.634.
- The longest tumor diameter had a median absolute error of 3.6 mm, and the side of the brain agreed with the expert mask whenever the outline overlapped the tumor.
- Classification accuracy on the public Kaggle test split was 98.9%, but 621 of its 1,311 images copy a training image; the paper reports both figures and argues that localization and measurement are the more informative benchmarks.

The work fits the journal's scope of image-processing tools for the diagnosis and management of disease. It uses only public, de-identified data; code and the patient-level split are publicly available. The manuscript is original, has not been published and is not under consideration elsewhere. The author declares no competing interests.

Suggested reviewers: [to be added — 3–4 researchers in brain MRI segmentation from other institutions, with e-mail addresses].

Sincerely,
Debobrato Biswas
Department of Computer Science and Engineering, Daffodil International University, Dhaka, Bangladesh
debobrato095@gmail.com
