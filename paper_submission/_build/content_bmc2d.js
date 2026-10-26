// 2D brain tumor paper for BMC Medical Imaging (research article).
// Single source for the Word (build_docx.js bmcsub / bmc), LaTeX (build_tex.js) and supplementary files.
// Every number is read from the evaluation outputs, never typed in by hand.
// BMC rules applied: no thousands separators, URLs only as references (except the required
// data-availability wording), figure/table titles of at most 15 words with separate legends,
// additional files cited in sequence and listed with name, format, title and description.
const fs = require('fs');
const path = require('path');
const ROOT = path.resolve(__dirname, '..', '..');
const M = JSON.parse(fs.readFileSync(path.join(ROOT, 'runs', 'seg2d_paper', 'metrics.json'), 'utf8'));
const SPLIT = JSON.parse(fs.readFileSync(path.join(ROOT, 'runs', 'seg2d', 'split.json'), 'utf8'));
// The U-Net baseline (src/train_unet_baseline.py) is included only when its test results exist;
// without them every mention of the baseline is left out of the paper.
const UNET_FILE = path.join(ROOT, 'runs', 'seg2d_paper', 'unet_baseline_test.json');
const UNET = fs.existsSync(UNET_FILE) ? JSON.parse(fs.readFileSync(UNET_FILE, 'utf8'))
  : { test_dice_mean: NaN, test_dice_median: NaN, 'test_dice_ge_0.5': NaN, parameters: 7849025, epoch: '' };

const f = (x, d = 3) => (Number.isNaN(Number(x)) ? '{TBD}' : Number(x).toFixed(d));
const pc = (x, d = 1) => (Number.isNaN(Number(x)) ? '{TBD}' : (100 * x).toFixed(d));
const ci = (a, d = 3) => `${f(a[0], d)}–${f(a[1], d)}`;
const pci = (a, d = 1) => `${pc(a[0], d)}–${pc(a[1], d)}`;
const int = (n) => String(n);   // BMC: no commas in numbers

const S = M.figshare_test_segmentation;
const MS = M.figshare_test_measurement;
const K = M.kaggle_test;
const pt = S.per_type;
const ab = S.ablation_tta_threshold;
const nFsPat = Object.fromEntries(Object.entries(SPLIT.figshare_patients).map(([k, v]) => [k, v.length]));
const nFs = SPLIT.figshare_slices;
const nKgTrainTotal = SPLIT.kaggle_train + SPLIT.kaggle_val + SPLIT.kaggle_removed_as_figshare_val_test_duplicates;
const det = K.detection_seg2d, det3 = K.detection_3d_detector, detMean = K.detection_mean_of_both;
const detC = K.detection_seg2d_clean, det3C = K.detection_3d_detector_clean;
const nTumor = det.tp + det.fn, nNone = det.tn + det.fp;
const params = (M.model.parameters / 1e6).toFixed(1);
const unetParams = (UNET.parameters / 1e6).toFixed(1);

const better = S.dice.mean > UNET.test_dice_mean;
const baselineDone = !Number.isNaN(UNET.test_dice_mean);
const BASELINE_RESULT = !baselineDone ? '' : better
  ? `The classic U-Net baseline reached a lower mean Dice of ${f(UNET.test_dice_mean)} (median ${f(UNET.test_dice_median)}).`
  : `The classic U-Net baseline reached a mean Dice of ${f(UNET.test_dice_mean)} (median ${f(UNET.test_dice_median)}), not lower than the proposed model.`;
const BASELINE_SENTENCE = !baselineDone ? '' : better
  ? `It also outperformed a randomly initialized U-Net trained on the same masks (mean Dice ${f(UNET.test_dice_mean)} versus ${f(S.dice.mean)}); because the baseline differs in initialization, encoder and the extra classification data at once, this comparison does not isolate which factor helps.`
  : `A randomly initialized U-Net trained on the same masks did as well or better (mean Dice ${f(UNET.test_dice_mean)} versus ${f(S.dice.mean)}), so the advantage of the proposed model lies in classification and measurement rather than in segmentation accuracy.`;

const title = 'Segmentation, type classification and measurement of brain tumors on single 2D MRI slices: a retrospective development and leakage-controlled internal validation study';

const meta = {
  title,
  running: 'Single-slice brain tumor segmentation and measurement',
  articleType: 'Research',
  authors: [{ name: 'Debobrato Biswas', initials: 'DB', affil: [1], corresponding: true, email: 'debobrato095@gmail.com' }],
  affils: ['Department of Computer Science and Engineering, Daffodil International University, Dhaka, Bangladesh'],
  abstract: [
    ['Background', 'Many brain tumor studies on public two-dimensional (2D) magnetic resonance imaging (MRI) collections report near-perfect classification, yet they rarely localize or measure the tumor, and the two most used collections share images, which can leak test images into training. We developed a model that outlines the tumor on a single 2D slice, names its type and reports its size and side, and evaluated it with patient-level, leakage-controlled splits.'],
    ['Methods', `A U-Net with an ImageNet-pretrained ResNet34 encoder and a four-class head (no tumor, glioma, meningioma, pituitary tumor) was trained on ${int(3064)} contrast-enhanced T1-weighted slices with expert masks from ${int(233)} patients (figshare dataset) and on class labels from the Kaggle Brain Tumor MRI Dataset. The figshare data were split by patient, and Kaggle images that copy held-out figshare slices were removed. Post-processing derives the longest and perpendicular diameters, the side of the brain and a low-confidence flag. Segmentation was tested on ${nFsPat.test} unseen patients (${nFs.test} slices)${baselineDone ? ' against a classic U-Net trained identically,' : ''} and classification on ${int(K.n)} Kaggle test images, including a subset without copies of training images.`],
    ['Results', `Mean Dice was ${f(S.dice.mean)} (95% confidence interval ${ci(S.dice.ci95)}; median ${f(S.dice.median)})${baselineDone ? ` versus ${f(UNET.test_dice_mean)} for the U-Net baseline` : ''}: ${f(pt.Meningioma.dice_mean)} for meningioma, ${f(pt['Pituitary tumor'].dice_mean)} for pituitary tumor and ${f(pt.Glioma.dice_mean)} for glioma. The median absolute error of the longest diameter was ${f(MS.longest_median_ae_mm, 1)} mm, and the side agreed with the expert mask on all ${M.side.n_overlapping} axial or coronal slices where the outline overlapped the tumor. Four-class accuracy was ${pc(K.accuracy_4class)}% (95% confidence interval ${pci(K.accuracy_4class_ci)}%) and tumor detection accuracy ${pc(det.accuracy, 2)}%, compared with ${pc(det3.accuracy)}% for a 3D detection network applied to the same images. Detection probabilities were well calibrated (expected calibration error ${f(K.ece_seg2d)}).`],
    ['Conclusions', 'A pretrained U-Net delineates and measures meningiomas and pituitary tumors on single slices with errors of a few millimetres, but subtle gliomas at the skull base remain a failure mode. Classification accuracy on the public Kaggle test split is near ceiling and inflated by duplicated images, so localization and measurement are the more informative targets for these datasets.'],
  ],
  keywords: ['Brain tumor', 'Magnetic resonance imaging', 'Segmentation', 'U-Net', 'Transfer learning', 'Data leakage', 'Tumor measurement'],
};

const body = [
  { h1: 'Background' },
  { p: 'Brain and central nervous system tumors caused an estimated 308000 new cases worldwide in 2020 [[cite:sung2021globocan]]. Their management depends on the tumor type, location and size: the World Health Organization (WHO) classification separates gliomas, meningiomas and pituitary adenomas with very different prognoses [[cite:louis2021who]], and response criteria such as Response Assessment in Neuro-Oncology (RANO) rely on bidimensional diameters measured on contrast-enhanced images [[cite:wen2010rano]]. Manual delineation is slow and varies between raters [[cite:menze2015]], which motivates automated support.' },
  { p: `Deep learning has been applied widely to brain MRI [[cite:litjens2017]]. Two public collections of 2D slices are used particularly often: the figshare dataset of Cheng et al. (${int(3064)} contrast-enhanced T1-weighted slices from ${int(233)} patients with expert tumor masks) [[cite:cheng2015,cheng2016,cheng2017figshare]] and the Kaggle Brain Tumor MRI Dataset, which adds tumor-free images and is organized into training and testing folders [[cite:nickparvar2021kaggle]]. Studies on these data mostly address classification and report accuracies close to 100% [[cite:swati2019,deepak2019]].` },
  { p: 'Two issues limit what such numbers mean. First, a class label alone does not tell a clinician where the tumor is, how large it is or on which side of the brain; these quantities need a segmentation. Second, the Kaggle collection re-uses figshare slices and contains duplicates across its own training and testing folders, and the figshare slices of one patient are highly correlated. Image-level random splits on such data let near-identical images appear in training and test sets, a well-documented source of optimistic results in medical imaging [[cite:varoquaux2022,roberts2021]].' },
  { p: 'This study therefore asks how well a single model can (i) outline the tumor on a single 2D slice, (ii) classify it as no tumor, glioma, meningioma or pituitary tumor, and (iii) report its diameters and side, when evaluated on unseen patients with duplicated images removed. The contributions are:' },
  { ol: [
    `a multi-task U-Net with a pretrained ResNet34 encoder, trained on expert masks and class labels from the two public collections${baselineDone ? ', compared with a classic U-Net trained identically' : ''};`,
    'a leakage-controlled protocol: patient-level splits for the figshare data, removal of Kaggle images that copy held-out figshare slices, and a separate report on the Kaggle test images that do not copy any training image;',
    'post-processing that converts the outline into longest and perpendicular diameters, area, side (radiological convention) and a low-confidence flag, with agreement against expert masks;',
    'released code, split files and per-image results.',
  ] },

  { h1: 'Methods' },
  { h2: 'Study design' },
  { p: 'This is a retrospective development and internal-validation study on public, de-identified data. Reporting follows the Checklist for Artificial Intelligence in Medical Imaging (CLAIM) [[cite:mongan2020claim]] and the relevant items of TRIPOD+AI [[cite:collins2024tripod]]; the completed CLAIM checklist and supplementary tables are given in Additional file 1. Fig. 1 summarizes the system.' },
  { wide: true, figure: { path: 'figures/fig1_pipeline.png', file: 'Figure1',
    title: 'Overview of the segmentation, classification and measurement system',
    caption: 'A 2D slice is converted to a 256 × 256 grayscale input. A U-Net with a pretrained ResNet34 encoder predicts a tumor probability map and four class probabilities. Post-processing restricts the mask to the head, removes small components, flags low-confidence outlines and derives diameters, area, side and depth. Without pixel spacing, the millimetre scale is estimated from the head width.' } },
  { h2: 'Datasets' },
  { p: `**figshare brain tumor dataset** [[cite:cheng2015,cheng2016,cheng2017figshare]]: ${int(3064)} contrast-enhanced T1-weighted slices (axial, coronal and sagittal) from ${int(233)} patients with meningioma (708 slices), glioma (${int(1426)}) or pituitary tumor (930), each with a tumor mask drawn by experts. The dataset documentation states a pixel size of 0.49 mm at 512 × 512 pixels.` },
  { p: `**Kaggle Brain Tumor MRI Dataset** [[cite:nickparvar2021kaggle]]: 2D images in four classes (glioma, meningioma, pituitary, no tumor) with a Training folder (${int(nKgTrainTotal)} images) and a Testing folder (${int(K.n)} images: ${int(nTumor)} with tumor and ${int(nNone)} without). It has no masks and no patient identifiers, and part of its tumor images are taken from figshare.` },
  { h2: 'Data splits and leakage control' },
  { p: `The figshare patients were split at random (seed 42) into training (${nFsPat.train} patients, ${int(nFs.train)} slices), validation (${nFsPat.val} patients, ${nFs.val} slices) and test (${nFsPat.test} patients, ${nFs.test} slices) sets, so no patient contributes to more than one set; the split is listed in Additional file 2. Images were compared after resizing to 32 × 32 pixels and per-image z-scoring; a mean absolute difference below 0.08 was treated as a copy (copies measured below 0.05 and different slices above 0.1). ${SPLIT.kaggle_removed_as_figshare_val_test_duplicates} Kaggle Training images that copy a figshare validation or test slice were removed; of the rest, ${int(SPLIT.kaggle_train)} were used for training and ${SPLIT.kaggle_val} for validation. In the Kaggle Testing folder, ${K.n - K.n_clean} images copy a figshare training slice or a Kaggle Training image; results are reported for all ${int(K.n)} images and for the ${K.n_clean} images without a copy (Additional file 1: Table S2).` },
  { h2: 'Preprocessing' },
  { p: 'Each image was converted to grayscale, zero-padded to a square around its center (so one pixel covers the same distance horizontally and vertically), resized to 256 × 256 pixels and scaled to [0, 1]. The figshare slices were first min–max scaled to 8 bits, as in the public JPEG versions.' },
  { h2: 'Model' },
  { p: `The network is a U-Net [[cite:ronneberger2015]] whose encoder is a ResNet34 [[cite:he2016resnet]] initialized with ImageNet weights [[cite:deng2009imagenet]] (the grayscale input is repeated over three channels and normalized with ImageNet statistics). Four decoder stages upsample bilinearly, concatenate the encoder features of matching resolution and apply two 3 × 3 convolutions with batch normalization and rectified linear units; the segmentation head (3 × 3 convolution with batch normalization, rectified linear unit and a 1 × 1 convolution) outputs one logit per pixel at full resolution. A classification head (dropout 0.3, linear layer) on the globally pooled deepest encoder features outputs four logits. The model has ${params} million parameters.` },
  { h2: 'Training' },
  { p: `Each epoch used all ${int(nFs.train)} figshare training slices and ${int(1500)} Kaggle training images drawn at random. The figshare slices contribute both losses; Kaggle tumor images contribute only the classification loss, and Kaggle no-tumor images contribute both with an empty mask. The loss is binary cross-entropy plus soft Dice [[cite:milletari2016vnet]] (Dice only on images with tumor) plus 0.5 × weighted cross-entropy with label smoothing 0.05. Optimization used AdamW [[cite:loshchilov2019adamw]] (weight decay 10^{−4}), a one-cycle learning-rate schedule with maximum 3 × 10^{−4} [[cite:smith2019onecycle]], batch size 12 and up to 12 epochs. Augmentation comprised horizontal flips, rotation (±15°), scaling (0.85–1.15), translation (±8%), gamma (0.7–1.5), contrast and brightness changes, Gaussian blur and noise. The checkpoint with the highest sum of validation Dice (figshare) and validation accuracy (Kaggle) was kept (epoch ${M.model.epoch}; validation Dice ${f(M.model.val_dice)}). Training ran on a 12-core central processing unit in PyTorch [[cite:paszke2019pytorch]], about 13 min per epoch; all hyper-parameters and software versions are listed in Additional file 1: Table S3.` },
  ...(!baselineDone ? [] : [{ p: `**Baseline.** A classic four-level U-Net [[cite:ronneberger2015]] (base width 32, batch normalization, ${unetParams} million parameters), randomly initialized and trained for segmentation only on the same figshare training slices with the same augmentation, loss and schedule (maximum learning rate 10^{−3}), selected by validation Dice (epoch ${UNET.epoch}).` }]),
  { h2: 'Inference and post-processing' },
  { p: 'At inference the prediction is averaged with that of the horizontally flipped image. The tumor probability is P(tumor) = 1 − P(no tumor) from the classification head, and the type is the most probable class. The head is segmented by Otsu thresholding [[cite:otsu1979]] of the smoothed image (largest component, holes filled). The tumor mask is the probability map thresholded at 0.5, restricted to the head, without components smaller than 25 pixels or 10% of the largest component. The longest diameter is the largest distance between mask pixels (Feret diameter); the perpendicular diameter is the extent orthogonal to it. Area follows from the pixel count, and a volume estimate assumes an ellipsoid whose third axis equals the perpendicular diameter [[cite:kothari1996]].' },
  { p: 'Side is reported in the radiological convention (image left = patient right) from the horizontal offset of the tumor centroid relative to the head midline; offsets within 8% of the half-width are reported as midline. Because left and right are undefined on sagittal slices, the view is checked by the correlation between the head region and its mirror image: below 0.5 the slice is treated as sagittal and no side is given; between 0.5 and 0.7 the side is reported as conditional. JPEG and PNG files carry no pixel spacing; when none is supplied, the millimetre scale is estimated by assuming a head width of 150 mm on axial and coronal slices, close to the adult head breadth in anthropometric surveys [[cite:gordon2014ansur]]. An outline whose mean probability is below 0.8 is flagged as uncertain. A rule-based concern score (size, type, depth, midline involvement, ring pattern) is shown in the software but was not evaluated and is not part of the results.' },
  { h2: 'Evaluation and statistics' },
  { p: 'Segmentation was evaluated on the figshare test patients against the expert masks with the Dice coefficient [[cite:dice1945]], intersection over union (IoU), precision, recall and the 95th-percentile Hausdorff distance (HD95), following current recommendations for segmentation metrics [[cite:maierhein2024metrics]] and using the nominal pixel size of the 256-pixel images (0.98 mm). Diameters were computed identically on predicted and expert masks and compared with Bland–Altman analysis [[cite:bland1986]]. Side agreement was assessed on axial or coronal test slices whose expert-mask centroid lies outside the midline band. Confidence intervals (CI) for segmentation metrics come from 2000 bootstrap resamples of patients [[cite:efron1993bootstrap]], because slices of one patient are not independent; those for classification come from 2000 image-level resamples. Classification and detection were evaluated on the Kaggle Testing folder (accuracy, sensitivity, specificity, area under the receiver operating characteristic curve (AUC), per-class F1) and compared with a 3D convolutional detection network (squeeze-and-excitation 3D U-Net encoder) trained earlier in the same project on the same Kaggle training images. Calibration of P(tumor) was measured with the expected calibration error (ECE) over ten bins [[cite:guo2017calibration]]. Ablations varied the mask threshold (0.3, 0.5, 0.7) and test-time flipping.' },

  { h1: 'Results' },
  { h2: 'Segmentation' },
  { p: `On ${nFs.test} slices from ${nFsPat.test} unseen patients, the mean Dice was ${f(S.dice.mean)} (95% CI ${ci(S.dice.ci95)}) and the median ${f(S.dice.median)} (Table 1). Of the slices, ${pc(S['dice_ge_0.5'])}% reached Dice ≥ 0.5; on ${S['dice_lt_0.1']} slices the outline missed the tumor (Dice < 0.1), and on ${S.missed_slices} no outline was produced. ${BASELINE_RESULT ? BASELINE_RESULT + ' ' : ''}Performance differed by tumor type: meningiomas were delineated best (Dice ${f(pt.Meningioma.dice_mean)}), pituitary tumors well (${f(pt['Pituitary tumor'].dice_mean)}), and gliomas least well (${f(pt.Glioma.dice_mean)}; Table 2). Fig. 2 shows the best, median and worst slice of each type, and the result for every test slice is listed in Additional file 3.` },
  { table: {
    title: 'Segmentation performance on the figshare test patients',
    widths: baselineDone ? [24, 46, 30] : [30, 70],
    header: [['Metric', 'Proposed (pretrained ResNet34 U-Net)', 'U-Net baseline (from scratch)'].slice(0, baselineDone ? 3 : 2)],
    rows: [
      ['Dice', `${f(S.dice.mean)} [${ci(S.dice.ci95)}] (${f(S.dice.median)})`, `${f(UNET.test_dice_mean)} (${f(UNET.test_dice_median)})`],
      ['IoU', `${f(S.iou.mean)} [${ci(S.iou.ci95)}] (${f(S.iou.median)})`, '—'],
      ['Precision', `${f(S.precision.mean)} [${ci(S.precision.ci95)}]`, '—'],
      ['Recall', `${f(S.recall.mean)} [${ci(S.recall.ci95)}]`, '—'],
      ['HD95 (mm)', `${f(S.hd95_mm.mean, 1)} [${ci(S.hd95_mm.ci95, 1)}] (${f(S.hd95_mm.median, 1)})`, '—'],
      ['Slices with Dice ≥ 0.5 (%)', `${pc(S['dice_ge_0.5'])}`, `${pc(UNET['test_dice_ge_0.5'])}`],
      ['Parameters (millions)', params, unetParams],
    ].map((r) => r.slice(0, baselineDone ? 3 : 2)),
    note: `${nFsPat.test} patients, ${nFs.test} slices. Values are means with patient-level bootstrap 95% CI in brackets and medians in parentheses. ` + (baselineDone ? 'Both models use the same post-processing (threshold 0.5, mask restricted to the head). ' : 'Post-processing: threshold 0.5, mask restricted to the head. ') + 'IoU, intersection over union; HD95, 95th-percentile Hausdorff distance (nominal 0.98 mm pixels).',
  } },
  { table: {
    title: 'Segmentation performance by tumor type on the figshare test patients',
    widths: [24, 16, 36, 24],
    header: [['Type', 'Patients / slices', 'Dice mean [95% CI]', 'Median Dice / HD95 (mm)']],
    rows: ['Meningioma', 'Pituitary tumor', 'Glioma'].map((t) => [t, `${pt[t].patients} / ${pt[t].slices}`,
      `${f(pt[t].dice_mean)} [${ci(pt[t].ci95)}]`, `${f(pt[t].dice_median)} / ${f(pt[t].hd95_median, 1)}`]),
    note: 'CI, patient-level bootstrap confidence interval; HD95, 95th-percentile Hausdorff distance.',
  } },
  { wide: true, figure: { path: 'figures/fig3_qualitative.png', file: 'Figure2',
    title: 'Best, median and worst test slice of each tumor type',
    caption: 'Rows show the best (top), median (middle) and worst (bottom) test slice of each tumor type ranked by Dice. Red, expert mask; green, model outline. The worst glioma lies in the posterior fossa and the model outlines the temporal lobe instead; in the worst pituitary case the model outlines tissue above the tumor.' } },
  { h2: 'Size and side' },
  { p: `On the ${MS.n} test slices with an outline, the median absolute error of the longest diameter was ${f(MS.longest_median_ae_mm, 1)} mm (median relative error ${pc(MS.longest_median_rel_error)}%), the mean absolute error ${f(MS.longest_mae_mm, 1)} mm, the bias ${f(MS.longest_bias_mm, 1)} mm and the correlation r = ${f(MS.longest_pearson_r, 2)} (Fig. 3). Restricted to the ${MS.n_overlapping} outlines that overlap the tumor, the bias was ${f(MS.longest_bias_mm_overlapping, 1)} mm with limits of agreement from ${f(MS.longest_loa_mm_overlapping[0], 1)} to ${f(MS.longest_loa_mm_overlapping[1], 1)} mm: most errors are small, but a minority of large errors, mainly irregular gliomas outlined too widely or too narrowly, widens the limits. Tumor area correlated with r = ${f(MS.area_pearson_r, 2)} (median relative error ${pc(MS.area_median_rel_error)}%).` },
  { p: `The side agreed with the expert mask on ${pc(M.side.agreement)}% of ${M.side.n} eligible slices; all disagreements came from outlines that missed the tumor, and agreement was ${pc(M.side.agreement_overlapping, 0)}% on the ${M.side.n_overlapping} slices where the outline overlapped it. With the scale estimated from the head width, the estimated pixel size was ${f(M.scale_estimation.ratio_estimated_to_nominal_p10_p50_p90[1], 2)} times the nominal 0.98 mm in the median (10th–90th percentile ${f(M.scale_estimation.ratio_estimated_to_nominal_p10_p50_p90[0], 2)}–${f(M.scale_estimation.ratio_estimated_to_nominal_p10_p50_p90[2], 2)}; Additional file 1: Table S5); the nominal spacing implies a median head width of about 180 mm, larger than typical adult head breadth, so either the nominal spacing does not hold for all slices or the 150 mm assumption is too small. Sizes from images without pixel spacing should therefore be read as approximate.` },
  { wide: true, figure: { path: 'figures/fig4_diameter_agreement.png', file: 'Figure3',
    title: 'Agreement of the longest tumor diameter between model and expert masks',
    caption: 'Figshare test patients, nominal pixel size. Left: scatter plot of predicted versus expert longest diameter; crosses mark outlines that miss the tumor. Right: Bland–Altman plot for outlines that overlap the tumor; the solid line is the mean difference and the dashed lines are the 95% limits of agreement.' } },
  { h2: 'Classification and detection' },
  { p: `On all ${int(K.n)} Kaggle test images the four-class accuracy was ${pc(K.accuracy_4class)}% (95% CI ${pci(K.accuracy_4class_ci)}%) with macro-F1 ${f(K.macro_f1)} (Fig. 4; per-class results in Additional file 1: Table S4); on the ${K.n_clean} images without a copy in training it was ${pc(K.accuracy_4class_clean)}%. For tumor versus no tumor, the model reached ${pc(det.accuracy, 2)}% accuracy, ${pc(det.sensitivity, 2)}% sensitivity and ${pc(det.specificity, 2)}% specificity (Table 3), with ${det.fn} missed tumor and ${det.fp} false positives. The 3D detection network missed ${det3.fn} tumors (sensitivity ${pc(det3.sensitivity)}%), and averaging both models was worse than the 2D model alone. The type was correct on ${pc(M.figshare_test_type.accuracy)}% of figshare test slices. The prediction for every Kaggle test image is listed in Additional file 4.` },
  { wide: true, table: {
    title: 'Tumor detection on the Kaggle Testing folder',
    widths: [27, 25, 13, 13, 11, 11],
    header: [['Model', 'Accuracy (%)', 'Sens. (%)', 'Spec. (%)', 'AUC', 'ECE']],
    rows: [
      ['Proposed (2D)', `${pc(det.accuracy, 2)} [${pci(det.accuracy_ci, 2)}]`, pc(det.sensitivity, 2), pc(det.specificity, 2), f(det.auc, 4), f(K.ece_seg2d)],
      ['3D detection network', `${pc(det3.accuracy, 2)} [${pci(det3.accuracy_ci, 2)}]`, pc(det3.sensitivity, 2), pc(det3.specificity, 2), f(det3.auc, 4), f(K.ece_3d_detector)],
      ['Mean of both', pc(detMean.accuracy, 2), pc(detMean.sensitivity, 2), pc(detMean.specificity, 2), f(detMean.auc, 4), '—'],
      [`Proposed, no copies (n = ${detC.n})`, pc(detC.accuracy, 2), pc(detC.sensitivity, 2), pc(detC.specificity, 2), f(detC.auc, 4), '—'],
      [`3D network, no copies (n = ${det3C.n})`, pc(det3C.accuracy, 2), pc(det3C.sensitivity, 2), pc(det3C.specificity, 2), f(det3C.auc, 4), '—'],
    ],
    note: `${int(nTumor)} images with tumor and ${int(nNone)} without; image-level bootstrap 95% CI in brackets. "No copies" excludes test images that copy a training image; it contains only ${K.n_clean_per_class['No tumor']} tumor-free images, so its specificity is imprecise. Sens., sensitivity; Spec., specificity; AUC, area under the receiver operating characteristic curve; ECE, expected calibration error.`,
  } },
  { wide: true, figure: { path: 'figures/fig5_confusion.png', file: 'Figure4',
    title: 'Confusion matrices of the classification head',
    caption: 'Left: Kaggle Testing folder, four classes. Right: tumor type on the figshare test patients, which contain no tumor-free slices. Numbers are image counts; shading is the row-normalized proportion.' } },
  { h2: 'Calibration, ablations, flag and run time' },
  { p: `The expected calibration error of P(tumor) was ${f(K.ece_seg2d)} (Fig. 5). Without test-time flipping the mean Dice was ${f(ab['raw_0.5'].dice_mean)} at threshold 0.5 (${f(ab['raw_0.3'].dice_mean)} at 0.3, ${f(ab['raw_0.7'].dice_mean)} at 0.7), and with flipping ${f(ab['tta_0.5'].dice_mean)} (${f(ab['tta_0.3'].dice_mean)} and ${f(ab['tta_0.7'].dice_mean)}; Additional file 1: Table S6): flipping did not improve segmentation and the result is insensitive to the threshold. Of ${M.outline_flag.n} outlines on validation and test slices, ${M.outline_flag.failed_outlines} missed the tumor; the low-confidence flag marked ${M.outline_flag.flagged_failed} of them and ${M.outline_flag.flagged_correct} correct outlines (the threshold was chosen on these slices, so this is descriptive). A full analysis of one image took ${f(M.timing_cpu.median_s, 2)} s (median) on ${M.timing_cpu.threads} processor threads. Fig. 6 shows the output for two test patients.` },
  { wide: true, figure: { path: 'figures/fig6_training_calibration.png', file: 'Figure5',
    title: 'Validation metrics during training and calibration of the tumor probability',
    caption: 'Left: validation Dice (figshare) and validation accuracies during training. Right: reliability diagram of P(tumor) on the Kaggle Testing folder with five confidence bins; the lower panel gives the number of images per bin on a log scale. The low-confidence bins hold few images, so their accuracies are imprecise.' } },
  { wide: true, figure: { path: 'figures/fig7_example_output.png', file: 'Figure6',
    title: 'System output for two unseen test patients',
    caption: 'Each panel shows the outline, the longest diameter, the side, the predicted type and the estimated size, with the scale estimated from the head width. Expert longest diameters measured on the corresponding figshare masks at the same estimated scale are 32.2 mm (a) and 19.9 mm (b).' } },

  { h1: 'Discussion' },
  { p: `A U-Net with a pretrained ResNet34 encoder, trained on fewer than 2500 slices with masks, delineated meningiomas (Dice ${f(pt.Meningioma.dice_mean, 2)}) and pituitary tumors (${f(pt['Pituitary tumor'].dice_mean, 2)}) on unseen patients and measured the longest tumor diameter with a median error of ${f(MS.longest_median_ae_mm, 1)} mm. ${BASELINE_SENTENCE ? BASELINE_SENTENCE + ' ' : ''}The side of the brain was always right when the outline overlapped the tumor, so side errors reduce to segmentation failures, which the low-confidence flag catches only in part.` },
  { p: 'Gliomas were the hardest type. They are infiltrative and often heterogeneous, and several failures in this test set lie at the skull base or in the posterior fossa, where vessels, sinuses and bone also enhance on contrast-enhanced T1-weighted images. The worst cases in Fig. 2 show the typical failure: an enhancing structure elsewhere is outlined instead of the tumor. Expert masks of gliomas also sometimes include the non-enhancing surrounding tissue, whereas the model tends to outline the enhancing part only, which lowers both Dice and the measured diameter.' },
  { p: `Classification results illustrate why localization is the more informative target. Four-class accuracy (${pc(K.accuracy_4class)}%) and detection accuracy (${pc(det.accuracy, 2)}%) are near ceiling, as in prior reports on these data [[cite:swati2019,deepak2019]]. However, ${K.n - K.n_clean} of the ${int(K.n)} Kaggle test images copy a training image, and the remaining subset holds only ${K.n_clean_per_class['No tumor']} tumor-free images. Patient identifiers are missing from the Kaggle data, so near-duplicates from the same patient cannot be excluded completely. Accuracy on this split mainly shows that the model recognizes images from the same sources and should not be read as a measure of clinical performance [[cite:varoquaux2022,roberts2021]]. The segmentation results, by contrast, come from patients never seen in training.` },
  { p: 'Measurements from 2D images have two further limits. Without pixel spacing from the Digital Imaging and Communications in Medicine (DICOM) header the scale must be estimated; the head-width assumption introduces an error of the order of 15%, and the documented spacing of the source data itself appears inconsistent with typical head sizes. A single slice also gives no third dimension, so volumes are ellipsoid approximations and the measured diameter depends on which slice was chosen. Clinical response assessment requires the slice with the largest tumor extent and consistent acquisition [[cite:wen2010rano]].' },
  { h2: 'Limitations' },
  { ol: [
    'Internal validation only: both datasets are public, partly overlapping and acquired at few centers; no external test set was used.',
    'One contrast (contrast-enhanced T1-weighted) and three tumor types; other tumors, other sequences, post-operative scans, photographs of film and screenshots are out of scope.',
    'The figshare test set has 24 patients and the pituitary group only 3, so its confidence interval is wide.',
    baselineDone
      ? 'Single training run and seed; no hyper-parameter search; the baseline is a classic U-Net and not a self-configuring framework such as nnU-Net [[cite:isensee2021nnunet]].'
      : 'Single training run and seed; no hyper-parameter search; no segmentation baseline (for example a classic U-Net or the self-configuring nnU-Net [[cite:isensee2021nnunet]]) was trained on the same split, so the segmentation accuracy is not compared with other methods under identical conditions.',
    'Size estimates without pixel spacing are approximate, and the concern score shown by the software is rule-based and not validated.',
  ] },

  { h1: 'Conclusions' },
  { p: 'On single 2D contrast-enhanced T1-weighted slices, a pretrained U-Net outlines meningiomas and pituitary tumors close to expert masks and reports their diameter and side with small errors, while subtle gliomas at the skull base remain a failure mode. With leakage controlled, near-perfect classification on the public Kaggle split adds little information; localization and measurement on patient-level splits are the more meaningful benchmarks. External validation on DICOM data with known pixel spacing is the next step before any clinical use.' },
];

const abbreviations = [
  ['2D/3D', 'two-/three-dimensional'], ['AUC', 'area under the receiver operating characteristic curve'],
  ['CI', 'confidence interval'], ['CLAIM', 'Checklist for Artificial Intelligence in Medical Imaging'],
  ['DICOM', 'Digital Imaging and Communications in Medicine'], ['ECE', 'expected calibration error'],
  ['HD95', '95th-percentile Hausdorff distance'], ['IoU', 'intersection over union'],
  ['MRI', 'magnetic resonance imaging'], ['RANO', 'Response Assessment in Neuro-Oncology'],
  ['WHO', 'World Health Organization'],
];

const declarations = [
  ['Ethics approval and consent to participate', 'Not applicable. The study used only publicly available, de-identified datasets (figshare brain tumor dataset; Kaggle Brain Tumor MRI Dataset) released by their creators for research; no new data were collected from humans.'],
  ['Consent for publication', 'Not applicable.'],
  ['Availability of data and materials', 'The datasets supporting the conclusions of this article are available in the figshare repository, https://doi.org/10.6084/m9.figshare.1512427 [[cite:cheng2017figshare]], and in the Kaggle repository, https://doi.org/10.34740/KAGGLE/DSV/2645886 [[cite:nickparvar2021kaggle]]. The patient-level split, the per-slice segmentation results and the per-image classification results supporting the conclusions of this article are included within its additional files (Additional files 2–4). Software: project name, BrainTumorDetection; project home page, https://github.com/Debobrato06/BrainTumorDetection [[cite:biswas2026repo]]; archived version, {TBD: Zenodo DOI of the release}; operating system, platform independent (tested on Windows 11); programming language, Python 3.12; other requirements, PyTorch 2.5.1, torchvision 0.20.1, NumPy 2.3.5, SciPy 1.17.0, scikit-image 0.26.0, OpenCV 4.13; license, {TBD: e.g. MIT}; restrictions to use by non-academics, none. The trained model weights are available from the corresponding author on request.'],
  ['Competing interests', 'The author declares no competing interests.'],
  ['Funding', 'No funding was received for this study or for the publication fees.'],
  ["Authors' contributions", 'DB conceived and designed the study, implemented the software, performed the experiments and analysis, and wrote and revised the manuscript. The author read and approved the final manuscript.'],
  ['Acknowledgements', 'The author thanks Jun Cheng and colleagues and Msoud Nickparvar for making their datasets publicly available.'],
];

const additionalFiles = [
  { name: 'Additional file 1', file: 'Additional_file_1.pdf', format: '.pdf',
    title: 'Supplementary tables and CLAIM checklist',
    description: 'Table S1, completed CLAIM checklist with the location of each item; Table S2, dataset partitions and leakage control; Table S3, training hyper-parameters and software versions; Table S4, per-class classification metrics; Table S5, measurement, side and scale-estimation details; Table S6, threshold and test-time flipping ablation.' },
  { name: 'Additional file 2', file: 'Additional_file_2.csv', format: '.csv',
    title: 'Patient-level split of the figshare dataset',
    description: 'One row per figshare patient identifier with its partition (training, validation or test) and number of slices.' },
  { name: 'Additional file 3', file: 'Additional_file_3.csv', format: '.csv',
    title: 'Segmentation and measurement results for every figshare test slice',
    description: 'For each test slice: file name, patient identifier, true and predicted type, Dice, intersection over union, precision, recall, 95th-percentile Hausdorff distance, expert and predicted diameters and areas, and side offsets.' },
  { name: 'Additional file 4', file: 'Additional_file_4.csv', format: '.csv',
    title: 'Classification results for every Kaggle test image',
    description: 'For each Kaggle Testing image: file path, true and predicted class, tumor probability of the proposed model and of the 3D detection network, copy-of-training-image flag and the four class probabilities.' },
];

// Back matter in the order and wording of the Springer Nature LaTeX template (sn-article.tex):
// \bmhead{Supplementary information}, \bmhead{Acknowledgements}, then "Declarations" as an itemized list.
const decl = Object.fromEntries(declarations);
const sn = {
  affils: [{ orgdiv: 'Department of Computer Science and Engineering', orgname: 'Daffodil International University', city: 'Dhaka', country: 'Bangladesh' }],
  supplementary: additionalFiles.map((a) => `${a.name} (${a.format}): ${a.title}.`).join(' '),
  acknowledgements: decl.Acknowledgements,
  declarations: [
    ['Funding', decl.Funding],
    ['Competing interests', decl['Competing interests']],
    ['Ethics approval and consent to participate', decl['Ethics approval and consent to participate']],
    ['Consent for publication', decl['Consent for publication']],
    ['Data availability', 'The datasets supporting the conclusions of this article are available in the figshare repository, https://doi.org/10.6084/m9.figshare.1512427 [[cite:cheng2017figshare]], and in the Kaggle repository, https://doi.org/10.34740/KAGGLE/DSV/2645886 [[cite:nickparvar2021kaggle]]. The patient-level split, the per-slice segmentation results and the per-image classification results are included in the supplementary information (Additional files 2–4).'],
    ['Materials availability', 'Not applicable.'],
    ['Code availability', 'The source code is available at https://github.com/Debobrato06/BrainTumorDetection [[cite:biswas2026repo]] (Python 3.12; PyTorch 2.5.1, torchvision 0.20.1, NumPy 2.3.5, SciPy 1.17.0, scikit-image 0.26.0, OpenCV 4.13; platform independent; archived version {TBD: Zenodo DOI of the release}; license {TBD: e.g. MIT}). The trained model weights are available from the corresponding author on request.'],
    ['Author contribution', decl["Authors' contributions"]],
  ],
};

module.exports = { title, meta, body, abbreviations, declarations, additionalFiles, sn, M, SPLIT, K, S, MS, f, pc, ci, pci };
