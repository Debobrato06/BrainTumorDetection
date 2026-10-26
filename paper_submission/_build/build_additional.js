// Builds Additional file 1 (supplementary tables S1-S6 and the CLAIM checklist) for the BMC submission.
// Every number comes from the same evaluation outputs as the manuscript (content_bmc2d.js).
// Usage: node build_additional.js <out.docx> [jiim]
//   jiim: Online Resource 1 for the Journal of Imaging Informatics in Medicine: JIIM section names,
//   "Online Resource" naming, no author details (double-blind review).
const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, BorderStyle, AlignmentType, Footer, PageNumber,
} = require('docx');
const C = require('./content_bmc2d.js');

const { M, K, S, MS, f, pc, ci } = C;
const SPLIT = C.SPLIT;
const FONT = 'Times New Roman';
const W = 9026;
const OUT = process.argv[2];
const JIIM = process.argv[3] === 'jiim';
// Section names and file names of the JIIM manuscript; reference [32] (the code repository) is not
// in the blinded reference list
const jiim = (s) => !JIIM ? s : s
  .replace('Abstract (Background, Methods, Results, Conclusions)', 'Abstract')
  .replace(/Background/g, 'Introduction').replace(/Methods:/g, 'Materials and Methods:')
  .replace(/Declarations: /g, 'Title page, Statements and Declarations: ').replace('Availability of data and materials', 'title page (Code availability)')
  .replace('source code repository [32]', 'source code repository (title page)')
  .replace(/Additional files (\d)-(\d)/g, 'Online Resources $1-$2').replace(/Additional file (\d)/g, 'Online Resource $1');

const T = (text, o = {}) => new TextRun({ text: String(text), font: FONT, size: o.size || 20, bold: o.bold, italics: o.italics });
const P = (children, o = {}) => new Paragraph({ children: Array.isArray(children) ? children : [T(children, o)], spacing: { after: o.after ?? 120, line: 276 }, alignment: o.align || AlignmentType.LEFT });
const line = { style: BorderStyle.SINGLE, size: 6, color: '000000' };
const none = { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' };

function table(title, header, rows, widths, legend) {
  const sum = widths.reduce((a, b) => a + b, 0);
  const w = widths.map((x) => Math.floor(W * x / sum));
  w[w.length - 1] += W - w.reduce((a, b) => a + b, 0);
  const mk = (cells, head, first, last) => new TableRow({ tableHeader: head, children: cells.map((c, i) => new TableCell({
    width: { size: w[i], type: WidthType.DXA }, margins: { top: 30, bottom: 30, left: 60, right: 60 },
    borders: { top: first ? line : none, bottom: last ? line : none, left: none, right: none },
    children: [new Paragraph({ children: [T(c, { size: 17, bold: head })], alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.LEFT })] })) });
  rows = rows.map((r) => r.map(jiim));
  legend = jiim(legend || '');
  const out = [P([T(title.split(' ')[0] + ' ' + title.split(' ')[1] + ' ', { bold: true }), T(title.split(' ').slice(2).join(' '), { bold: true })], { after: 80 }),
    new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: w, rows: [mk(header, true, true, false), ...rows.map((r, i) => mk(r, false, false, i === rows.length - 1))] })];
  out.push(P([T(legend || '', { size: 16 })], { after: 280 }));
  return out;
}

// ── Table S1: CLAIM checklist (Mongan et al. 2020), location of each item in the manuscript ──
const claim = [
  ['1', 'Identification as a study of AI methodology', 'Title; Abstract'],
  ['2', 'Structured summary', 'Abstract (Background, Methods, Results, Conclusions)'],
  ['3', 'Scientific and clinical background, intended use', 'Background'],
  ['4', 'Study objectives and hypotheses', 'Background, last paragraph'],
  ['5', 'Prospective or retrospective study', 'Methods: Study design (retrospective)'],
  ['6', 'Study goal (model creation, exploratory, feasibility, non-inferiority)', 'Methods: Study design (development and internal validation)'],
  ['7', 'Data sources', 'Methods: Datasets'],
  ['8', 'Eligibility criteria', 'Methods: Datasets (all slices of both public datasets used)'],
  ['9', 'Data pre-processing', 'Methods: Preprocessing'],
  ['10', 'Selection of data subsets', 'Methods: Data splits and leakage control; Table S2'],
  ['11', 'Definitions of data elements', 'Methods: Datasets; Additional files 2-4 (column definitions)'],
  ['12', 'De-identification methods', 'Declarations: Ethics (public de-identified data)'],
  ['13', 'How missing data were handled', 'Not applicable: no missing images or masks in figshare; Kaggle has no masks by design (Methods: Training)'],
  ['14', 'Definition of ground truth reference standard', 'Methods: Datasets (expert masks; dataset class labels)'],
  ['15', 'Rationale for choosing the reference standard', 'Methods: Datasets; Discussion (expert masks and their extent)'],
  ['16', 'Source of ground-truth annotations; qualifications of annotators', 'Methods: Datasets (annotations by the dataset creators [6-8]; qualifications as reported there)'],
  ['17', 'Annotation tools', 'Not applicable: masks taken from the public dataset'],
  ['18', 'Measurement of inter- and intrarater variability', 'Not available for the public masks; stated as limitation of reference standards (Background [4])'],
  ['19', 'Intended sample size and how it was determined', 'Methods: Data splits (all available data; no sample-size calculation)'],
  ['20', 'How data were assigned to partitions; proportions', 'Methods: Data splits and leakage control; Additional file 2'],
  ['21', 'Level at which partitions are disjoint', 'Methods: Data splits (patient level for figshare; image-level copy removal for Kaggle)'],
  ['22', 'Detailed description of model', 'Methods: Model'],
  ['23', 'Software libraries, frameworks and packages', 'Methods: Training; Table S3; Availability of data and materials'],
  ['24', 'Initialization of model parameters', 'Methods: Model (ImageNet weights for the encoder)'],
  ['25', 'Details of training approach (augmentation, hyper-parameters, stopping)', 'Methods: Training; Table S3'],
  ['26', 'Method of selecting the final model', 'Methods: Training (best validation score)'],
  ['27', 'Ensembling techniques', 'Methods: Inference (horizontal-flip averaging only); Table S6'],
  ['28', 'Metrics of model performance', 'Methods: Evaluation and statistics'],
  ['29', 'Statistical measures of significance and uncertainty', 'Methods: Evaluation and statistics (bootstrap 95% CI)'],
  ['30', 'Robustness or sensitivity analysis', 'Results: Calibration, ablations; Table S6'],
  ['31', 'Methods for explainability or interpretability', 'Results: Fig. 2 and Fig. 6 (outlines shown with the prediction)'],
  ['32', 'Validation or testing on external data', 'Not performed; stated in Limitations'],
  ['33', 'Flow of participants or cases', 'Methods: Data splits; Table S2'],
  ['34', 'Demographic and clinical characteristics of cases', 'Not available in the public datasets; tumor-type composition in Methods: Datasets'],
  ['35', 'Performance metrics for optimal model on all data partitions', 'Results; Tables 1-3; Fig. 5 (validation curves)'],
  ['36', 'Estimates of diagnostic accuracy and their precision', 'Results; Table 3 (95% CI)'],
  ['37', 'Failure analysis of incorrectly classified cases', 'Results: Fig. 2 (worst cases); Discussion; Additional files 3-4'],
  ['38', 'Study limitations', 'Discussion: Limitations'],
  ['39', 'Implications for practice', 'Discussion; Conclusions'],
  ['40', 'Registration number and name of registry', 'Not applicable (retrospective study on public data)'],
  ['41', 'Where the full study protocol can be accessed', 'Methods; source code repository [32]'],
  ['42', 'Sources of funding and other support', 'Declarations: Funding'],
];

// ── Table S2: partitions ──
const nPat = Object.fromEntries(Object.entries(SPLIT.figshare_patients).map(([k, v]) => [k, v.length]));
const s2 = [
  ['figshare training', `${nPat.train} patients / ${SPLIT.figshare_slices.train} slices`, 'Segmentation and type losses'],
  ['figshare validation', `${nPat.val} patients / ${SPLIT.figshare_slices.val} slices`, 'Model selection'],
  ['figshare test', `${nPat.test} patients / ${SPLIT.figshare_slices.test} slices`, 'Segmentation, measurement, side and type evaluation'],
  ['Kaggle Training, removed', `${SPLIT.kaggle_removed_as_figshare_val_test_duplicates} images`, 'Copies of figshare validation or test slices'],
  ['Kaggle training subset', `${SPLIT.kaggle_train} images`, 'Classification loss (no-tumor images also segmentation loss)'],
  ['Kaggle validation subset', `${SPLIT.kaggle_val} images`, 'Model selection'],
  ['Kaggle Testing, all', `${K.n} images`, 'Classification and detection evaluation'],
  ['Kaggle Testing, copies of a training image', `${K.n - K.n_clean} images`, 'Reported separately'],
  ['Kaggle Testing, no copies', `${K.n_clean} images (no tumor ${K.n_clean_per_class['No tumor']}, glioma ${K.n_clean_per_class.Glioma}, meningioma ${K.n_clean_per_class.Meningioma}, pituitary ${K.n_clean_per_class['Pituitary tumor']})`, 'Leakage-controlled subset'],
];

// ── Table S3: training set-up ──
const s3 = [
  ['Input', '256 × 256 grayscale, zero-padded to square, scaled to [0, 1]'],
  ['Architecture', `U-Net decoder, ResNet34 encoder (ImageNet weights), ${(M.model.parameters / 1e6).toFixed(1)} million parameters`],
  ['Heads', 'Segmentation (1 channel, sigmoid); classification (4 classes, dropout 0.3)'],
  ['Loss', 'BCE + soft Dice (segmentation) + 0.5 × weighted cross-entropy, label smoothing 0.05; class weights 1, 1, 1.2, 1'],
  ['Optimizer / schedule', 'AdamW, weight decay 1e-4; one-cycle, maximum learning rate 3e-4, 10% warm-up'],
  ['Batch size / epochs', '12 / up to 12 (selected epoch ' + M.model.epoch + ')'],
  ['Data per epoch', 'All figshare training slices + 1500 random Kaggle training images'],
  ['Augmentation', 'Horizontal flip p = 0.5; rotation ±15°; scale 0.85–1.15; translation ±8%; gamma 0.7–1.5; contrast ×0.85–1.15; brightness ±0.08; Gaussian blur p = 0.2; Gaussian noise p = 0.3'],
  ['Selection criterion', 'Validation Dice (figshare) + validation accuracy (Kaggle)'],
  ['Inference', 'Average with horizontally flipped input; mask threshold 0.5'],
  ['Hardware', '12-core CPU, no GPU; about 13 min per epoch'],
  ['Software', 'Python 3.12.10; PyTorch 2.5.1; torchvision 0.20.1; NumPy 2.3.5; SciPy 1.17.0; scikit-image 0.26.0; OpenCV 4.13.0; scikit-learn 1.8.0'],
  ['Random seed', 'Patient split 42; training 0'],
];

// ── Table S4: per-class classification ──
const s4 = Object.entries(K.per_class).map(([c, v]) => [c, String(v.n), f(v.precision), f(v.recall), f(v.f1)]);
const cm = K.confusion_rows_true_cols_pred;
const cmText = C.meta ? ['No tumor', 'Glioma', 'Meningioma', 'Pituitary tumor'].map((c, i) => `${c}: ${cm[i].join(' / ')}`).join('; ') : '';

// ── Table S5: measurement details ──
const sr = M.scale_estimation.ratio_estimated_to_nominal_p10_p50_p90;
const s5 = [
  ['Longest diameter, all outlines (n = ' + MS.n + ')', `median absolute error ${f(MS.longest_median_ae_mm, 1)} mm; mean ${f(MS.longest_mae_mm, 1)} mm; bias ${f(MS.longest_bias_mm, 1)} mm; 95% limits ${f(MS.longest_loa_mm[0], 1)} to ${f(MS.longest_loa_mm[1], 1)} mm; r = ${f(MS.longest_pearson_r, 2)}`],
  ['Longest diameter, overlapping outlines (n = ' + MS.n_overlapping + ')', `median absolute error ${f(MS.longest_median_ae_mm_overlapping, 1)} mm; mean ${f(MS.longest_mae_mm_overlapping, 1)} mm; bias ${f(MS.longest_bias_mm_overlapping, 1)} mm; 95% limits ${f(MS.longest_loa_mm_overlapping[0], 1)} to ${f(MS.longest_loa_mm_overlapping[1], 1)} mm; r = ${f(MS.longest_pearson_r_overlapping, 2)}`],
  ['Tumor area', `r = ${f(MS.area_pearson_r, 2)}; median relative error ${pc(MS.area_median_rel_error)}%`],
  ['Side agreement', `${pc(M.side.agreement)}% of ${M.side.n} eligible slices; ${pc(M.side.agreement_overlapping, 0)}% of ${M.side.n_overlapping} slices with overlapping outline`],
  ['Scale estimate / nominal spacing', `ratio 10th, 50th, 90th percentile ${f(sr[0], 2)}, ${f(sr[1], 2)}, ${f(sr[2], 2)}`],
  ['Low-confidence flag (mean probability < 0.8)', `${M.outline_flag.failed_outlines} of ${M.outline_flag.n} validation and test outlines missed the tumor; flagged ${M.outline_flag.flagged_failed} of these and ${M.outline_flag.flagged_correct} correct outlines`],
  ['Segmentation failures', `${S['dice_lt_0.1']} slices with Dice < 0.1; ${S.missed_slices} slices without outline`],
  ['Run time', `${f(M.timing_cpu.median_s, 2)} s median (90th percentile ${f(M.timing_cpu.p90_s, 2)} s) per image on ${M.timing_cpu.threads} CPU threads`],
];

// ── Table S6: ablation ──
const ab = S.ablation_tta_threshold;
const s6 = [0.3, 0.5, 0.7].map((t) => [String(t), f(ab[`raw_${t}`].dice_mean), f(ab[`raw_${t}`].dice_median), f(ab[`tta_${t}`].dice_mean), f(ab[`tta_${t}`].dice_median)]);

const children = [
  P([T(JIIM ? 'Online Resource 1' : 'Additional file 1', { bold: true, size: 28 })], { after: 60 }),
  P([T('Supplementary tables and CLAIM checklist', { bold: true, size: 24 })], { after: 60 }),
  P([T(C.title, { italics: true })], { after: 60 }),
  JIIM ? P('Journal of Imaging Informatics in Medicine. Author details are given on the title page (double-blind review).', { after: 300 })
    : P('Debobrato Biswas, Department of Computer Science and Engineering, Daffodil International University, Dhaka, Bangladesh. E-mail: debobrato095@gmail.com', { after: 300 }),
  ...table('Table S1 Completed CLAIM checklist', ['Item', 'Topic', 'Where reported'], claim, [6, 44, 50],
    'CLAIM, Checklist for Artificial Intelligence in Medical Imaging (Mongan et al., Radiol Artif Intell 2020). Section names refer to the main manuscript; numbers in brackets are its references. Item wording is abbreviated; the official checklist should be consulted for the full text.'),
  ...table('Table S2 Dataset partitions and leakage control', ['Partition', 'Size', 'Use'], s2, [30, 40, 30],
    'Copies were identified by comparing 32 × 32 z-scored thumbnails (mean absolute difference < 0.08). The patient-level split is listed in Additional file 2.'),
  ...table('Table S3 Training set-up and software', ['Item', 'Setting'], s3, [26, 74],
    'BCE, binary cross-entropy; CPU, central processing unit.'),
  ...table('Table S4 Per-class classification metrics on the Kaggle Testing folder', ['Class', 'Images', 'Precision', 'Recall', 'F1'], s4, [34, 14, 17, 17, 18],
    `Four-class accuracy ${pc(K.accuracy_4class)}% (95% CI ${pc(K.accuracy_4class_ci[0])}–${pc(K.accuracy_4class_ci[1])}%), macro-F1 ${f(K.macro_f1)}. Confusion matrix rows (true class: predicted no tumor / glioma / meningioma / pituitary) — ${cmText}.`),
  ...table('Table S5 Measurement, side and scale-estimation details', ['Quantity', 'Result'], s5, [34, 66],
    'Diameters were measured on predicted and expert masks with the nominal 0.98 mm pixel size of the 256-pixel figshare images. Limits are Bland–Altman 95% limits of agreement.'),
  ...table('Table S6 Effect of mask threshold and test-time flipping on Dice', ['Threshold', 'Dice mean, no flipping', 'Dice median, no flipping', 'Dice mean, flipping', 'Dice median, flipping'], s6, [16, 21, 21, 21, 21],
    `figshare test patients (${nPat.test} patients, ${SPLIT.figshare_slices.test} slices). The manuscript reports the setting with flipping and threshold 0.5.`),
];

const doc = new Document({ creator: JIIM ? 'Anonymous' : 'Debobrato Biswas', title: JIIM ? 'Online Resource 1' : 'Additional file 1', styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] }) },
    children }] });
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log('wrote', OUT); });
