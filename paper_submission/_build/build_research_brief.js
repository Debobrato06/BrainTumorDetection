// Research brief: what is new in the 2D brain tumor system and paper compared with recent work
// (2024-2026), what is not new, and the plan for future work.
// Usage: node build_research_brief.js <out.docx>   (then to_pdf.ps1 for the PDF)
// Our own numbers are read from the evaluation outputs through content_jiim.js.
const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, BorderStyle, AlignmentType,
  Footer, PageNumber, ShadingType, LevelFormat, ExternalHyperlink,
} = require('docx');
const C = require('./content_jiim.js');

const { S, MS, K, M, f, pc, ci } = C;
const pt = S.per_type;
const OUT = process.argv[2];
const FONT = 'Calibri';
const W = 9638;   // A4 text width with 2 cm margins (DXA)
const ACCENT = '1F4E79';

// ─── Helpers ────────────────────────────────────────────────────────────────
// Inline markup: **bold**, *italic*, [n] stays as text
function runs(text, o = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  const push = (t, x = {}) => out.push(new TextRun({ text: t, font: FONT, size: o.size || 21, color: o.color, bold: o.bold, italics: o.italics, ...x }));
  while ((m = re.exec(text))) {
    if (m.index > last) push(text.slice(last, m.index));
    if (m[0].startsWith('**')) push(m[0].slice(2, -2), { bold: true });
    else push(m[0].slice(1, -1), { italics: true });
    last = m.index + m[0].length;
  }
  if (last < text.length) push(text.slice(last));
  return out;
}
const P = (text, o = {}) => new Paragraph({ children: runs(text, o), alignment: o.align || AlignmentType.LEFT,
  spacing: { before: o.before || 0, after: o.after ?? 120, line: 276 } });
const H1 = (text) => new Paragraph({ children: [new TextRun({ text, font: FONT, size: 30, bold: true, color: ACCENT })],
  spacing: { before: 320, after: 120 }, keepNext: true,
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 2 } } });
const H2 = (text) => new Paragraph({ children: [new TextRun({ text, font: FONT, size: 24, bold: true, color: ACCENT })],
  spacing: { before: 200, after: 80 }, keepNext: true });
const bullets = (items) => items.map((t) => new Paragraph({ children: runs(t), numbering: { reference: 'bullets', level: 0 },
  spacing: { after: 60, line: 276 } }));
const numbered = (items, instance) => items.map((t) => new Paragraph({ children: runs(t), numbering: { reference: 'numbers', level: 0, instance },
  spacing: { after: 60, line: 276 } }));

const line = { style: BorderStyle.SINGLE, size: 4, color: 'BFBFBF' };
function table(header, rows, widths, o = {}) {
  const sum = widths.reduce((a, b) => a + b, 0);
  const w = widths.map((x) => Math.floor(W * x / sum));
  w[w.length - 1] += W - w.reduce((a, b) => a + b, 0);
  const cell = (t, i, head, fill) => new TableCell({
    width: { size: w[i], type: WidthType.DXA }, margins: { top: 50, bottom: 50, left: 90, right: 90 },
    shading: head ? { type: ShadingType.CLEAR, fill: ACCENT, color: 'auto' } : fill ? { type: ShadingType.CLEAR, fill, color: 'auto' } : undefined,
    borders: { top: line, bottom: line, left: line, right: line },
    children: [new Paragraph({ children: runs(t, { size: 18, bold: head, color: head ? 'FFFFFF' : undefined }), spacing: { line: 252 } })],
  });
  const tableRows = [new TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, i, true)) }),
    ...rows.map((r, k) => new TableRow({ children: r.map((t, i) => cell(t, i, false, o.highlight?.(r, k) || (k % 2 ? 'F2F6FA' : undefined))) }))];
  return [new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: w, rows: tableRows }), P(o.note || '', { size: 17, italics: true, after: 160 })];
}
function callout(title, text) {
  return new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: [W], rows: [new TableRow({ children: [new TableCell({
    width: { size: W, type: WidthType.DXA }, margins: { top: 120, bottom: 120, left: 180, right: 180 },
    shading: { type: ShadingType.CLEAR, fill: 'EAF1F8', color: 'auto' },
    borders: { top: none, bottom: none, right: none, left: { style: BorderStyle.SINGLE, size: 24, color: ACCENT } },
    children: [P(`**${title}**`, { after: 60 }), ...[].concat(text).map((t) => P(t, { after: 60 }))],
  })] })] });
}
const none = { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' };

// ─── Content ────────────────────────────────────────────────────────────────
const REFS = [
  ['Vangala BP, Guda S, Peddi L, Vangala N. Scores that hold, benchmarks that leak: measuring dataset contamination in public brain-tumor MRI classification. arXiv:2610.00421, 30 Sep 2026.', 'https://arxiv.org/abs/2610.00421'],
  ['Does the quantum layer help? A leakage-aware benchmark of hybrid quantum–classical models for brain tumor MRI classification. Diagnostics 16(21):3485, 2026.', 'https://www.mdpi.com/2075-4418/16/21/3485'],
  ['Complete patient-level leakage among traceable images in a widely used brain tumor MRI benchmark. Research Square preprint rs-11078963, 2026.', 'https://www.researchsquare.com/article/rs-11078963/v1'],
  ['Brain tumor segmentation using multi-scale attention U-Net with EfficientNetB4 encoder for enhanced MRI analysis. Scientific Reports, 2025.', 'https://www.nature.com/articles/s41598-025-94267-9'],
  ['A densely connected deep learning method for joint brain tumor segmentation and classification. Elsevier, 2026.', 'https://www.sciencedirect.com/science/article/pii/S2772442526000481'],
  ['Benchmarking foundation models for tumor segmentation across multiple cancer types (U-Net, Swin UNETR, nnU-Net, MedSAM, Medical SAM 2). Scientific Reports, 2026.', 'https://www.nature.com/articles/s41598-026-71014-2'],
  ['Glioma segmentation on multicenter 2D-based MRI using low-rank adaptation tuning of a foundation model: an external test evaluation. Journal of Imaging Informatics in Medicine, 2026.', 'https://link.springer.com/article/10.1007/s10278-026-02222-5'],
  ['Chang K, Beers AL, Bai HX, et al. Automatic assessment of glioma burden: a deep learning algorithm for fully automated volumetric and bidimensional measurement (AutoRANO). Neuro-Oncology 21(11):1412-1422, 2019.', 'https://academic.oup.com/neuro-oncology/article/21/11/1412/5514498'],
  ['Deep learning automates bidimensional and volumetric tumor burden measurement from MRI in pre- and post-operative glioblastoma patients. Computers in Biology and Medicine, 2023.', 'https://www.sciencedirect.com/science/article/abs/pii/S0010482523000689'],
  ['Review of machine-learning tools for RANO / RANO 2.0 response assessment. Frontiers in Neurology, 2026.', 'https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2026.1797487/full'],
  ['MICCAI 2025 Lighthouse Challenge: Brain Tumor Segmentation (BraTS) cluster of challenges.', 'https://zenodo.org/records/15094823'],
  ['Capellán-Martín D, et al. Adaptable segmentation pipeline for diverse brain tumors with radiomic-guided subtyping and lesion-wise model ensemble (BraTS 2025). arXiv:2512.14648, 2025.', 'https://arxiv.org/abs/2512.14648'],
  ['Domain-adaptive MRI learning model for precision diagnosis of CNS tumors, 2026.', 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12838589/'],
  ['Isensee F, et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods 18:203-211, 2021.', 'https://doi.org/10.1038/s41592-020-01008-z'],
];

const children = [
  new Paragraph({ children: [new TextRun({ text: 'RESEARCH BRIEF', font: FONT, size: 18, bold: true, color: '7F7F7F' })], spacing: { after: 60 } }),
  new Paragraph({ children: [new TextRun({ text: 'Brain tumor detection system and paper: what is new, and what comes next', font: FONT, size: 40, bold: true, color: ACCENT })], spacing: { after: 120 } }),
  P('Single-slice 2D MRI tumor segmentation, type classification and measurement, compared with research published 2024–2026', { italics: true, color: '595959', after: 60 }),
  P('Debobrato Biswas · Daffodil International University · 26 October 2026', { color: '595959', after: 240 }),

  callout('In short', [
    'The system takes **one 2D MRI slice** (JPG or PNG) and returns the tumor **outline, type, size in millimetres and side of the brain**, with a warning when the outline is uncertain. The paper evaluates this **on patients the model never saw**, after removing copied images between the two public datasets.',
    '**What is new:** one model that segments, classifies *and measures*, checked against expert masks on patient-level splits; measurement agreement (Bland–Altman) and side agreement reported; failure cases, calibration and an uncertainty flag reported; runs on a normal CPU in about 0.15 s.',
    '**What is not new:** the network (ResNet34 U-Net) and the observation that the Kaggle and figshare benchmarks leak. In September 2026 another group published a large leakage audit for *classification*, so the paper must cite it and present its own leakage control as applied to *segmentation and measurement*.',
    '**Most important next steps:** an external test set, a fair segmentation baseline (U-Net, nnU-Net), several training seeds, and real pixel spacing from DICOM files.',
  ]),

  H1('1  Where the field stands (2024–2026)'),
  H2('1.1  Classification on public 2D datasets is at the ceiling, and the benchmarks leak'),
  ...bullets([
    'Most studies on the figshare (Cheng) and Kaggle (Nickparvar) datasets report 97–100% accuracy for four-class classification.',
    '**Vangala et al. (arXiv, September 2026)** [1] audited these datasets. In the current Kaggle test split (version with 1600 test images; our study used the earlier 1311-image version), 28.8% of test images (460 of 1600) have a near-copy in training. Of the test images that can be traced to a figshare patient, 95.5% share a patient with training. File headers alone (size, JPEG tables, no anatomy) reach 0.959 balanced accuracy for tumor vs no tumor. Removing leaked images changed accuracy by only −0.009, so high scores mainly show that the task on these data is easy, not that the model is clinically ready.',
    'A second audit of Kaggle version 2 [2] reports a copy in training for 79% of test images. A Research Square preprint [3] reports patient-level leakage for all traceable test images.',
    'All of these audits are about **classification only**. None of them evaluates segmentation, measurement or side.',
  ]),
  H2('1.2  2D segmentation papers report high Dice, often on image-level splits'),
  ...bullets([
    'Recent figshare segmentation papers report Dice of about 0.90–0.93 (for example an attention U-Net with EfficientNetB4 encoder, Scientific Reports 2025 [4]). Many use random 80:20 or 90:10 image splits, so slices of the same patient appear in training and test, which inflates Dice.',
    'Joint segmentation + classification models exist (for example a densely connected network, 2026 [5]), but they rarely report tumor size, side, calibration or failure analysis.',
  ]),
  H2('1.3  Foundation models (SAM, MedSAM, SAM 2)'),
  ...bullets([
    'A 2026 benchmark across five cancer types [6] found that fine-tuned Medical SAM 2 is strongest overall, but on brain tumors (BraTS) the supervised Swin UNETR ranked first. Zero-shot SAM models are clearly weaker on brain MRI.',
    'In our target journal (JIIM, 2026) [7], a five-centre Japanese study fine-tuned SAM and MedSAM with low-rank adaptation (LoRA) on **2D glioma slices** and reached Dice of about 0.91–0.92 on an external test set. This is the closest recent work to ours and should be cited and compared.',
  ]),
  H2('1.4  Automated tumor measurement'),
  ...bullets([
    'AutoRANO (2019) [8] and later glioblastoma work (2023) [9] compute bidimensional RANO diameters and volumes automatically, but from **3D multi-sequence MRI** and for gliomas only.',
    'A 2026 review [10] reports that no machine-learning tool had yet been validated against the updated RANO 2.0 criteria.',
    'Measurement from a **single 2D image without pixel spacing**, as uploaded by a patient or general doctor, has received very little attention.',
  ]),
  H2('1.5  3D challenges and generalization'),
  ...bullets([
    'BraTS 2025 [11] now covers adult glioma, meningioma, meningioma radiotherapy, metastases and paediatric tumors on 3D T1, T1-contrast, T2 and FLAIR. Leading teams report that the choice of backbone has plateaued and that workflow design (ensembles, subtype-aware training) matters more [12].',
    'Models lose reliability on data from other hospitals and scanners because of differences in hardware, protocols and intensities [13]. External validation is now expected by reviewers.',
  ]),

  H1('2  What our system and paper add'),
  P('The table compares typical recent work with this study. Rows marked in blue are where this study goes beyond the typical paper on the same public data.'),
  ...table(['Aspect', 'Typical 2D papers on figshare / Kaggle', 'Leakage audits 2026 [1–3]', 'This study'], [
    ['Output', 'Class label, sometimes a mask', 'Class label', 'Outline + type + longest and perpendicular diameter + area + side + uncertainty flag'],
    ['Data split', 'Random image-level', 'Deduplicated test split', 'Patient-level (figshare) + removal of Kaggle copies of held-out slices + separate copy-free subset'],
    ['Segmentation evaluation', 'Dice, often image-level', 'Not done', `Dice ${f(S.dice.mean)} (95% CI ${ci(S.dice.ci95)}), median ${f(S.dice.median)}, HD95, per tumor type, patient-level bootstrap CIs`],
    ['Measurement accuracy', 'Rarely reported', 'Not done', `Longest diameter vs expert masks: median error ${f(MS.longest_median_ae_mm, 1)} mm; Bland–Altman analysis`],
    ['Side of the brain', 'Not reported', 'Not done', 'Radiological convention; sagittal slices detected and excluded; 100% agreement when the outline overlaps the tumor'],
    ['Reliability', 'Accuracy only', 'Balanced accuracy, calibration recommended', `Calibration (ECE ${f(K.ece_seg2d)}), low-confidence flag, worst-case analysis`],
    ['Reporting standard', 'Rarely', '—', 'CLAIM checklist completed, TRIPOD+AI cited'],
    ['Open material', 'Sometimes code', 'File lists, splits', 'Code, patient split, result for every test image'],
    ['Practical use', 'Notebook', '—', `Web application, CPU only, ${f(M.timing_cpu.median_s, 2)} s per image`],
  ], [16, 24, 20, 40], { highlight: (r, k) => (k <= 4 ? 'DEEAF6' : undefined),
    note: 'CI, confidence interval; HD95, 95th-percentile Hausdorff distance; ECE, expected calibration error.' }),

  H2('2.1  Our main results at a glance'),
  ...table(['Measure', 'Result'], [
    [`Segmentation, ${C.SPLIT.figshare_slices.test} slices from ${C.SPLIT.figshare_patients.test.length} unseen patients`, `Mean Dice ${f(S.dice.mean)}; meningioma ${f(pt.Meningioma.dice_mean)}, pituitary ${f(pt['Pituitary tumor'].dice_mean)}, glioma ${f(pt.Glioma.dice_mean)}; ${pc(S['dice_ge_0.5'])}% of slices with Dice ≥ 0.5`],
    ['Longest diameter', `Median absolute error ${f(MS.longest_median_ae_mm, 1)} mm; 95% limits of agreement ${f(MS.longest_loa_mm_overlapping[0], 1)} to ${f(MS.longest_loa_mm_overlapping[1], 1)} mm (wide, because of a few large glioma errors)`],
    ['Side', `${pc(M.side.agreement)}% agreement overall; 100% when the outline overlaps the tumor`],
    ['Four-class classification (Kaggle test)', `${pc(K.accuracy_4class)}% (all ${K.n} images); ${pc(K.accuracy_4class_clean)}% on the ${K.n_clean} images without a copy in training`],
    ['Tumor detection', `${pc(K.detection_seg2d.accuracy, 2)}%, versus ${pc(K.detection_3d_detector.accuracy)}% for the earlier 3D detection network`],
  ], [35, 65]),

  callout('How to state the contribution (suggested wording for the paper)', [
    '"Leakage in the public brain tumor MRI benchmarks has recently been documented for classification [Vangala et al. 2026]. We extend leakage-controlled evaluation to segmentation and measurement: on patient-level splits, a single model outlines, classifies and measures the tumor on one 2D slice, and we quantify diameter and side agreement against expert masks, which classification-only studies cannot assess."',
    '**Action:** add references [1] and [7] to the JIIM manuscript before submission. Reviewers who know these papers will otherwise question the novelty claim.',
  ]),

  H1('3  Weak points reviewers will raise'),
  ...table(['Weak point', 'Why it matters', 'How to fix it'], [
    ['No external test set', 'Both datasets are public, overlapping and from few centres', 'Test on slices from another source with expert masks, e.g. 2D slices extracted from BraTS (glioma, meningioma) or a local hospital set'],
    ['No segmentation baseline', 'Dice cannot be compared fairly with other methods', 'Finish the classic U-Net run (stopped by low memory) and add a 2D nnU-Net on the same split [14]'],
    ['Small test set', '24 test patients, only 3 with pituitary tumor', 'Patient-level cross-validation (5 folds) over all 233 patients'],
    ['Single seed, no tuning', 'Result may depend on chance', '3 seeds; report mean ± standard deviation'],
    ['Size without pixel spacing', 'Scale estimated from head width, about 15% error', 'Accept DICOM input and read Pixel Spacing; validate on data with known spacing'],
    ['Risk score not validated', 'Could mislead users', 'Keep it out of the paper (already done) or validate it with clinicians'],
    ['Glioma failures', 'Skull-base and posterior-fossa gliomas are missed', 'More glioma data, 2.5D input (neighbouring slices), FLAIR when available'],
  ], [22, 33, 45]),

  H1('4  What we want to bring next (roadmap)'),
  H2('Phase 1 — before or during review (1–2 months)'),
  ...numbered([
    'Cite the 2026 leakage audits and the JIIM LoRA-SAM paper; sharpen the novelty statement as above.',
    'Run the classic U-Net baseline and a 2D nnU-Net on the identical split; add 3 seeds and 5-fold patient-level cross-validation.',
    'Add a header-only baseline (as recommended in [1]) to show how much of the Kaggle accuracy comes from file properties rather than anatomy.',
    'Release the code with a Zenodo DOI and licence; add the deduplicated split files.',
  ], 1),
  H2('Phase 2 — second paper (3–6 months)'),
  ...numbered([
    '**External validation:** extract axial and coronal slices with masks and known pixel spacing from BraTS 2023–2025 (glioma, meningioma) and test segmentation, measurement and side without retraining.',
    '**Foundation model comparison:** fine-tune MedSAM / SAM 2 with LoRA on the same patient-level split and compare accuracy, speed and CPU memory with our model.',
    '**DICOM support with real pixel spacing**, so the reported size in millimetres is exact rather than estimated.',
    '**2.5D input** (the slice plus its neighbours) to reduce glioma failures when a volume is available.',
    '**Robustness to real-world uploads:** test on photographs of film and phone screenshots, which no recent study has evaluated, and add a quality check that rejects unusable images.',
  ], 2),
  H2('Phase 3 — clinical direction (6–18 months)'),
  ...numbered([
    '**Reader study with radiologists in Bangladesh** (with ethics approval): does the outline and measurement save time or change agreement between readers?',
    '**Fix the 3D pipeline:** train the segmentation head of the 3D model on BraTS so that NIfTI volumes also get outlines and volumes.',
    '**RANO / RANO 2.0-style bidimensional measurement** on the slice with the largest tumor extent, compared with radiologist measurements.',
    '**Uncertainty-aware triage:** use calibrated probabilities and the low-confidence flag to send uncertain cases to a specialist.',
    '**Report in Bengali and English** for patients and general doctors, clearly marked as decision support, not diagnosis.',
  ], 3),

  H1('5  Recommended order of work'),
  ...table(['Priority', 'Task', 'Effort', 'Effect on acceptance'], [
    ['1', 'Cite 2026 leakage audits and JIIM LoRA-SAM paper; adjust novelty text', '1 day', 'High: avoids a novelty objection'],
    ['2', 'U-Net and 2D nnU-Net baselines on the same split', '1–2 weeks (CPU)', 'High: reviewers expect a baseline'],
    ['3', 'External test on BraTS-derived 2D slices', '2–3 weeks', 'Very high: answers the main limitation'],
    ['4', 'Seeds and 5-fold cross-validation', '2–4 weeks (CPU)', 'Medium'],
    ['5', 'DICOM pixel spacing and MedSAM comparison', '1–2 months', 'Second paper'],
  ], [10, 50, 18, 22], { note: 'Training runs on this computer take about 13 minutes per epoch and need free memory; close other programs or use a free GPU service (Kaggle, Google Colab) for the baselines and cross-validation.' }),

  H1('References'),
  ...REFS.map(([t, url], i) => new Paragraph({ children: [new TextRun({ text: `[${i + 1}]  ${t} `, font: FONT, size: 18 }),
    new ExternalHyperlink({ link: url, children: [new TextRun({ text: url, font: FONT, size: 18, color: '0563C1', underline: {} })] })],
    indent: { left: 440, hanging: 440 }, spacing: { after: 80, line: 252 } })),
  P('Preprints [1] and [3] have not been peer reviewed. Their numbers are as reported by the authors.', { size: 17, italics: true, before: 120 }),
];

const doc = new Document({
  creator: 'Debobrato Biswas', title: 'Research brief: novelty and future work',
  styles: { default: { document: { run: { font: FONT, size: 21 } } } },
  numbering: { config: [
    { reference: 'bullets', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 400, hanging: 260 } } } }] },
    { reference: 'numbers', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 400, hanging: 300 } } } }] },
  ] },
  sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: 'Research brief · page ', font: FONT, size: 16, color: '7F7F7F' }), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: '7F7F7F' })] })] }) },
    children }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log('wrote', OUT); });
