// 2D brain tumor paper for the Journal of Imaging Informatics in Medicine (Springer, hybrid, double-blind).
// Derived from content_bmc2d.js so text and numbers stay identical; only the journal's rules change:
//   hypothesis-driven research structure (Introduction, Materials and Methods, Results, Discussion,
//   Conclusions), unstructured abstract of 150-250 words, 4-6 keywords, supplementary files cited as
//   "Online Resource", panels labelled (a), (b), use of a large language model documented in Methods,
//   Statements and Declarations and Acknowledgements on the title page.
const B = require('./content_bmc2d.js');
const { M, K, S, MS, SPLIT, f, pc, ci } = B;

const pt = S.per_type;
const det = K.detection_seg2d;
const nTest = SPLIT.figshare_patients.test.length;

const onlineResource = (s) => s
  .replace(/Additional files (\d)–(\d)/g, 'Online Resources $1–$2')
  .replace(/Additional file (\d): Table/g, 'Online Resource $1, Table')
  .replace(/Additional file (\d)/g, 'Online Resource $1');

const abstract = `Public two-dimensional (2D) brain magnetic resonance imaging (MRI) collections are widely used to train tumor classifiers with near-perfect reported accuracy, yet such models rarely localize or measure the tumor, and the two most used collections share images. This study developed a multi-task U-Net with an ImageNet-pretrained ResNet34 encoder that outlines the tumor on a single contrast-enhanced T1-weighted slice, classifies it as no tumor, glioma, meningioma or pituitary tumor, and reports its diameters and side. The model was trained on expert masks from 233 patients (figshare dataset) and on class labels from the Kaggle Brain Tumor MRI Dataset, with patient-level splits and removal of Kaggle images that copy held-out slices. On ${SPLIT.figshare_slices.test} slices from ${nTest} unseen patients, mean Dice was ${f(S.dice.mean)} (95% confidence interval ${ci(S.dice.ci95)}): ${f(pt.Meningioma.dice_mean)} for meningioma, ${f(pt['Pituitary tumor'].dice_mean)} for pituitary tumor and ${f(pt.Glioma.dice_mean)} for glioma. The median absolute error of the longest diameter was ${f(MS.longest_median_ae_mm, 1)} mm, and the side agreed with the expert mask on all ${M.side.n_overlapping} axial or coronal slices where the outline overlapped the tumor. On ${K.n} Kaggle test images, four-class accuracy was ${pc(K.accuracy_4class)}% and tumor detection accuracy ${pc(det.accuracy, 2)}%, but ${K.n - K.n_clean} of these images copy a training image. A pretrained U-Net delineates and measures meningiomas and pituitary tumors on single slices with errors of a few millimetres, whereas gliomas at the skull base remain a failure mode; on these datasets, localization and measurement are more informative benchmarks than classification accuracy.`;

const meta = {
  ...B.meta,
  articleType: 'Original research (hypothesis-driven research)',
  abstract,
  keywords: ['Brain tumor', 'Magnetic resonance imaging', 'Segmentation', 'Deep learning', 'Data leakage', 'Tumor measurement'],
  orcid: '{TBD: 16-digit ORCID}',
  address: 'Department of Computer Science and Engineering, Daffodil International University, Daffodil Smart City, Birulia, Savar, Dhaka 1216, Bangladesh',
};

const HEADINGS = { Background: 'Introduction', Methods: 'Materials and Methods' };
const AI_USE = 'A large language model (Claude, Anthropic) assisted with writing the analysis and figure code and with drafting and language editing of the manuscript. The author reviewed and verified all code, numbers and text and takes full responsibility for the content.';

// Panel-letter captions: Springer asks for (a), (b) labels instead of "left"/"right"
const CAPTIONS = {
  Figure2: 'Panels (a)–(c) show the best, (d)–(f) the median and (g)–(i) the worst test slice of glioma, meningioma and pituitary tumor (columns), ranked by Dice. Red, expert outline (thicker line); green, model outline (thinner line). The worst glioma (g) lies in the posterior fossa and the model outlines the temporal lobe instead; in the worst pituitary case (i) the model outlines tissue above the tumor',
  Figure3: 'Figshare test patients, nominal pixel size. (a) Predicted versus expert longest diameter; crosses mark outlines that miss the tumor and the dashed line is the line of identity. (b) Bland–Altman plot for outlines that overlap the tumor; the solid line is the mean difference (bias) and the dashed lines are the 95% limits of agreement',
  Figure4: '(a) Kaggle Testing folder, four classes. (b) Tumor type on the figshare test patients, which contain no tumor-free slices. Numbers are image counts; shading is the row-normalized proportion',
  Figure5: '(a) Validation Dice (figshare) and validation accuracies during training. (b) Reliability diagram of P(tumor) on the Kaggle Testing folder with five confidence bins; the lower panel gives the number of images per bin on a logarithmic scale. The low-confidence bins hold few images, so their accuracies are imprecise',
  Figure6: 'Each panel shows the outline, the longest diameter, the side, the predicted type and the estimated size, with the scale estimated from the head width. Expert longest diameters measured on the corresponding figshare masks at the same estimated scale are 32.2 mm (a) and 19.9 mm (b)',
};

const body = [];
let inDesign = false;
for (const b of B.body) {
  if (b.h1) body.push({ h1: HEADINGS[b.h1] || b.h1 });
  else if (b.p !== undefined) body.push({ ...b, p: onlineResource(b.p) });
  else if (b.ol) body.push({ ol: b.ol.map(onlineResource) });
  else if (b.figure) {
    const n = Number(b.figure.file.replace('Figure', ''));
    body.push({ ...b, figure: { ...b.figure, path: `../_build/jiim_png/Fig${n}.png`, file: `Fig${n}`,
      caption: CAPTIONS[b.figure.file] || b.figure.caption.replace(/\.$/, '') } });
  } else body.push(b);
  // the statement on language-model use closes the "Study design" subsection
  if (inDesign && b.p !== undefined) { body.push({ p: AI_USE }); inDesign = false; }
  if (b.h2) inDesign = b.h2 === 'Study design';
}

// Online Resources: same files as the BMC additional files, renamed ESM_<n> as Springer asks
const onlineResources = B.additionalFiles.map((a, i) => ({
  ...a, name: `Online Resource ${i + 1}`, file: `ESM_${i + 1}${i ? '.xlsx' : '.pdf'}`, format: i ? '.xlsx' : '.pdf',
  description: onlineResource(a.description),
}));

const decl = Object.fromEntries(B.declarations);
// Title page "Statements and Declarations" (citations written out, since the title page has no reference list)
const statements = [
  ['Funding', 'The author did not receive support from any organization for the submitted work.'],
  ['Competing interests', 'The author has no relevant financial or non-financial interests to disclose.'],
  ['Author contributions', 'Debobrato Biswas conceived and designed the study, implemented the software, performed the experiments and analysis, and wrote and revised the manuscript. The author read and approved the final manuscript.'],
  ['Ethics approval', 'Not applicable. This retrospective study used only publicly available, de-identified datasets released by their creators for research (figshare brain tumor dataset; Kaggle Brain Tumor MRI Dataset); no new data were collected from humans.'],
  ['Consent to participate', 'Not applicable (public, de-identified data).'],
  ['Consent to publish', 'Not applicable. The images shown contain no identifying information.'],
  ['Data availability', 'The datasets analyzed in this study are available in the figshare repository (https://doi.org/10.6084/m9.figshare.1512427) and the Kaggle repository (https://doi.org/10.34740/KAGGLE/DSV/2645886). The patient-level split and the per-image results are provided as Online Resources 2–4.'],
  ['Code availability', 'The source code, with a user guide, is available at https://github.com/Debobrato06/BrainTumorDetection (archived version {TBD: Zenodo DOI of the release}; license {TBD: e.g. MIT}). The trained model weights are available from the corresponding author on request.'],
];
// Same statements at the end of the full (non-blinded) manuscript, there with numbered citations
const statementsCited = statements.map(([h, t]) => [h, t
  .replace('(https://doi.org/10.6084/m9.figshare.1512427)', '(https://doi.org/10.6084/m9.figshare.1512427) [[cite:cheng2017figshare]]')
  .replace('(https://doi.org/10.34740/KAGGLE/DSV/2645886)', '(https://doi.org/10.34740/KAGGLE/DSV/2645886) [[cite:nickparvar2021kaggle]]')
  .replace('https://github.com/Debobrato06/BrainTumorDetection', 'https://github.com/Debobrato06/BrainTumorDetection [[cite:biswas2026repo]]')]);

module.exports = { ...B, meta, body, abstract, onlineResources, statements, statementsCited,
  acknowledgements: decl.Acknowledgements, declarations: undefined, abbreviations: undefined, additionalFiles: undefined };
