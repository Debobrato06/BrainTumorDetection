// IEEE J-BHI version. Mirrors ../2_IEEE_JBHI/manuscript.tex.
const title = 'BrainHybridNet: A Lightweight Hybrid CNN–Transformer for Joint Brain Tumor Segmentation, Detection and Grading with Self-Supervised Pre-training and Uncertainty Estimation';

function front(runs, Paragraph, TextRun, AlignmentType) {
  const T = (text, size, o = {}) => new Paragraph({ children: runs(text, { size, ...(o.run || {}) }), alignment: o.align || AlignmentType.CENTER,
    spacing: { after: o.after ?? 80, line: o.line || 240 } });
  return [
    T('DRAFT — every red [TBD] must be filled from experiment outputs before submission', 16, { run: { italics: true, color: 'C00000' } }),
    T(title, 44, { after: 200, line: 260 }),
    T('Debobrato Biswas, {TBD: IEEE membership or remove}, {TBD: co-author names}', 22, { after: 200 }),
    new Paragraph({ children: [new TextRun({ text: 'Abstract', bold: true, italics: true, font: 'Times New Roman', size: 18 }),
      new TextRun({ text: '—', bold: true, font: 'Times New Roman', size: 18 }),
      ...runs('Objective: Automated brain tumor analysis from magnetic resonance images usually requires separate models for segmentation, detection and grading, and most report no measure of confidence. We present a single lightweight model that performs all three tasks and estimates its own uncertainty. Methods: The proposed network combines a two-stage convolutional stem, which preserves high-resolution detail, with a four-layer transformer that models long-range volumetric context. In one forward pass it segments the whole tumor, tumor core and enhancing tumor, detects tumor presence and predicts low- versus high-grade glioma. The encoder is pre-trained by reconstructing masked regions of unlabeled scans; repeated stochastic inference with dropout yields a voxel-wise uncertainty map, and transformer attention provides spatial explanations. Five established networks were re-trained on identical patient-level splits of the 2021 Brain Tumor Segmentation challenge data (1,251 patients) for comparison. Results: On the held-out test set the model reaches Dice scores of {TBD}, {TBD} and {TBD} for whole tumor, tumor core and enhancing tumor, with {TBD} million parameters and {TBD} milliseconds per volume crop; {TBD: grading area under the curve, calibration error and external-validation result}. Conclusion: {TBD: measured trade-off between accuracy, model size and calibration}. Significance: A compact model that segments, detects and grades brain tumors while flagging uncertain regions can support radiologist review on standard clinical hardware.', { size: 18, bold: true })],
      alignment: AlignmentType.JUSTIFIED, spacing: { after: 100, line: 230 } }),
    new Paragraph({ children: [new TextRun({ text: 'Index Terms', bold: true, italics: true, font: 'Times New Roman', size: 18 }),
      ...runs('—Brain tumor segmentation, glioma grading, magnetic resonance imaging, masked image modeling, transformer, uncertainty estimation.', { size: 18, bold: true })],
      alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 } }),
    T('D. Biswas is with the Department of Computer Science and Engineering, Daffodil International University, Dhaka, Bangladesh (e-mail: {TBD: institutional e-mail}). {TBD: co-author affiliations}. Manuscript received {TBD}.', 16, { align: AlignmentType.LEFT, after: 160 }),
  ];
}

const body = [
  { h1: 'Introduction' },
  { p: 'Glioblastoma (WHO grade 4) has a median overall survival of 14.6 months under standard treatment [[cite:stupp2005]], and about 308,000 new brain and central nervous system tumors were estimated worldwide in 2020 [[cite:sung2021globocan]]. Delineating the whole tumor (WT), tumor core (TC) and enhancing tumor (ET) on multi-modal MRI (T1, contrast-enhanced T1ce, T2, FLAIR) [[cite:gordillo2013]] guides surgery, radiotherapy and response assessment, but manual delineation is slow and raters agree only to Dice 0.74–0.85 [[cite:menze2015]].' },
  { p: 'CNNs such as 3D U-Net [[cite:ronneberger2015,cicek2016]] capture local structure but not long-range context; transformers [[cite:vaswani2017,dosovitskiy2021]] do, at high data and compute cost. Hybrid designs [[cite:chen2021transunet,wang2021transbts,wu2023dformer]] combine both, but most solve one task, give point predictions only [[cite:gal2016]] and have tens of millions of parameters [[cite:ker2018]].' },
  { p: 'We propose BrainHybridNet. Our contributions are:' },
  { ul: [
    'a lightweight hybrid encoder (CNN stem with high-resolution skips, four-layer transformer bottleneck) with {TBD} million parameters;',
    '3D masked image modeling applied before the CNN stem of a hybrid encoder, extending ViT self pre-training on target data [[cite:zhou2023selfmae]], with measured effects on accuracy and label efficiency;',
    'joint segmentation, detection and LGG/HGG grading in one pass;',
    'Monte Carlo dropout uncertainty, evaluated for calibration and correlation with error, and attention maps evaluated against reference masks;',
    'a reproducible comparison with five re-trained baselines on identical splits, with confidence intervals, paired tests and external validation; code, splits and weights are public.',
  ] },

  { h1: 'Related Work' },
  { p: '*Segmentation networks.* 3D U-Net [[cite:cicek2016]], SegResNet [[cite:myronenko2019]] and the self-configuring nnU-Net [[cite:isensee2021nnunet]] are strong CNN baselines. UNETR [[cite:hatamizadeh2022unetr]] and SwinUNETR [[cite:tang2022swinunetr,liu2021swin]] use transformer encoders; TransBTS [[cite:wang2021transbts]], MedT [[cite:valanarasu2021medt]], UCTransNet [[cite:wang2022uctransnet]] and D-Former [[cite:wu2023dformer]] mix convolution and attention.' },
  { p: '*Self-supervised learning.* Contrastive learning [[cite:chen2020simclr,caron2021dino,sowrirajan2021mococxr]] and masked image modeling [[cite:he2022mae,xie2022simmim]] reduce label needs; MAE self pre-training improves BraTS segmentation [[cite:zhou2023selfmae]].' },
  { p: '*Uncertainty.* Bayesian networks [[cite:neal1996]], deep ensembles [[cite:lakshminarayanan2017]] and Monte Carlo dropout [[cite:gal2016]] estimate epistemic uncertainty; in neuroimaging it has been linked to errors [[cite:nair2020]] and used for quality control [[cite:roy2018]].' },
  { p: '*Grading.* WHO CNS5 grading uses histology and molecular markers [[cite:louis2021who]]; imaging-based grading uses radiomics [[cite:zhou2018radiomics]] or CNNs [[cite:chang2018idh]], usually separate from segmentation.' },

  { h1: 'Method' },
  { wide: true, figure: { path: 'figures/fig1_architecture.png', caption: 'BrainHybridNet. Sequence attention and a two-stage CNN stem give skip features c_{1}, c_{2}; 512 tokens from 8^{3} patches of c_{2} pass four transformer blocks. Pooled tokens feed detection and grade heads; the decoder fuses c_{2} and c_{1} before the WT/TC/ET head. Monte Carlo dropout gives uncertainty, last-block attention a saliency map.' } },
  { h2: 'Architecture' },
  { p: 'Fig. 1 shows the network. Input **X** ∈ ℝ^{B×4×D×H×W} (four z-scored sequences, 128^{3} crops) is re-weighted per sequence by squeeze-and-excitation and passed through two stem stages: *c*_{1} = ReLU(BN(Conv_{3,1}(**X**))) (32 channels, full resolution) and *c*_{2} = ReLU(BN(Conv_{3,2}(*c*_{1}))) (64 channels, half resolution). A strided convolution turns 8^{3} patches of *c*_{2} into *N* = 512 tokens of size *E* = 128 with learned positional embeddings **P**, followed by *L* = 4 pre-norm transformer blocks (4 heads, MLP ratio 4, GELU [[cite:hendrycks2016gelu]], dropout 0.1) [[cite:ba2016layernorm]]:' },
  { eq: '*z*′_{l} = *z*_{l} + MHSA(LN(*z*_{l})),  *z*_{l+1} = *z*′_{l} + MLP(LN(*z*′_{l}))' },
  { p: 'The decoder reshapes tokens to an 8^{3} grid, applies 3D dropout (0.1), and upsamples four times with transposed convolutions and residual squeeze-and-excitation (ResSE) blocks, concatenating *c*_{2} and *c*_{1} at matching resolutions. Heads: a 1×1×1 convolution with sigmoids for WT/TC/ET; detection Linear(E,128) → ReLU → Linear(128,2) and grading Linear(E,64) → ReLU → Linear(64,2) on mean-pooled tokens. As BraTS has no tumor-free subjects, detection is trained per crop (positive if ≥100 WT voxels); subjects without grade labels are ignored by the grading loss.' },
  { h2: 'Masked image modeling' },
  { p: 'Following SimMIM [[cite:xie2022simmim]] in 3D, 75% of token positions are masked; their 16^{3} input regions are zeroed before the stem (so nothing leaks through convolutions) and the tokens replaced by a learned mask embedding. A linear head reconstructs the 4×16^{3} voxels of each masked region; the loss is the mean squared error to per-patch normalized targets over masked patches only. Pre-training uses only the training folds.' },
  { h2: 'Uncertainty' },
  { p: 'With batch normalization frozen and dropout active [[cite:gal2016]], *T* = 20 sliding-window passes give the mean μ and voxel-wise variance σ^{2} = (1/T) Σ_{t} (σ(Ŷ^{(t)}) − μ)^{2}.' },
  { h2: 'Training' },
  { p: 'ℒ = 0.6 ℒ_{seg} + 0.2 ℒ_{det} + 0.2 ℒ_{grade}, with cross-entropy for detection and grading and the Tversky loss [[cite:salehi2017tversky]] (α = 0.3, β = 0.7) per region for segmentation. AdamW [[cite:loshchilov2019adamw]] (learning rate 10^{−4}, weight decay 10^{−4}, clipping 1.0), 10 warm-up epochs then cosine decay [[cite:loshchilov2017sgdr]], 300 epochs (pre-training 100 epochs at 1.5 × 10^{−4}), batch 2, mixed precision; flips, intensity scaling/shift and Gaussian noise; crops centered on tumor with probability 0.66; best mean validation Dice per fold.' },

  { h1: 'Experimental Setup' },
  { p: '*Data.* BraTS 2021 [[cite:baid2021brats]]: 1,251 subjects, four co-registered skull-stripped sequences at 1 mm^{3}; WT, TC and ET from the expert labels. A subject-level hold-out test set (15%, {TBD} subjects) is used once; the rest forms five folds shared by all models. Grading uses UCSF-PDGM [[cite:calabrese2022ucsfpdgm]] (WHO grade), which also serves as an external segmentation test set.' },
  { p: '*Baselines.* 3D U-Net, SegResNet, UNETR, SwinUNETR (MONAI [[cite:cardoso2022monai]]) and nnU-Net v2, all re-trained on the same folds and preprocessing.' },
  { p: '*Metrics.* Dice and HD95 per region (BraTS conventions for empty masks); AUC, balanced accuracy, sensitivity and specificity; voxel-level expected calibration error [[cite:guo2017calibration]]; Spearman correlation of case uncertainty with 1 − Dice; parameters, MACs, latency and memory on {TBD: GPU}. Mean ± SD, 95% bootstrap intervals, Wilcoxon tests with Holm correction. Reporting follows CLAIM [[cite:mongan2020claim]]. Implementation: PyTorch [[cite:paszke2019pytorch]], PyTorch Lightning [[cite:falcon2019lightning]].' },

  { h1: 'Results' },
  { wide: true, table: {
    caption: 'Segmentation on the BraTS 2021 Hold-Out Test Set ({TBD} Subjects), Mean ± SD',
    widths: [24, 8, 8, 8, 8, 8, 8, 9, 8, 11],
    header: [[{ t: 'Method' }, { t: 'Dice ↑', span: 3 }, { t: 'HD95 (mm) ↓', span: 3 }, { t: 'Params (M)' }, { t: 'GMACs' }, { t: 'Latency (ms)' }],
             ['', 'WT', 'TC', 'ET', 'WT', 'TC', 'ET', '', '', '']],
    rows: ['3D U-Net [[cite:cicek2016]]', 'SegResNet [[cite:myronenko2019]]', 'UNETR [[cite:hatamizadeh2022unetr]]', 'SwinUNETR [[cite:tang2022swinunetr]]', 'nnU-Net v2 [[cite:isensee2021nnunet]]', 'BrainHybridNet']
      .map((m) => [m, ...Array(9).fill('{TBD}')]),
  } },
  { table: {
    caption: 'Detection (BraTS Crops) and Grading (UCSF-PDGM)',
    widths: [30, 14, 14, 14, 14, 14],
    header: [['', 'Det. AUC', 'Sens.', 'Spec.', 'Grade AUC', 'Grade BA']],
    rows: [['BrainHybridNet', '{TBD}', '{TBD}', '{TBD}', '{TBD}', '{TBD}']],
  } },
  { p: '*Segmentation.* Table I; significance versus the strongest baseline: {TBD}. Five-fold cross-validation: {TBD}.' },
  { p: '*Detection and grading.* Table II.' },
  { p: '*Calibration and uncertainty.* {TBD: ECE with/without Monte Carlo dropout; Spearman ρ and p}.' },
  { p: '*External validation.* {TBD: UCSF-PDGM without retraining}.' },
  { table: {
    caption: 'Ablations (5-Fold Cross-Validation, Dice)',
    widths: [40, 15, 15, 15, 15],
    header: [['Configuration', 'WT', 'TC', 'ET', 'Params']],
    rows: ['CNN only', 'No CNN skips', 'No sequence attention', 'Segmentation only', 'No pre-training', 'Full model']
      .map((m) => [m, '{TBD}', '{TBD}', '{TBD}', '{TBD}']),
  } },
  { p: '*Ablations.* Table III; masking ratio, depth and loss-weight sensitivity: {TBD}.' },

  { h1: 'Discussion' },
  { p: '{TBD: interpret only the measured results: where the hybrid helps, the effect of pre-training with 10%/25% labels, whether uncertainty tracks error, attention-map overlap}.' },
  { p: '*Limitations.* External validation is limited to {TBD}; grading ignores molecular markers [[cite:louis2021who]]; pre-training uses only BraTS training folds; detection is evaluated on crops, not healthy scans; reference labels vary between raters [[cite:menze2015]].' },

  { h1: 'Conclusion' },
  { p: 'BrainHybridNet jointly segments, detects and grades brain tumors with a compact hybrid encoder, masked-image-modeling pre-training and uncertainty estimates. {TBD: one-sentence measured summary}. Code, splits and weights: https://github.com/Debobrato06/BrainTumorDetection {TBD: release tag}.' },
  { h: 'Acknowledgment' },
  { p: '{TBD: or remove}.' },
  { h: 'Data Availability' },
  { p: 'BraTS 2021 (https://www.synapse.org/#!Synapse:syn27046444) and UCSF-PDGM (The Cancer Imaging Archive) are public; this study used only these de-identified datasets, so no ethics approval was required.' },
];

module.exports = { title, front, body };
