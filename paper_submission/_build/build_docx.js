// Builds the Word versions of both submissions from content files that mirror manuscript.tex.
// Usage: node build_docx.js springer|ieee|bmc|bmcsub|jiim|jiimblind|jiimtitle <out.docx>
//   (bmcsub / jiim / jiimblind + env PREVIEW=1: figures shown under their captions)
//   springer: BrainHybridNet, one column (A4);  ieee: BrainHybridNet, IEEE two columns (Letter);
//   bmc: 2D paper, Springer Nature two columns (A4), Vancouver references.
//   bmcsub: 2D paper as a BMC Medical Imaging submission: one column, double spacing, line and page
//   numbers, no page breaks, figure titles and legends after the references, additional-files list.
//   jiim / jiimblind: 2D paper for the Journal of Imaging Informatics in Medicine: one column, 10 pt
//   Times, double spacing, line and page numbers, JIIM reference style, figure captions after the
//   references; jiimblind leaves out every author detail (double-blind review); jiimtitle: title page
//   with Statements and Declarations; jiim2col: the same paper as a two-column Springer Nature
//   journal layout with figures in the text (for reading and preprints, not for submission).
// Inline markup in content strings:
//   [[cite:key1,key2]]  numbered citation     {TBD} / {TBD: note}  red placeholder
//   **bold**  *italic*  _{sub}  ^{sup}
const fs = require('fs');
const path = require('path');
const d = require('docx');
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, BorderStyle,
  AlignmentType, SectionType, TabStopType, Footer, Header, PageNumber, LevelFormat, VerticalAlign,
  LineNumberRestartFormat,
} = d;

const STYLE = process.argv[2];
const OUT = process.argv[3];
const ROOT = path.resolve(__dirname, '..');
const JIIM = ['jiim', 'jiimblind', 'jiimtitle', 'jiim2col'].includes(STYLE);
const JIIM2 = STYLE === 'jiim2col';
const JIIM_SUB = JIIM && !JIIM2;   // the single-column submission files
const BLIND = STYLE === 'jiimblind';
const TITLEPAGE = STYLE === 'jiimtitle';
const content = require(JIIM ? './content_jiim.js' : { ieee: './content_ieee.js', bmc: './content_bmc2d.js', bmcsub: './content_bmc2d.js' }[STYLE] || './content_springer.js');
const FOLDER = path.join(ROOT, JIIM ? '5_JIIM_Submission' : { ieee: '2_IEEE_JBHI', bmc: '3_SpringerNature_BMC_MedImaging', bmcsub: '4_BMC_MedicalImaging_Submission' }[STYLE] || '1_SpringerNature_NCA');
// The JIIM folder holds only files to upload, so the shared bibliography is read from the BMC folder
const BIB = path.join(JIIM ? path.join(ROOT, '4_BMC_MedicalImaging_Submission') : FOLDER, 'references.bib');

const IEEE = STYLE === 'ieee';
const BMCSUB = STYLE === 'bmcsub';
const PREVIEW = (BMCSUB || JIIM_SUB) && process.env.PREVIEW === '1';
const BMC = STYLE === 'bmc' || BMCSUB;   // Vancouver references, structured front matter
const TWO_COL = IEEE || STYLE === 'bmc' || JIIM2;
const FONT = 'Times New Roman';
const BASE = BMCSUB ? 24 : TWO_COL || JIIM ? 20 : 22;   // half-points: 12 pt / 10 pt / 11 pt
// Text widths in DXA: full text width and one column
const FULL_W = IEEE ? 10368 : TWO_COL ? 9746 : 9026;
const COL_W = IEEE ? 4986 : TWO_COL ? 4693 : 9026;
const RED = 'C00000';

// ─── BibTeX ────────────────────────────────────────────────────────────────
function parseBib(text) {
  const entries = {};
  const re = /@(\w+)\s*\{\s*([^,\s]+)\s*,/g;
  let m;
  while ((m = re.exec(text))) {
    const type = m[1].toLowerCase(), key = m[2];
    let i = re.lastIndex, depth = 1;
    const start = i;
    while (i < text.length && depth > 0) {
      if (text[i] === '{') depth++;
      else if (text[i] === '}') depth--;
      i++;
    }
    const body = text.slice(start, i - 1);
    const fields = {};
    const fre = /(\w+)\s*=\s*/g;
    let f;
    while ((f = fre.exec(body))) {
      let j = fre.lastIndex, val = '';
      if (body[j] === '{') {
        let dd = 1; j++;
        const s = j;
        while (j < body.length && dd > 0) { if (body[j] === '{') dd++; else if (body[j] === '}') dd--; j++; }
        val = body.slice(s, j - 1);
      } else {
        const s = j;
        while (j < body.length && body[j] !== ',' && body[j] !== '\n') j++;
        val = body.slice(s, j).trim();
      }
      fields[f[1].toLowerCase()] = val;
      fre.lastIndex = j;
    }
    entries[key] = { type, key, ...fields };
    re.lastIndex = i;
  }
  return entries;
}

function delatex(s) {
  if (!s) return s;
  return s
    .replace(/\\url\{([^}]*)\}/g, '$1')
    .replace(/\{\\c\{C\}\}/g, 'Ç').replace(/\{\\c\{c\}\}/g, 'ç').replace(/\\c\{c\}/g, 'ç')
    .replace(/\{\\"O\}/g, 'Ö').replace(/\{\\"o\}/g, 'ö').replace(/\{\\"u\}/g, 'ü').replace(/\\"u/g, 'ü').replace(/\\"o/g, 'ö')
    .replace(/\{\\"e\}/g, 'ë').replace(/\\"e/g, 'ë')
    .replace(/--/g, '–')
    .replace(/[{}]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function splitAuthors(a) {
  return a.split(/\s+and\s+/).map((x) => x.trim());
}
// A name wrapped in one brace group (e.g. {The PyTorch Lightning team}) is an organization
function isCorporate(n) {
  return n.startsWith('{') && n.endsWith('}') && !n.includes(',');
}
function initials(first) {
  return first.split(/[\s.]+/).filter(Boolean).map((p) => p.split('-').map((q) => q[0] + '.').join('-'));
}
function fmtAuthorsSpringer(a) {
  return splitAuthors(a).map((n) => {
    if (n === 'others') return 'et al.';
    if (isCorporate(n)) return delatex(n);
    const [last, first = ''] = n.split(',').map((x) => x.trim());
    const suffix = first.match(/,?\s*Jr\.?$/) ? ' Jr.' : '';
    return `${delatex(last)}, ${initials(delatex(first.replace(/,?\s*Jr\.?$/, ''))).join('')}${suffix}`;
  }).join(', ').replace(/, et al\.$/, ', et al.');
}
function fmtAuthorsIEEE(a) {
  const list = splitAuthors(a).map((n) => {
    if (n === 'others') return null;
    if (isCorporate(n)) return delatex(n);
    const parts = n.split(',').map((x) => x.trim());
    const last = parts[0], first = parts.slice(1).join(' ').replace(/Jr\.?/, '').trim();
    return `${initials(delatex(first)).join(' ')} ${delatex(last)}`.trim();
  });
  const etal = list.includes(null);
  const names = list.filter(Boolean);
  if (etal) return names.join(', ') + ' et al.';
  if (names.length <= 2) return names.join(' and ');
  return names.slice(0, -1).join(', ') + ', and ' + names[names.length - 1];
}

// Index Medicus / MEDLINE journal abbreviations (BMC reference style)
const MEDLINE = {
  'CA: A Cancer Journal for Clinicians': 'CA Cancer J Clin', 'Neuro-Oncology': 'Neuro Oncol',
  'Journal of Clinical Oncology': 'J Clin Oncol', 'IEEE Transactions on Medical Imaging': 'IEEE Trans Med Imaging',
  'Medical Image Analysis': 'Med Image Anal', 'PLoS ONE': 'PLoS One',
  'Computerized Medical Imaging and Graphics': 'Comput Med Imaging Graph', 'Computers in Biology and Medicine': 'Comput Biol Med',
  'npj Digital Medicine': 'NPJ Digit Med', 'Nature Machine Intelligence': 'Nat Mach Intell',
  'Radiology: Artificial Intelligence': 'Radiol Artif Intell', 'BMJ': 'BMJ', 'Stroke': 'Stroke',
  'Nature Methods': 'Nat Methods', 'Ecology': 'Ecology', 'The Lancet': 'Lancet',
  'IEEE Transactions on Systems, Man, and Cybernetics': 'IEEE Trans Syst Man Cybern',
  'Proceedings of Machine Learning Research': 'Proc Mach Learn Res',
  'International Conference on Learning Representations (ICLR)': 'International Conference on Learning Representations (ICLR)',
};

// Vancouver (BMC): "Sung H, Ferlay J, Siegel RL, et al."
function fmtAuthorsVancouver(a) {
  return splitAuthors(a).map((n) => {
    if (n === 'others') return 'et al';
    if (isCorporate(n)) return delatex(n);
    const parts = n.split(',').map((x) => x.trim());
    const last = parts[0], first = parts.slice(1).join(' ').replace(/Jr\.?/, '').trim();
    return `${delatex(last)} ${initials(delatex(first)).join('').replace(/[.-]/g, '')}`.trim();
  }).join(', ');
}

// Returns an array of [text, italic] segments for one reference.
function formatRef(e) {
  const t = delatex(e.title);
  const doi = e.doi ? `https://doi.org/${e.doi}` : '';
  const note = delatex(e.note || e.howpublished || '');
  const venue = delatex(e.journal || e.booktitle || e.publisher || '');
  if (JIIM) {
    // "Zaidel M, Hopper K, Iyriboz T: Interactive web-based radiology teaching file. J Digit Imaging 12:203-204, 1999"
    const pages = e.pages ? delatex(e.pages).replace(/–/g, '-') : '';
    const segs = [[`${fmtAuthorsVancouver(e.author)}: ${t}. `, false]];
    if (e.type === 'article') {
      segs.push([`${MEDLINE[venue] || venue}${e.volume ? ' ' + e.volume : ''}${pages ? (e.volume ? ':' : ' ') + pages : ''}, ${e.year}`, false]);
    } else if (e.type === 'inproceedings') {
      segs.push([`In: ${venue}${e.series ? ` (${e.series}${e.volume ? ' ' + e.volume : ''})` : e.volume ? ', vol ' + e.volume : ''}. ${e.publisher ? delatex(e.publisher) + ', ' : ''}${e.year}${pages ? ', pp ' + pages : ''}`, false]);
    } else if (e.type === 'book') {
      segs.push([`${delatex(e.address || '')}: ${delatex(e.publisher)}, ${e.year}`, false]);
    } else if (e.type === 'techreport') {
      segs.push([`${delatex(e.institution)}, Report No. ${delatex(e.number)}, ${e.year}`, false]);
    } else {
      segs.push([note.includes('Accessed') ? note : `${note}, ${e.year}`, false]);
    }
    if (note && e.type !== 'misc') segs.push([`. ${note}`, false]);
    if (doi) segs.push([`. ${doi}`, false]);
    return segs;
  }
  if (BMC) {
    const segs = [[`${fmtAuthorsVancouver(e.author)}. ${t}. `, false]];
    if (e.type === 'article') {
      segs.push([MEDLINE[venue] || venue, false], [`. ${e.year}${e.volume ? ';' + e.volume : ''}${e.number ? '(' + delatex(e.number) + ')' : ''}${e.pages ? ':' + delatex(e.pages) : ''}.`, false]);
    } else if (e.type === 'book') {
      segs.push([`${delatex(e.address || '')}: ${delatex(e.publisher)}; ${e.year}.`, false]);
    } else if (e.type === 'inproceedings') {
      segs.push(['In: ', false], [venue, true], [`${e.series ? '. ' + e.series : ''}${e.volume ? ' ' + e.volume : ''}. ${e.publisher ? delatex(e.publisher) + '; ' : ''}${e.year}${e.pages ? '. p. ' + delatex(e.pages) : ''}.`, false]);
    } else if (e.type === 'techreport') {
      segs.push([`${delatex(e.institution)}; ${e.year}. Report No.: ${delatex(e.number)}.`, false]);
    } else {
      segs.push([note.includes('Accessed') ? `${e.year}. ${note}.` : `${note ? note + '; ' : ''}${e.year}.`, false]);
    }
    if (note && e.type !== 'misc') segs.push([` ${note}.`, false]);
    if (doi) segs.push([` ${doi}`, false]);
    return segs;
  }
  if (!IEEE) {
    const au = fmtAuthorsSpringer(e.author);
    const segs = [[`${au}: ${t}. `, false]];
    if (e.type === 'article') {
      segs.push([venue, true]);
      segs.push([`${e.volume ? ' ' + e.volume : ''}${e.number ? '(' + delatex(e.number) + ')' : ''}${e.pages ? ', ' + delatex(e.pages) : ''} (${e.year})`, false]);
    } else if (e.type === 'inproceedings') {
      segs.push(['In: ', false], [venue, true]);
      segs.push([`${e.series ? '. ' + e.series : ''}${e.volume ? ', vol. ' + e.volume : ''}${e.pages ? ', pp. ' + delatex(e.pages) : ''}${e.publisher ? '. ' + e.publisher : ''} (${e.year})`, false]);
    } else if (e.type === 'book') {
      segs.push([`${delatex(e.series || '')}${e.volume ? ', vol. ' + e.volume : ''}. ${delatex(e.publisher)}, ${delatex(e.address || '')} (${e.year})`, false]);
    } else {
      segs.push([`(${e.year})`, false]);
    }
    if (note) segs.push([`. ${note}`, false]);
    if (doi) segs.push([`. ${doi}`, false]);
    return segs;
  }
  const au = fmtAuthorsIEEE(e.author);
  const segs = [[`${au}, “${t},” `, false]];
  if (e.type === 'article') {
    segs.push([venue, true]);
    segs.push([`${e.volume ? ', vol. ' + e.volume : ''}${e.number ? ', no. ' + delatex(e.number) : ''}${e.pages ? ', pp. ' + delatex(e.pages) : ''}, ${e.year}`, false]);
  } else if (e.type === 'inproceedings') {
    segs.push(['in ', false], [venue, true]);
    segs.push([`${e.series && e.volume ? ', ' + e.series + ' vol. ' + e.volume : ''}${e.pages ? ', pp. ' + delatex(e.pages) : ''}, ${e.year}`, false]);
  } else if (e.type === 'book') {
    segs.length = 0;
    segs.push([`${au}, `, false], [t, true], [`. ${delatex(e.address || '')}: ${delatex(e.publisher)}, ${e.year}`, false]);
  } else {
    segs.push([`${e.year}`, false]);
  }
  if (note) segs.push([`, ${note}`, false]);
  if (doi) segs.push([`, doi: ${e.doi}`, false]);
  segs.push(['.', false]);
  return segs;
}

const bib = parseBib(fs.readFileSync(BIB, 'utf8'));
const citeOrder = [];
function citeNums(keys) {
  return keys.split(',').map((k) => {
    k = k.trim();
    if (!bib[k]) throw new Error(`unknown citation ${k}`);
    if (!citeOrder.includes(k)) citeOrder.push(k);
    return citeOrder.indexOf(k) + 1;
  });
}
function citeText(nums) {
  const s = [...nums].sort((a, b) => a - b);
  const parts = [];
  for (let i = 0; i < s.length; i++) {
    let j = i;
    while (j + 1 < s.length && s[j + 1] === s[j] + 1) j++;
    parts.push(j - i >= 2 ? `${s[i]}–${s[j]}` : s.slice(i, j + 1).join(', '));
    i = j;
  }
  return IEEE ? parts.map((p) => `[${p}]`).join(', ').replace(/\], \[/g, '], [') : `[${parts.join(', ')}]`;
}

// ─── Inline markup → TextRuns ──────────────────────────────────────────────
function runs(text, base = {}) {
  const out = [];
  const re = /(\[\[cite:[^\]]+\]\]|\{TBD[^}]*\}|\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]+\}|\^\{[^}]+\})/g;
  let last = 0, m;
  const push = (t, extra = {}) => out.push(new TextRun({ text: t, font: FONT, size: base.size || BASE, ...base, ...extra }));
  while ((m = re.exec(text))) {
    if (m.index > last) push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith('[[cite:')) push(citeText(citeNums(tok.slice(7, -2))));
    else if (tok.startsWith('{TBD')) push(`[${tok.slice(1, -1)}]`, { bold: true, color: RED });
    else if (tok.startsWith('**')) push(tok.slice(2, -2), { bold: true });
    else if (tok.startsWith('*')) push(tok.slice(1, -1), { italics: true });
    else if (tok.startsWith('_{')) push(tok.slice(2, -1), { subScript: true });
    else if (tok.startsWith('^{')) push(tok.slice(2, -1), { superScript: true });
    last = m.index + tok.length;
  }
  if (last < text.length) push(text.slice(last));
  return out;
}

// ─── Blocks ────────────────────────────────────────────────────────────────
const LINE = BMCSUB || JIIM_SUB ? 480 : TWO_COL ? 240 : 300;  // double (BMC submission) / single (two-column) / 1.25 lines
const ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII'];
let h1 = 0, h2 = 0, tableNo = 0, eqNo = 0, figNo = 0;

function para(text, opts = {}) {
  return new Paragraph({
    children: runs(text, opts.run || {}),
    // Justifying a line that holds a long URL stretches the gaps, so such paragraphs are left-aligned
    alignment: opts.align || (text.includes('https://') ? AlignmentType.LEFT : AlignmentType.JUSTIFIED),
    spacing: { line: LINE, after: opts.after ?? (IEEE ? 0 : BMC ? 80 : 120), before: opts.before || 0 },
    indent: opts.indent ?? (IEEE ? { firstLine: 202 } : undefined),
    keepNext: opts.keepNext,
  });
}

function heading1(title) {
  h1++; h2 = 0;
  const label = IEEE ? `${ROMAN[h1 - 1]}. ${title.toUpperCase()}` : BMCSUB || JIIM ? title : `${h1} ${title}`;
  return new Paragraph({
    children: [new TextRun({ text: label, font: FONT, size: IEEE ? BASE : BMC || JIIM ? 24 : 26, bold: !IEEE, smallCaps: false })],
    alignment: IEEE ? AlignmentType.CENTER : AlignmentType.LEFT,
    spacing: { before: IEEE ? 160 : 280, after: IEEE ? 80 : 120, line: BMCSUB || JIIM_SUB ? 480 : undefined },
    keepNext: true,
  });
}
function heading2(title) {
  h2++;
  const label = IEEE ? `${String.fromCharCode(64 + h2)}. ${title}` : BMCSUB || JIIM ? title : `${h1}.${h2} ${title}`;
  return new Paragraph({
    children: [new TextRun({ text: label, font: FONT, size: IEEE ? BASE : BMC || JIIM ? 21 : 23, italics: IEEE, bold: !IEEE })],
    spacing: { before: IEEE ? 100 : 200, after: IEEE ? 40 : 80 },
    keepNext: true,
  });
}
function unnumbered(title) {
  return new Paragraph({
    children: [new TextRun({ text: IEEE ? title.toUpperCase() : title, font: FONT, size: IEEE ? BASE : BMC || JIIM ? 24 : 26, bold: !IEEE })],
    alignment: IEEE ? AlignmentType.CENTER : AlignmentType.LEFT,
    spacing: { before: 240, after: 100 },
    keepNext: true,
  });
}

// Equations: centred text with a right-aligned number
function equation(text, numbered = true) {
  eqNo += numbered ? 1 : 0;
  const width = COL_W;   // one column (two-column layouts) or the page text width
  return new Paragraph({
    tabStops: [{ type: TabStopType.CENTER, position: Math.round(width / 2) }, { type: TabStopType.RIGHT, position: width }],
    children: [new TextRun({ text: '\t', font: FONT, size: BASE }), ...runs(text, { italics: false }),
      ...(numbered ? [new TextRun({ text: `\t(${eqNo})`, font: FONT, size: BASE })] : [])],
    spacing: { before: 80, after: 80, line: 240 },
  });
}

function list(items, ordered) {
  return items.map((t) => new Paragraph({
    children: runs(t),
    numbering: { reference: ordered ? 'numbers' : 'bullets', level: 0, instance: ordered ? listInstance : 0 },
    alignment: t.includes('https://') ? AlignmentType.LEFT : AlignmentType.JUSTIFIED,
    spacing: { line: LINE, after: IEEE ? 0 : 60 },
  }));
}
let listInstance = 0;

const border = { style: BorderStyle.SINGLE, size: 6, color: '000000' };
const none = { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' };

function table(spec, fullWidth) {
  tableNo++;
  const total = fullWidth ? FULL_W : COL_W;
  const ncols = spec.widths.length;
  const sum = spec.widths.reduce((a, b) => a + b, 0);
  const widths = spec.widths.map((w) => Math.floor(total * w / sum));
  widths[ncols - 1] += total - widths.reduce((a, b) => a + b, 0);
  const fs = TWO_COL ? 16 : BMCSUB ? 20 : 18;
  const mkRow = (cells, isHeader, isFirst, isLast) => {
    let col = 0;
    return new TableRow({
      tableHeader: isHeader,
      children: cells.map((c) => {
        const cell = typeof c === 'string' ? { t: c } : c;
        const span = cell.span || 1;
        const w = widths.slice(col, col + span).reduce((a, b) => a + b, 0);
        const first = col === 0;
        col += span;
        return new TableCell({
          width: { size: w, type: WidthType.DXA },
          columnSpan: span,
          verticalAlign: VerticalAlign.CENTER,
          margins: { top: 30, bottom: 30, left: 60, right: 60 },
          borders: { top: isFirst ? border : none, bottom: isLast ? border : none, left: none, right: none },
          children: [new Paragraph({
            children: runs(cell.t, { size: fs, bold: isHeader && !IEEE }),
            alignment: first ? AlignmentType.LEFT : AlignmentType.CENTER,
            spacing: { line: 220 },
          })],
        });
      }),
    });
  };
  const rows = [];
  spec.header.forEach((h, i) => rows.push(mkRow(h, true, i === 0, i === spec.header.length - 1)));
  spec.rows.forEach((r, i) => rows.push(mkRow(r, false, false, i === spec.rows.length - 1)));
  const caption = IEEE
    ? [new Paragraph({ children: [new TextRun({ text: `TABLE ${ROMAN[tableNo - 1]}`, font: FONT, size: fs })], alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160 } }),
       new Paragraph({ children: runs(spec.caption, { size: fs, smallCaps: true }), alignment: AlignmentType.CENTER, keepNext: true, spacing: { after: 60 } })]
    : [new Paragraph({ children: [new TextRun({ text: `Table ${tableNo} `, bold: true, font: FONT, size: fs }), ...runs(spec.title || spec.caption, { size: fs, bold: BMCSUB })], keepNext: true, spacing: { before: 200, after: 80, line: 240 } })];
  const out = [...caption, new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows, alignment: AlignmentType.CENTER })];
  if (spec.note) out.push(new Paragraph({ children: runs(spec.note, { size: TWO_COL ? 14 : 16 }), spacing: { before: 40, after: 160 } }));
  else out.push(new Paragraph({ children: [], spacing: { after: 120 } }));
  return out;
}

function pngSize(buf) {
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
}

const figures = [];
function figure(spec, fullWidth) {
  figNo++;
  figures.push(spec);
  if (BMCSUB || JIIM_SUB) return [];
  const img = fs.readFileSync(path.join(FOLDER, spec.path));
  const wIn = TWO_COL ? (fullWidth ? (IEEE ? 7.0 : 6.6) : 3.2) : 6.2;
  const { w, h } = pngSize(img);
  const px = Math.round(wIn * 96);
  const capSize = TWO_COL ? 16 : 18;
  return [
    new Paragraph({ children: [new ImageRun({ type: 'png', data: img, transformation: { width: px, height: Math.round(px * h / w) },
      altText: { title: `Figure ${figNo}`, description: spec.caption.slice(0, 200), name: `fig${figNo}` } })],
      alignment: AlignmentType.CENTER, spacing: { before: 120 }, keepNext: true }),
    new Paragraph({ children: [new TextRun({ text: IEEE ? `Fig. ${figNo}. ` : `Fig. ${figNo} `, bold: !IEEE, font: FONT, size: capSize }), ...runs(spec.title ? `${spec.title}. ${spec.caption}` : spec.caption, { size: capSize })],
      alignment: AlignmentType.JUSTIFIED, spacing: { after: 200 } }),
  ];
}

function render(blocks, fullWidth = false) {
  const out = [];
  for (const b of blocks) {
    if (b.h1) out.push(heading1(b.h1));
    else if (b.h2) out.push(heading2(b.h2));
    else if (b.h) out.push(unnumbered(b.h));
    else if (b.p !== undefined) out.push(para(b.p, b));
    else if (b.eq) out.push(equation(b.eq, b.n !== false));
    else if (b.ol) { listInstance++; out.push(...list(b.ol, true)); }
    else if (b.ul) out.push(...list(b.ul, false));
    else if (b.table) out.push(...table(b.table, fullWidth || b.wide));
    else if (b.figure) out.push(...figure(b.figure, fullWidth || b.wide));
  }
  return out;
}

function referenceList() {
  const items = citeOrder.map((k, i) => {
    const segs = formatRef(bib[k]);
    return new Paragraph({
      children: [new TextRun({ text: BMC ? `${i + 1}.\t` : `[${i + 1}]\t`, font: FONT, size: TWO_COL ? 16 : 20 }),
        ...segs.map(([t, it]) => new TextRun({ text: t, italics: it, font: FONT, size: TWO_COL ? 16 : 20 }))],
      tabStops: [{ type: TabStopType.LEFT, position: TWO_COL ? 400 : 560 }],
      indent: { left: TWO_COL ? 400 : 560, hanging: TWO_COL ? 400 : 560 },
      alignment: AlignmentType.LEFT,
      spacing: { after: TWO_COL ? 20 : 60, line: 240 },
    });
  });
  return [unnumbered('References'), ...items];
}

// ─── Front / back matter from structured metadata (bmc) ──────────────────────
function submissionFront(m) {
  const P = (children, o = {}) => new Paragraph({ children, alignment: o.align || AlignmentType.LEFT, spacing: { after: o.after ?? 120, line: LINE } });
  const T = (text, o = {}) => new TextRun({ text, font: FONT, size: o.size || BASE, bold: o.bold, italics: o.italics, superScript: o.sup });
  const out = [
    P([T('Title page', { bold: true })]),
    P([T('Article type: ', { bold: true }), T(m.articleType || 'Research')]),
    P([T(m.title, { bold: true, size: 28 })], { after: 200 }),
    P(m.authors.flatMap((a, i) => [T((i ? ', ' : '') + a.name), T(a.affil.join(',') + (a.corresponding ? '*' : ''), { sup: true })])),
    ...m.affils.map((x, i) => P([T(`${i + 1}`, { sup: true }), T(x)], { after: 60 })),
    P([T('E-mail addresses: ', { bold: true }), T(m.authors.map((a) => `${a.name}, ${a.email}`).join('; '))]),
    P([T('*Corresponding author: ', { bold: true }), T(m.authors.filter((a) => a.corresponding).map((a) => `${a.name}, ${m.affils[a.affil[0] - 1]}. E-mail: ${a.email}`).join('; '))], { after: 240 }),
    unnumbered('Abstract'),
  ];
  m.abstract.forEach(([label, text]) => out.push(new Paragraph({ children: [T(`${label}: `, { bold: true }), ...runs(text)],
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: LINE } })));
  out.push(P([T('Keywords: ', { bold: true }), T(m.keywords.join(', '))], { after: 240 }));
  return out;
}

function metaFront(m) {
  const P = (children, o = {}) => new Paragraph({ children, alignment: o.align || AlignmentType.LEFT,
    spacing: { after: o.after ?? 80, line: o.line || 240 }, border: o.border });
  const T = (text, o = {}) => new TextRun({ text, font: FONT, size: o.size || BASE, bold: o.bold, italics: o.italics, superScript: o.sup });
  const out = [
    P([T('Research Article', { size: 18, italics: true })], { after: 120 }),
    P([T(m.title, { size: 32, bold: true })], { after: 200, line: 300 }),
    P(m.authors.flatMap((a, i) => [T((i ? ', ' : '') + a.name, { size: 22 }), T(a.affil.join(',') + (a.corresponding ? '*' : ''), { size: 22, sup: true })]), { after: 80 }),
    ...m.affils.map((x, i) => P([T(`${i + 1}`, { size: 18, sup: true }), T(x, { size: 18 })], { after: 40 })),
    P([T('*Correspondence: ', { size: 18, italics: true }), T(m.authors.filter((a) => a.corresponding).map((a) => a.email).join(', '), { size: 18 })], { after: 200 }),
  ];
  out.push(P([T('Abstract', { size: 22, bold: true })], { after: 60 }));
  m.abstract.forEach(([label, text]) => out.push(new Paragraph({ children: [T(`${label}: `, { bold: true, size: 19 }), ...runs(text, { size: 19 })],
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: 240 } })));
  out.push(P([T('Keywords: ', { bold: true, size: 19 }), T(m.keywords.join(', '), { size: 19 })],
    { after: 240, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: '999999', space: 4 } } }));
  return out;
}

// Springer Nature LaTeX template (sn-jnl) front matter, as typeset by \maketitle: centred title, authors,
// affiliations and corresponding-author line, then the abstract and keywords.
function snFront(m) {
  const T = (text, o = {}) => new TextRun({ text, font: FONT, size: o.size || BASE, bold: o.bold, italics: o.italics, superScript: o.sup });
  const C = (children, o = {}) => new Paragraph({ children, alignment: o.align || AlignmentType.CENTER,
    spacing: { after: o.after ?? 60, before: o.before || 0, line: o.line || 240 }, indent: o.indent });
  const af = content.sn.affils;
  const out = [
    C([T(m.title, { size: 34, bold: true })], { after: 240, line: 320 }),
    C(m.authors.flatMap((a, i) => [T((i ? ', ' : '') + a.name, { size: 22 }), T(a.affil.join(',') + (a.corresponding ? '*' : ''), { size: 22, sup: true })]), { after: 160 }),
    ...af.map((x, i) => C([T(`${a1(i)}`, { size: 18, sup: true }), T(`${x.orgdiv}, ${x.orgname}, `, { size: 18, italics: true }), T(`${x.city}, ${x.country}`, { size: 18, italics: true })], { after: 60 })),
    C([T('*Corresponding author(s). E-mail(s): ', { size: 18 }), T(m.authors.filter((a) => a.corresponding).map((a) => a.email).join('; '), { size: 18 })], { before: 120, after: 280 }),
    C([T('Abstract', { size: 20, bold: true })], { after: 80 }),
  ];
  function a1(i) { return `${i + 1}${m.authors.some((a) => a.corresponding && a.affil[0] === i + 1) ? '*' : ''}`; }
  const ind = { left: 567, right: 567 };
  const abs = typeof m.abstract === 'string' ? runs(m.abstract, { size: 18 })
    : m.abstract.flatMap(([label, text], k) => [T(`${k ? ' ' : ''}${label}: `, { bold: true, size: 18 }), ...runs(text, { size: 18 })]);
  out.push(new Paragraph({ children: abs,
    alignment: AlignmentType.JUSTIFIED, indent: ind, spacing: { after: 120, line: 240 } }));
  out.push(new Paragraph({ children: [T('Keywords: ', { bold: true, size: 18 }), T(m.keywords.join(', '), { size: 18 })],
    alignment: AlignmentType.JUSTIFIED, indent: ind, spacing: { after: 360 } }));
  return out;
}

// Back matter in the template's order: \bmhead{Supplementary information}, \bmhead{Acknowledgements},
// abbreviations (required by BMC), then "Declarations" as an itemized list with run-in labels.
function snBackMatter() {
  const sn = content.sn;
  const runIn = (label, text) => new Paragraph({ children: [new TextRun({ text: `${label} `, bold: true, font: FONT, size: BASE }), ...runs(text)],
    alignment: text.includes('https://') ? AlignmentType.LEFT : AlignmentType.JUSTIFIED, spacing: { before: 120, after: 80, line: LINE } });
  const out = [
    runIn('Supplementary information', sn.supplementary),
    runIn('Acknowledgements', sn.acknowledgements),
    runIn('Abbreviations', content.abbreviations.map(([a, b]) => `${a}, ${b}`).join('; ') + '.'),
    unnumbered('Declarations'),
  ];
  for (const [label, text] of sn.declarations) {
    out.push(new Paragraph({ children: [new TextRun({ text: `${label}: `, bold: true, font: FONT, size: BASE }), ...runs(text)],
      numbering: { reference: 'bullets', level: 0 }, alignment: text.includes('https://') ? AlignmentType.LEFT : AlignmentType.JUSTIFIED, spacing: { after: 40, line: LINE } }));
  }
  return out;
}

function figureLegends() {
  const out = [unnumbered(JIIM ? 'Figure Captions' : 'Figure titles and legends')];
  figures.forEach((sp, i) => {
    if (JIIM) {
      // Springer: bold "Fig." and number, no punctuation after the number or at the end of the caption
      out.push(new Paragraph({ children: [new TextRun({ text: `Fig. ${i + 1} `, bold: true, font: FONT, size: BASE }),
        ...runs(`${sp.title}. ${sp.caption}`.replace(/\.$/, ''))], alignment: AlignmentType.JUSTIFIED, spacing: { before: 120, after: 80, line: LINE }, keepNext: PREVIEW }));
    } else {
      out.push(new Paragraph({ children: [new TextRun({ text: `Fig. ${i + 1} `, bold: true, font: FONT, size: BASE }), ...runs(sp.title, { bold: true })],
        spacing: { before: 120, line: LINE }, keepNext: true }));
      out.push(para(sp.caption));
    }
    if (PREVIEW) {
      const img = fs.readFileSync(path.join(FOLDER, sp.path));
      const { w, h } = pngSize(img);
      const px = Math.round(6.2 * 96);
      out.push(new Paragraph({ children: [new ImageRun({ type: 'png', data: img, transformation: { width: px, height: Math.round(px * h / w) },
        altText: { title: `Figure ${i + 1}`, description: sp.title, name: `fig${i + 1}` } })], alignment: AlignmentType.CENTER, spacing: { after: 200 } }));
    }
  });
  return out;
}

function additionalFilesList() {
  if (!content.additionalFiles) return [];
  const out = [unnumbered('Additional files')];
  for (const a of content.additionalFiles) {
    out.push(new Paragraph({ children: [new TextRun({ text: `${a.name} (${a.format})`, bold: true, font: FONT, size: BASE })], spacing: { before: 120, line: LINE }, keepNext: true }));
    out.push(para(`*${a.title}.* ${a.description}`));
  }
  return out;
}

// ─── Journal of Imaging Informatics in Medicine ─────────────────────────────
const P1 = (children, o = {}) => new Paragraph({ children, alignment: o.align || AlignmentType.LEFT, spacing: { before: o.before || 0, after: o.after ?? 120, line: LINE } });
const T1 = (text, o = {}) => new TextRun({ text, font: FONT, size: o.size || BASE, bold: o.bold, italics: o.italics, superScript: o.sup });

// Author block shared by the title page and the full manuscript
function jiimAuthors(m) {
  const a = m.authors[0];
  return [
    P1([T1(a.name), T1('1*', { sup: true })], { after: 60 }),
    P1([T1('1', { sup: true }), T1(m.address)], { after: 120 }),
    P1([T1('*Corresponding author: ', { bold: true }), T1(`${a.name}, ${m.address}. E-mail: ${a.email}`)]),
    P1([T1('ORCID: ', { bold: true }), ...runs(`${a.name}: ${m.orcid}`)], { after: 240 }),
  ];
}

function jiimFront(m) {
  const out = [P1([T1(m.title, { bold: true, size: 28 })], { after: 240 })];
  if (!BLIND) out.push(...jiimAuthors(m));
  out.push(unnumbered('Abstract'), para(m.abstract));
  out.push(P1([T1('Keywords: ', { bold: true }), T1(m.keywords.join(' · '))], { after: 240 }));
  return out;
}

function statementsList(list) {
  const out = [];
  for (const [h, t] of list) {
    out.push(P1([T1(`${h}: `, { bold: true }), ...runs(t)], { align: t.includes('https://') ? AlignmentType.LEFT : AlignmentType.JUSTIFIED, after: 80 }));
  }
  return out;
}

// Full manuscript: Acknowledgements and Statements and Declarations before the references
function jiimBack() {
  if (BLIND) return [];
  return [unnumbered('Acknowledgements'), para(content.acknowledgements), unnumbered('Statements and Declarations'), ...statementsList(content.statementsCited)];
}

function onlineResourcesList() {
  const out = [unnumbered('Electronic Supplementary Material')];
  for (const a of content.onlineResources) {
    out.push(P1([T1(`${a.name} `, { bold: true }), T1(`(${a.file}) `), ...runs(`*${a.title}.* ${a.description}`)], { align: AlignmentType.JUSTIFIED, after: 80 }));
  }
  return out;
}

function jiimTitlePage(m) {
  return [
    P1([T1('Title Page', { bold: true, size: 24 })], { after: 200 }),
    P1([T1('Article type: ', { bold: true }), T1(m.articleType)]),
    P1([T1('Title: ', { bold: true }), T1(m.title, { bold: true })], { after: 240 }),
    P1([T1('Author', { bold: true })], { after: 60 }),
    ...jiimAuthors(m),
    unnumbered('Acknowledgements'),
    para(content.acknowledgements),
    unnumbered('Statements and Declarations'),
    ...statementsList(content.statements),
  ];
}

function backMatter() {
  const out = [];
  if (content.abbreviations) {
    out.push(unnumbered('Abbreviations'));
    out.push(...content.abbreviations.map(([a, b]) => new Paragraph({ children: [new TextRun({ text: `${a}: `, bold: true, font: FONT, size: BMCSUB ? BASE : 18 }), new TextRun({ text: b, font: FONT, size: BMCSUB ? BASE : 18 })], spacing: { after: 20, line: BMCSUB ? LINE : undefined } })));
  }
  if (content.declarations) {
    out.push(unnumbered('Declarations'));
    for (const [h, t] of content.declarations) {
      out.push(new Paragraph({ children: [new TextRun({ text: h, bold: true, font: FONT, size: BMCSUB ? BASE : 19 })], spacing: { before: 80, after: 20, line: BMCSUB ? LINE : undefined }, keepNext: true }));
      out.push(para(t, { run: { size: BMCSUB ? BASE : 19 } }));
    }
  }
  return out;
}

// ─── Assemble ──────────────────────────────────────────────────────────────
const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] });

const numbering = { config: [
  { reference: 'bullets', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 360, hanging: 240 } } } }] },
  { reference: 'numbers', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 400, hanging: 300 } } } }] },
] };

const styles = { default: { document: { run: { font: FONT, size: BASE }, paragraph: BMCSUB || JIIM_SUB ? { spacing: { line: LINE } } : undefined } } };

let sections;
if (JIIM_SUB) {
  const page = { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } };
  const children = TITLEPAGE ? jiimTitlePage(content.meta)
    : [...jiimFront(content.meta), ...render(content.body), ...jiimBack(), ...referenceList(), ...figureLegends(), ...onlineResourcesList()];
  sections = [{ properties: { page, lineNumbers: TITLEPAGE ? undefined : { countBy: 1, restart: LineNumberRestartFormat.CONTINUOUS, distance: 300 } },
    footers: { default: footer }, children }];
} else if (BMCSUB) {
  const page = { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } };
  const children = [...submissionFront(content.meta), ...render(content.body), ...backMatter(), ...referenceList(), ...figureLegends(), ...additionalFilesList()];
  sections = [{ properties: { page, lineNumbers: { countBy: 1, restart: LineNumberRestartFormat.CONTINUOUS, distance: 300 } },
    footers: { default: footer }, children }];
} else if (!TWO_COL) {
  const children = [...content.front(runs, Paragraph, TextRun, AlignmentType), ...render(content.body)];
  children.push(...referenceList());
  sections = [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: footer }, children }];
} else {
  // Front matter in one column, body in two columns; wide figures/tables get their own one-column sections.
  const page = IEEE ? { size: { width: 12240, height: 15840 }, margin: { top: 1080, bottom: 1440, left: 936, right: 936 } }
                    : { size: { width: 11906, height: 16838 }, margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 } };
  const two = { count: 2, space: IEEE ? 396 : 360, equalWidth: true };
  const SN = content.sn && (STYLE === 'bmc' || JIIM2);
  const front = SN ? snFront(content.meta) : content.meta ? metaFront(content.meta) : content.front(runs, Paragraph, TextRun, AlignmentType);
  // Springer Nature template header: running title (left) and page number (right); no footer
  const snHeader = SN ? { default: new Header({ children: [new Paragraph({ tabStops: [{ type: TabStopType.RIGHT, position: FULL_W }],
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: '000000', space: 2 } },
    children: [new TextRun({ text: content.meta.running, italics: true, font: FONT, size: 16 }), new TextRun({ text: '\t', font: FONT, size: 16 }), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16 })] })] }) } : undefined;
  const feet = snHeader ? undefined : { default: footer };
  sections = [{ properties: { page }, headers: snHeader, footers: feet, children: front }];
  let buf = [];
  const flush = () => { if (buf.length) { sections.push({ properties: { type: SectionType.CONTINUOUS, page, column: two }, headers: snHeader, footers: feet, children: buf }); buf = []; } };
  for (const b of content.body) {
    if (b.wide) { flush(); sections.push({ properties: { type: SectionType.CONTINUOUS, page }, headers: snHeader, footers: feet, children: render([b], true) }); }
    else buf.push(...render([b]));
  }
  buf.push(...(JIIM2 ? jiimBack() : SN ? snBackMatter() : backMatter()), ...referenceList());
  flush();
  // An empty continuous section at the end makes Word balance the last page's two columns
  sections.push({ properties: { type: SectionType.CONTINUOUS, page }, headers: snHeader, footers: feet, children: [new Paragraph({ children: [] })] });
}

// The blinded manuscript must not name the author anywhere, including the file properties
const doc = new Document({ creator: BLIND ? 'Anonymous' : 'Debobrato Biswas', lastModifiedBy: BLIND ? 'Anonymous' : undefined, title: content.title, styles, numbering, sections });
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log(`wrote ${OUT}: ${citeOrder.length} references, ${tableNo} tables, ${figNo} figures, ${eqNo} equations`);
});
