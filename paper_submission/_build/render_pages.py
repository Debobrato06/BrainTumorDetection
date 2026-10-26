"""Renders PDF pages to PNG for visual checking: python render_pages.py <in.pdf> <out_dir>"""
import os
import sys

import fitz

pdf, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
doc = fitz.open(pdf)
for i, page in enumerate(doc):
    page.get_pixmap(dpi=70).save(os.path.join(out, f'page-{i + 1:02d}.png'))
print(f'{pdf}: {len(doc)} pages')
