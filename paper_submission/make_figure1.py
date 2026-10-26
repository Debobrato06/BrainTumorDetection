"""Draws Figure 1 (BrainHybridNet architecture) for both submissions.
Usage: python paper_submission/make_figure1.py"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
C = {'in': '#e8eef7', 'cnn': '#cfe3f5', 'vit': '#f9dcc4', 'dec': '#d8efd3', 'head': '#efd9f0', 'aux': '#f3f3f3'}


def box(ax, x, y, w, h, text, color, fs=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                                fc=color, ec='#333333', lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs)


def arrow(ax, p, q, style='-|>', ls='-', color='#333333'):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8, lw=0.8, ls=ls, color=color))


fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.set_xlim(0, 14.4)
ax.set_ylim(0, 7.2)
ax.axis('off')

# Encoder column (left to right along the top)
box(ax, 0.1, 5.6, 1.9, 1.2, 'Input crop\n4 x 128$^3$\nT1, T1ce, T2, FLAIR', C['in'], 6.5)
box(ax, 2.4, 5.6, 1.6, 1.2, 'Sequence\nattention\n(SE)', C['cnn'])
box(ax, 4.4, 5.6, 1.8, 1.2, 'CNN stage 1\n32 ch, 1/1\n($c_1$)', C['cnn'])
box(ax, 6.6, 5.6, 1.8, 1.2, 'CNN stage 2\n64 ch, 1/2\n($c_2$)', C['cnn'])
box(ax, 8.8, 5.6, 2.1, 1.2, 'Patch embed\np = 8, E = 128\n512 tokens + pos.', C['vit'], 7)
box(ax, 11.3, 5.6, 2.9, 1.2, 'Transformer encoder\nL = 4 blocks, 4 heads\nMHSA + MLP (pre-norm)', C['vit'], 7)
for a, b in [((2.0, 6.2), (2.4, 6.2)), ((4.0, 6.2), (4.4, 6.2)), ((6.2, 6.2), (6.6, 6.2)),
             ((8.4, 6.2), (8.8, 6.2)), ((10.9, 6.2), (11.3, 6.2))]:
    arrow(ax, a, b)

# Heads from pooled tokens
box(ax, 11.3, 3.7, 1.35, 1.1, 'Detection\nhead\n(tumor / none)', C['head'], 6.5)
box(ax, 12.85, 3.7, 1.35, 1.1, 'Grade\nhead\n(LGG / HGG)', C['head'], 6.5)
ax.text(12.75, 5.2, 'mean pooling', ha='center', va='center', fontsize=6.5, style='italic',
        bbox=dict(fc='white', ec='none', pad=0.5))
arrow(ax, (12.2, 5.6), (11.98, 4.8))
arrow(ax, (13.3, 5.6), (13.52, 4.8))

# Decoder (right to left along the bottom)
box(ax, 8.8, 3.7, 2.1, 1.1, 'Reshape to 8$^3$\nUp x2, Up x2\n(ResSE blocks)', C['dec'], 6.8)
arrow(ax, (11.6, 5.6), (10.4, 4.8))
box(ax, 6.6, 1.9, 1.8, 1.1, 'Up x2 + concat $c_2$\nResSE', C['dec'], 6.8)
box(ax, 4.4, 1.9, 1.8, 1.1, 'Up x2 + concat $c_1$\nResSE', C['dec'], 6.8)
box(ax, 2.0, 1.9, 2.0, 1.1, 'Segmentation head\n1x1x1 conv, sigmoid\nWT / TC / ET', C['head'], 6.5)
arrow(ax, (9.85, 3.7), (7.9, 3.0))
arrow(ax, (6.6, 2.45), (6.2, 2.45))
arrow(ax, (4.4, 2.45), (4.0, 2.45))
# Skip connections
arrow(ax, (7.5, 5.6), (7.5, 3.0), ls='--', color='#1f5fa8')
arrow(ax, (5.3, 5.6), (5.3, 3.0), ls='--', color='#1f5fa8')
ax.text(7.6, 4.3, 'skip $c_2$', fontsize=6.5, color='#1f5fa8')
ax.text(5.4, 4.3, 'skip $c_1$', fontsize=6.5, color='#1f5fa8')

# Auxiliary outputs
box(ax, 0.1, 0.2, 3.9, 1.1, 'MC Dropout, T = 20 passes\n→ voxel-wise uncertainty map', C['aux'], 6.8)
box(ax, 4.4, 0.2, 4.0, 1.1, 'Last-block attention weights\n→ saliency map', C['aux'], 6.8)
box(ax, 8.8, 0.2, 5.4, 1.1, 'Optional pre-training: 3D masked image modeling\n75% of patches masked, reconstruct masked voxels', C['aux'], 6.8)
arrow(ax, (3.0, 1.9), (2.0, 1.3), ls=':')

plt.tight_layout(pad=0.1)
for folder in ('1_SpringerNature_NCA', '2_IEEE_JBHI'):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(HERE, folder, 'figures', f'fig1_architecture.{ext}'), dpi=300)
print('saved')
