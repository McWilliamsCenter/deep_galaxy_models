import os
import sys
from pylab import *
from matplotlib.colors import LogNorm
import seaborn as sns
from absl import flags

red = '#A6192E'
blue= '#002d72'
yellow = '#F2A900'

sns.set()
sns.set_style("dark")

def moments_plots(cat, m_real, m_mock, prefix=""):
    """
    Compare moments distributions between real images and mocks
    """
    mask_r = m_real['flag']
    mask_m = m_mock['flag']
    mask = mask_r & mask_m
    sns.set()
    sns.set_style("dark")

    figure()
    m = mask
    g = sns.jointplot(cat['flux_radius'][m], (m_real['sigma_e'][m] - m_mock['sigma_e'][m])/ cat['flux_radius'][m],gridsize=50, kind='hex', xscale='log', color=blue, bins='log')
    g.ax_joint.hlines(0,cat['flux_radius'][m].min(),30,color=red)
    g.ax_joint.xaxis.set_label_text('Input half-light radius [pix]');
    g.ax_joint.yaxis.set_label_text('(cosmos - gen) / half_light_radius');
    err_rel = (m_real['sigma_e'][mask] - m_mock['sigma_e'][mask])/ cat['flux_radius'][mask]
    n = 10
    b = logspace(log10(3),log10(30),n)
    inds = digitize(cat['flux_radius'][mask], b)
    res_m = zeros(n)
    res_std = zeros(n)
    for i in range(n):
        res_m[i] = mean(err_rel[inds == i])
        res_std[i] = std(err_rel[inds == i])
    g.ax_joint.errorbar(b,res_m, res_std, color=yellow)
    savefig(prefix+"size_error.pdf", transparent=True)


    figure()
    m = mask & (cat['mag_auto'] > 18 ) & (m_mock['amp'] / m_real['amp'] < 2)
    g = sns.jointplot(cat['mag_auto'][m],clip(m_mock['amp'][m] / m_real['amp'][m],0.,2),
                  gridsize=50, kind='hex', color=blue, bins='log',
                  ylim=(0.,2), xlim=(18,25.2))

    g.ax_joint.hlines(1, 18,25.2, color=red)
    g.ax_joint.xaxis.set_label_text('mag_auto');
    g.ax_joint.yaxis.set_label_text('(flux_mock / flux_real)');

    err_rel = (m_mock['amp'][mask] / m_real['amp'][mask])
    n = 10
    b = linspace(18,25.2,n)
    inds = digitize(cat['mag_auto'][mask], b)
    res_m = zeros(n)
    res_std = zeros(n)
    for i in range(n):
        res_m[i] = mean(err_rel[inds == i])
        res_std[i] = std(err_rel[inds == i])
    g.ax_joint.errorbar(b,res_m, res_std, color=yellow)
    savefig(prefix+"flux_error.pdf", transparent=True)


    figure(figsize=(5,5))
    bins = linspace(0,0.8,25)
    a = hist(m_real['g'][mask], bins,color=red,alpha=0.3, normed=True, label='COSMOS galaxies');
    b = hist(m_mock['g'][mask], bins,color=blue,alpha=0.3, normed=True, label='mock galaxies');
    xlabel("Ellipticity $|e|$")
    xlim(0,0.8)
    legend()
    savefig(prefix+"ellipticity.pdf", transparent=True, bbox_inches='tight', pad_inches=0)

    figure(figsize=(5,5))
    bins = linspace(2.5,16,32)
    a = hist(m_real['sigma_e'][mask], bins,color=red,alpha=0.3, normed=True, label='COSMOS galaxies');
    b = hist(m_mock['sigma_e'][mask], bins,color=blue,alpha=0.3, normed=True, label='mock galaxies');
    xlabel("Size $\sigma$ [pix]")
    legend()
    ylim(0,0.20)
    savefig(prefix+"size.pdf", transparent=True, bbox_inches='tight', pad_inches=0)
