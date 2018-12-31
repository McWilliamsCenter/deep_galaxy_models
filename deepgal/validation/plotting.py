import os
import sys
from pylab import *
from matplotlib.colors import LogNorm
import seaborn as sns
from absl import flags

def moments_plots(cat, m_real, m_mock, prefix=""):
    """
    Compare moments distributions between real images and mocks
    """
    mask_r = m_real['flag']
    mask_m = m_mock['flag']
    mask = mask_r & mask_m
    # Add mask to remove objects larger than half the frame
    mask = mask & (cat['flux_radius'] < 32)

    sns.set_context("paper", font_scale=3.)
    # Compare the distributions of sizes
    figure(figsize=(15,5))
    subplot(121)
    scatter(cat['flux_radius'][mask], m_real['sigma_e'][mask], c=cat['mag_auto'][mask],cmap='rainbow',vmin=20,alpha=0.6); colorbar(label='Input magnitude')
    xlim(0,50)
    ylim(0,20)
    xlabel("Input half-light radius [pix]")
    ylabel("Measured moments sigma [pix]")
    title("COSMOS images")
    subplot(122)
    scatter(cat['flux_radius'][mask], m_mock['sigma_e'][mask], c=cat['mag_auto'][mask],cmap='rainbow',vmin=20,alpha=0.6); colorbar(label='Input magnitude')
    xlim(0,50)
    ylim(0,20)
    xlabel("Input half-light radius [pix]")
    ylabel("Measured moments sigma [pix]")
    title("Generated images")
    suptitle("Comparison of Galaxy sizes from moments")
    savefig(prefix+"radius_vs_sigma.pdf", transparent=True)

    figure(figsize=(7,5))
    scatter( m_real['sigma_e'][mask], m_mock['sigma_e'][mask], c=cat['flux_radius'][mask],cmap='magma',alpha=0.5, norm=LogNorm(vmin=1, vmax=25))
    cbar = colorbar(label='Input half-light radius [pix]' )
    cbar.set_alpha(1); cbar.draw_all()
    plot([0,20],[0,20],'r',lw=2)
    xlim(1.5,20)
    ylim(1.5,20)
    yscale('log')
    xscale('log')
    xlabel("COSMOS size [pix]")
    ylabel("C-VAE sample size [pix]")
    #title("Comparison of galaxy sizes measured from moments")
    savefig(prefix+"size_conditioning.pdf", transparent=True)
    savefig(prefix+"size_conditioning.png", transparent=True, bbox_inches='tight', pad_inches=0 )

    figure(figsize=(7,5))
    scatter(m_real['amp'], m_mock['amp'],c=cat['mag_auto'],cmap='magma_r',alpha=0.5,vmax=28,vmin=20)
    yscale('log')
    xscale('log')
    plot([0.5,1000],[0.5,1000],'r',lw=2)
    xlim(0.5,500)
    ylim(0.5,500)
    xlabel("COSMOS brightness")
    ylabel("C-VAE sample brightness")
    cbar =colorbar(label='Input magnitude')
    cbar.set_alpha(1); cbar.draw_all()
    savefig(prefix+"magnitude_conditioning.pdf", transparent=True)
    savefig(prefix+"magnitude_conditioning.png", transparent=True, bbox_inches='tight', pad_inches=0 )

    figure()
    m = mask
    g = sns.jointplot(cat['flux_radius'][m], (m_real['sigma_e'][m] - m_mock['sigma_e'][m])/ cat['flux_radius'][m],gridsize=50, kind='hex', xscale='log', bins='log')
    g.ax_joint.hlines(0,cat['flux_radius'][m].min(),30,color='r')
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
    g.ax_joint.errorbar(b,res_m, res_std, color='y')
    savefig(prefix+"size_error.pdf", transparent=True)

    sns.set_context("paper", font_scale=1.8)
    figure(figsize=(5,5))
    bins = linspace(0,0.8,17)
    hist(m_real['g'][m], bins,label='COSMOS galaxies',alpha=0.5, normed=True);
    hist(m_mock['g'][m], bins,label='CVAE samples',alpha=0.5, normed=True);
    xlabel("Ellipticity $|e|$")
    xlim(0,0.8)
    legend()
    savefig(prefix+"ellipticity.pdf", transparent=True)

    figure(figsize=(5,5))
    bins = linspace(0,16,17)
    hist(m_real['sigma_e'][m], bins,label='COSMOS galaxies',alpha=0.5, normed=True);
    hist(m_mock['sigma_e'][m], bins,label='CVAE samples',alpha=0.5, normed=True);
    xlabel("Size $\sigma$ [pix]")
    legend()
    ylim(0,0.20)
    savefig(prefix+"size.pdf", transparent=True)
