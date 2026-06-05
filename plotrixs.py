import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

jctc_colors = [
    (1.00, 1.00, 1.00),   # white
    (0.85, 0.85, 1.00),   # very light blue
    (0.40, 0.50, 0.95),   # blue
    (0.20, 0.70, 0.40),   # green
    (0.95, 0.70, 0.20),   # orange
    (0.85, 0.20, 0.15),   # red
]
jctc_cmap = LinearSegmentedColormap.from_list("jctc", jctc_colors, N=256)

def plotrixs(rixs_map, incident_en_grid, loss_en_grid,
                     title="", savepath="rixsmap.png",
                     incident_window=None, loss_window=(-5, 15),
                     resonance_markers=None,percentile_clip=97):
    """
    incident_window: tuple (min, max) eV. If None, uses where intensity is significant.
    resonance_markers: list of incident energies to mark with dashed lines.
    """
    rixs_norm= rixs_map
    if incident_window is None:
        intensity_per_incident = rixs_norm.sum(axis=1)
        sig_mask = intensity_per_incident > 0.01 * intensity_per_incident.max()
        sig_indices = np.where(sig_mask)[0]
        if len(sig_indices) > 0:
            i_lo = max(0, sig_indices[0] - 20)
            i_hi = min(len(incident_en_grid)-1, sig_indices[-1] + 20)
            incident_window = (incident_en_grid[i_lo], incident_en_grid[i_hi])
        else:
            incident_window = (incident_en_grid.min(), incident_en_grid.max())
    
    vmax_val = rixs_map.max() * 0.05   # ⟵ tune this: 0.01 to 0.2
    
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='#fffef5')
    ax.set_facecolor('#fffef5')
    mesh = ax.pcolormesh(
        incident_en_grid, loss_en_grid, rixs_norm.T,
        shading='gouraud',
        cmap=jctc_cmap,
        vmin=0.0, vmax=vmax_val
    )
    cbar = plt.colorbar(mesh, ax=ax)   # ⟵ removed ticks=np.arange(0, 1.1, 0.1)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label('RIXS intensity (arb. units)', fontsize=10)
    
    # rest unchanged
    if resonance_markers is not None:
        for re in resonance_markers:
            ax.axvline(x=re, color='red', linestyle='--', linewidth=1.2, alpha=0.8)
    
    ax.set_xlim(*incident_window)
    ax.set_ylim(*loss_window)
    ax.set_xlabel("Absorption energy, ω (eV)", fontsize=11)
    ax.set_ylabel("Energy Transfer, ω − ω' (eV)", fontsize=11)
    ax.tick_params(direction='in', length=4, labelsize=9)
    ax.text(0.04, 0.95, title, transform=ax.transAxes,
            fontsize=11, va='top', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()
    print(f"Saved to {savepath}")
# Find your resonance position(s) automatically
# xas = rixs_map.sum(axis=1)
# peak_idx = np.argmax(xas)
# resonance_energy = incident_en_grid[peak_idx]
# print(f"Strongest resonance at: {resonance_energy:.2f} eV")
