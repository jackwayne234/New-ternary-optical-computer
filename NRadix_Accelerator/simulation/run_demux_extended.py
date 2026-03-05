import sys, numpy as np
from pathlib import Path
sys.path.insert(0, '/workspace/New-ternary-optical-computer/NRadix_Accelerator/simulation')
import fdtd_inverse_design as f

fa = f.load_frequency_assignment(Path('results'))
freqs = f.get_demux_freqs(fa)
vals = sorted(freqs.keys())
target = np.array([freqs[v] for v in vals])
wgs, owgs, mons, dr, gs = f.make_demux_waveguides(fa)
nb = f.make_n_background(gs, wgs, owgs)
d0 = np.load('results/demux_density_fdtd.npy')
src = max(int(wgs[0]['x_end']/f.DX)-3, f.PML_CELLS+2)

f.optimize_density(
    d0, nb, dr, wgs[0], mons, target, 60000, src,
    n_iterations=1000,
    learning_rate=0.008,
    save_path='results/demux_density_fdtd.npy',
)
