"""Embed the completed fresh restoration evaluation in a results notebook."""
import base64
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent
folders=[p for p in (ROOT/'runs/full').glob('results_ssim_rerun_*') if (p/'metrics_long.csv').exists()]
folders = [p for p in folders
           if set(pd.read_csv(p/'metrics_long.csv')['Method']) == {'JPEG', 'CNN + L1'}]
if not folders:raise RuntimeError('Run rerun_evaluation.py to produce a spatial-only evaluation first')
folder=sorted(folders)[-1]
results=pd.read_csv(folder/'metrics_long.csv')
cells=[dict(cell_type='markdown',id='title',metadata={},source=['# JPEG restoration — freshly rerun PSNR, SSIM and BPP\n','The spatial-loss CNN was re-evaluated on all 1,576 held-out 256×256 crops at Q10, Q20, Q30, Q40 and Q50. Weights were not retrained. JPEG input bpp is shared by every restoration method. SSIM uses TorchMetrics RGB data_range=1 and the default Gaussian 11×11 window, sigma 1.5.\n'])]
code=f"from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nRESULTS_DIR = Path({str(folder)!r})\nresults = pd.read_csv(RESULTS_DIR / 'metrics_long.csv')\ndisplay(results)\n"
cells.append(dict(cell_type='code',id='metrics',metadata={},execution_count=1,source=code.splitlines(True),outputs=[dict(output_type='display_data',metadata={},data={'text/html':results.to_html(index=False),'text/plain':results.to_string(index=False)})]))
for i,name in enumerate(['bitrate_curves.png','comparison_000_q10.png','comparison_000_q30.png','comparison_000_q50.png'],start=2):
 path=folder/name
 cells.append(dict(cell_type='code',id=f'figure-{i}',metadata={},execution_count=i,source=[f"display(Image(filename=str(RESULTS_DIR / {name!r})))\n"],outputs=[dict(output_type='display_data',metadata={},data={'image/png':base64.b64encode(path.read_bytes()).decode(),'text/plain':'<Evaluation figure>'})]))
metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.10'))
path=ROOT/'JPEG_Restoration_Results_Executed.ipynb'
path.write_text(json.dumps(dict(cells=cells,metadata=metadata,nbformat=4,nbformat_minor=5),indent=1))
print(path)
