"""Re-evaluate existing restoration checkpoints with PSNR, SSIM and file bpp."""
from pathlib import Path
from datetime import datetime
import json
import torch
import matplotlib.pyplot as plt
from evaluate import load_checkpoint, evaluate_all
from visualize import show_comparison

ROOT=Path(__file__).resolve().parent
torch.set_num_threads(2)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models={}
configs=[]
for name,folder in [('CNN + L1','spatial_only')]:
    models[name],config=load_checkpoint(ROOT/'runs/full'/folder/'best.pt',device)
    configs.append(config)
cfg=dict(configs[0],test_dir=str(ROOT.parent/'data_1um_bicubic/test'),batch_size=8,num_workers=0)
output=ROOT/'runs/full'/('results_ssim_rerun_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
output.mkdir(parents=True)
(output/'protocol.json').write_text(json.dumps(dict(config=cfg,checkpoint_folders=['spatial_only'],
    note='Fresh evaluation, existing trained weights; all test images, no retraining.'),indent=2))
print('OUTPUT',output,flush=True)
results=evaluate_all(models,cfg,[10,20,30,40,50],device=device,train_qualities=configs[0]['qualities'])
results.to_csv(output/'metrics_long.csv',index=False)
results.pivot(index=['Q','BPP'],columns='Method',values=['PSNR','SSIM']).to_csv(output/'metrics_table.csv')
fig,axes=plt.subplots(1,3,figsize=(16,4))
reference=results[results.Method=='JPEG']
reference[['Q','BPP','Images']].to_csv(output/'quality_bpp.csv',index=False)
axes[0].plot(reference.Q,reference.BPP,'o-');axes[0].set(xlabel='JPEG Q',ylabel='JPEG file bpp')
for name,group in results.groupby('Method',sort=False):
    axes[1].plot(group.BPP,group.PSNR,'o-',label=name)
    axes[2].plot(group.BPP,group.SSIM,'o-',label=name)
axes[1].set(xlabel='JPEG input bpp',ylabel='PSNR (dB)')
axes[2].set(xlabel='JPEG input bpp',ylabel='SSIM')
for ax in axes:ax.grid(alpha=.2)
axes[1].legend();axes[2].legend();fig.tight_layout();fig.savefig(output/'bitrate_curves.png',dpi=160);plt.close(fig)
for quality in (10,30,50):show_comparison(models,cfg,0,quality,output,device=device)
print(results.to_string(index=False),flush=True)
print('FINISHED',output,flush=True)
