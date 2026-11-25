"""Measure JPEG input file bpp at each Q, without retraining restoration models.

Existing metrics can be augmented after checking image counts and JPEG PSNR.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from dataset import JPEGRestorationDataset
from metrics import psnr, batch_ssim
from torch.utils.data import DataLoader


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--patch_size',type=int,default=256)
    parser.add_argument('--qualities',type=int,nargs='+',default=[10,20,30,40,50])
    parser.add_argument('--existing_metrics',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for q in args.qualities:
        dataset=JPEGRestorationDataset(args.data_dir,args.patch_size,[q],train=False,return_bpp=True)
        bpp_sum,psnr_sum,ssim_sum,count=0.,0.,0.,0
        for degraded,clean,_,bpp in DataLoader(dataset,batch_size=8,shuffle=False):
            bpp_sum+=bpp.sum().item();psnr_sum+=psnr(degraded,clean).sum().item()
            ssim_sum+=batch_ssim(degraded,clean).item()*len(clean);count+=len(clean)
        row=dict(Q=q,BPP=bpp_sum/count,Images=count,JPEG_PSNR=psnr_sum/count,JPEG_SSIM=ssim_sum/count,Patch_size=args.patch_size)
        rows.append(row);print(row,flush=True)
    table=pd.DataFrame(rows);table.to_csv(args.output/'quality_bpp.csv',index=False)
    if args.existing_metrics:
        existing=pd.read_csv(args.existing_metrics).drop(columns=['BPP'],errors='ignore')
        reference=existing[existing.Method=='JPEG'].merge(table,on='Q',suffixes=('_old','_measured'))
        if len(reference)!=len(table) or not np.array_equal(reference.Images_old,reference.Images_measured):
            raise ValueError('Existing metric quality/image-count protocol does not match')
        if not np.allclose(reference.PSNR,reference.JPEG_PSNR,atol=1e-4,rtol=0):
            raise ValueError('Existing JPEG PSNR does not match freshly measured inputs; do not merge')
        combined=existing.merge(table[['Q','BPP']],on='Q',validate='many_to_one')
        combined.to_csv(args.output/'metrics_long.csv',index=False)
        combined.pivot(index=['Q','BPP'],columns='Method',values=['PSNR','SSIM']).to_csv(args.output/'metrics_table.csv')
        fig,axes=plt.subplots(1,3,figsize=(16,4))
        axes[0].plot(table.Q,table.BPP,'o-');axes[0].set(xlabel='JPEG quality factor Q',ylabel='JPEG input bpp')
        for method,group in combined.groupby('Method',sort=False):
            axes[1].plot(group.BPP,group.PSNR,'o-',label=method)
            axes[2].plot(group.BPP,group.SSIM,'o-',label=method)
        axes[1].set(xlabel='JPEG input bpp',ylabel='PSNR (dB)');axes[1].legend()
        axes[2].set(xlabel='JPEG input bpp',ylabel='SSIM');axes[2].legend()
        for ax in axes:ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(args.output/'bitrate_curves.png',dpi=160);plt.close(fig)
        (args.output/'README.txt').write_text('BPP measured freshly from exact 4:4:4, optimize=False JPEG streams on the configured crops.\nRestoration PSNR/SSIM are retained from the supplied existing metrics, not retrained/recomputed.\nImage counts and JPEG PSNR were verified against fresh decoding before merging.\nAll methods receive the same JPEG bytes; restored PNG file sizes are not measured rates.\n')

if __name__=='__main__':main()
