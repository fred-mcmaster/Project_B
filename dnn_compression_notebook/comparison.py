"""Measured latent-payload DNN and complete-file JPEG rates on the same images."""
import csv
import io
import json
import math
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from codec import compress, decompress, payload_size
from metrics import array_ssim


def psnr(original, reconstruction):
    mse = float(np.mean((original.astype(np.float64)-reconstruction.astype(np.float64))**2))
    return -10 * math.log10(max(mse, 1e-12))


def metric_functions(psnr_fn=None, ssim_fn=None):
    """Adapt notebook [B,C,H,W] metrics to the comparison's HWC arrays."""
    def adapt(fn):
        def score(original, reconstruction):
            pred = torch.from_numpy(np.array(reconstruction, dtype=np.float32, copy=True)).permute(2,0,1)[None]
            target = torch.from_numpy(np.array(original, dtype=np.float32, copy=True)).permute(2,0,1)[None]
            return float(fn(pred, target).item())
        return score
    return (psnr if psnr_fn is None else adapt(psnr_fn),
            array_ssim if ssim_fn is None else adapt(ssim_fn))


def jpeg_roundtrip(array, quality, optimize=False):
    image = Image.fromarray(np.rint(array*255).clip(0,255).astype(np.uint8))
    stream = io.BytesIO()
    image.save(stream, format='JPEG', quality=int(quality), subsampling=0, optimize=optimize)
    payload = stream.getvalue()
    with Image.open(io.BytesIO(payload)) as decoded:
        reconstruction = np.asarray(decoded.convert('RGB'), dtype=np.float32)/255
    return payload, reconstruction


@torch.no_grad()
def compare_jpeg(model, loader, device, output_dir, qualities=(1, 10, 30, 50, 70, 90), examples=4, *, psnr_fn=None, ssim_fn=None):
    """Evaluate every image; count DNN payload bytes and full JPEG file bytes.

    All metrics are calculated against the same [0,1] RGB tensors. No test
    metric is used to select a checkpoint. The first examples are saved.
    """
    score_psnr, score_ssim = metric_functions(psnr_fn, ssim_fn)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, saved, used = [], [], set()
    model.eval()
    for batch in loader:
        for tensor in batch:
            index = len(rows)//(len(qualities)+1)
            x = tensor.unsqueeze(0).to(device)
            array = tensor.permute(1,2,0).cpu().numpy()
            h, w = array.shape[:2]
            stream = compress(model, x)
            decoded_tensor = decompress(model, stream)
            result = model(x)
            torch.testing.assert_close(decoded_tensor, result['reconstruction'])
            used.update(result['indices'].cpu().flatten().tolist())
            decoded = decoded_tensor[0].cpu().permute(1,2,0).numpy()
            latent_bytes = payload_size(stream)
            mode = 'zlib' if stream[:4] == b'ECV1' else ('zlib', 'arithmetic-left', 'arithmetic-up')[json.loads(stream[8:8+int.from_bytes(stream[4:8], 'big')])['c']]
            rows.append(dict(image=index, method='DNN', bytes=latent_bytes, file_bytes=len(stream),
                             coding_mode=mode, pixels=h*w, actual_bpp=8*latent_bytes/(h*w),
                             psnr=score_psnr(array, decoded), ssim=score_ssim(array, decoded),
                             estimated_bpp=float(result['estimated_bpp'])))
            jpeg_examples = {}
            for quality in qualities:
                payload, jpeg = jpeg_roundtrip(array, quality)
                rows.append(dict(image=index, method=f'JPEG Q{quality}', bytes=len(payload), file_bytes=len(payload),
                                 coding_mode='jpeg', pixels=h*w,
                                 actual_bpp=8*len(payload)/(h*w), psnr=score_psnr(array, jpeg), ssim=score_ssim(array, jpeg), estimated_bpp=''))
                if index < examples:
                    (output_dir/f'image_{index}_jpeg_q{quality}.jpg').write_bytes(payload)
                    jpeg_examples[quality] = jpeg
            if index < examples:
                (output_dir/f'image_{index}.ecvq').write_bytes(stream)
                for label, value in [('original', array), ('decoded', decoded)]:
                    Image.fromarray(np.rint(value*255).clip(0,255).astype(np.uint8)).save(output_dir/f'image_{index}_{label}.png')
                saved.append((array, decoded, jpeg_examples))
    if not rows:
        raise ValueError('Comparison loader is empty')
    with (output_dir/'per_image.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = []
    for method in dict.fromkeys(row['method'] for row in rows):
        group = [row for row in rows if row['method'] == method]
        summary.append(dict(method=method, images=len(group),
                            actual_bpp=8*sum(r['bytes'] for r in group)/sum(r['pixels'] for r in group),
                            mean_psnr=float(np.mean([r['psnr'] for r in group])),
                            mean_ssim=float(np.mean([r['ssim'] for r in group])),
                            estimated_bpp=float(np.mean([r['estimated_bpp'] for r in group])) if method == 'DNN' else None))
    modes = {mode: sum(row['coding_mode'] == mode for row in rows if row['method'] == 'DNN')
             for mode in ('zlib', 'arithmetic-left', 'arithmetic-up')}
    (output_dir/'summary.json').write_text(json.dumps(dict(metrics=summary, used_codewords=len(used),
        dnn_coding_modes=modes,
        codebook_size=model.config['codebook_size'], jpeg_subsampling=0,
        rate_definition='DNN: compressed latent payload bytes only; JPEG: complete JPEG file bytes; shared model excluded'), indent=2))
    fig, ax = plt.subplots(figsize=(8,5))
    jpeg = [r for r in summary if r['method'].startswith('JPEG')]
    ax.plot([r['actual_bpp'] for r in jpeg], [r['mean_psnr'] for r in jpeg], 'o-', label='JPEG')
    for row in jpeg:
        ax.annotate(row['method'].replace('JPEG ', ''), (row['actual_bpp'], row['mean_psnr']), xytext=(4,5), textcoords='offset points')
    dnn = summary[0]
    ax.scatter(dnn['actual_bpp'], dnn['mean_psnr'], marker='*', s=180, label='DNN')
    ax.set(xlabel='Bits per input pixel', ylabel='Mean PSNR (dB)')
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(output_dir/'rate_distortion.png', dpi=160); plt.show()
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot([r['actual_bpp'] for r in jpeg], [r['mean_ssim'] for r in jpeg], 'o-', label='JPEG')
    ax.scatter(dnn['actual_bpp'], dnn['mean_ssim'], marker='*', s=180, label='DNN')
    ax.set(xlabel='Bits per pixel', ylabel='Mean RGB SSIM')
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(output_dir/'rate_ssim.png', dpi=160); plt.show()
    for row in summary:
        print(f'{row["method"]:10s} {row["images"]:4d} images | {row["actual_bpp"]:.4f} bpp | {row["mean_psnr"]:.2f} dB | SSIM={row["mean_ssim"]:.4f}')
    print('DNN coding modes:', modes)
    print('DNN bpp counts compressed latent bytes only; JPEG bpp counts complete JPEG bytes.')
    print(f'Codewords used over evaluation set: {len(used)}/{model.config["codebook_size"]}')
    print('DNN estimated bpp:', dnn['estimated_bpp'])
    print('Rates may not overlap: this is not a matched-bitrate superiority claim.')
    return summary, saved


@torch.no_grad()
def visualize_decoded(model, stream_path, original_path, output_dir, jpeg_quality=30, *, psnr_fn=None, ssim_fn=None):
    """Last notebook visualization: decode actual saved bytes, not soft outputs."""
    score_psnr, score_ssim = metric_functions(psnr_fn, ssim_fn)
    output_dir = Path(output_dir)
    with Image.open(original_path) as image:
        original = np.asarray(image.convert('RGB'), dtype=np.float32)/255
    decoded = decompress(model, Path(stream_path).read_bytes())[0].cpu().permute(1,2,0).numpy()
    jpeg_payload, jpeg = jpeg_roundtrip(original, jpeg_quality)
    h,w = original.shape[:2]
    dnn_bpp = 8*payload_size(Path(stream_path).read_bytes())/(h*w)
    jpeg_bpp = 8*len(jpeg_payload)/(h*w)
    fig, axes = plt.subplots(2,3,figsize=(12,8))
    values = [original, decoded, jpeg]
    titles = ['Original', f'Decoded DNN latent\n{score_psnr(original,decoded):.2f} dB | {dnn_bpp:.3f} latent bpp\nSSIM {score_ssim(original,decoded):.4f}',
              f'JPEG Q{jpeg_quality}\n{score_psnr(original,jpeg):.2f} dB | {jpeg_bpp:.3f} file bpp\nSSIM {score_ssim(original,jpeg):.4f}']
    for ax, value, title in zip(axes[0], values, titles):
        ax.imshow(value); ax.set_title(title); ax.axis('off')
    # Same zoom coordinates for all methods; avoid selecting favorable details.
    y0,y1,x0,x1 = h//4,3*h//4,w//4,3*w//4
    axes[1,0].imshow(decoded[y0:y1,x0:x1]); axes[1,0].set_title('DNN decoded center zoom'); axes[1,0].axis('off')
    errors = [np.abs(original-decoded).mean(-1), np.abs(original-jpeg).mean(-1)]
    vmax = max(float(error.max()) for error in errors) or 1e-6
    for ax, error, title in zip(axes[1,1:], errors, ['DNN absolute RGB error', 'JPEG absolute RGB error']):
        im=ax.imshow(error, cmap='magma', vmin=0, vmax=vmax); ax.set_title(title); ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.suptitle('Reconstruction from the saved compressed file — different actual rates', fontsize=13)
    fig.tight_layout(); fig.savefig(output_dir/'decoded_image_comparison.png', dpi=180)
    Image.fromarray(np.rint(decoded*255).clip(0,255).astype(np.uint8)).save(output_dir/'decoded_image.png')
    print('Decoded from:', stream_path)
    print('Saved:', output_dir/'decoded_image.png')
    plt.show()
    return decoded
