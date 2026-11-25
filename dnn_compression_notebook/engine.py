"""Provided end-to-end training and hard-quantized evaluation."""
import csv
import json
import time
from pathlib import Path
import torch
from model import CompressionModel
from metrics import ssim_per_image


@torch.no_grad()
def evaluate(model, loader, device, rate_weight=0.01, *, psnr_fn=None, ssim_fn=None):
    model.eval()
    ssim_fn = ssim_per_image if ssim_fn is None else ssim_fn
    totals = dict(mse=0., psnr=0., ssim=0., estimated_bpp=0.)
    count = 0
    for x in loader:
        x = x.to(device)
        result = model(x)
        mse = (result['reconstruction'] - x).square().flatten(1).mean(1)
        n = len(x)
        totals['mse'] += mse.sum().item()
        totals['ssim'] += ssim_fn(result['reconstruction'], x).sum().item()
        psnr_values = (-10 * mse.clamp_min(1e-12).log10() if psnr_fn is None
                       else psnr_fn(result['reconstruction'], x))
        totals['psnr'] += psnr_values.sum().item()
        totals['estimated_bpp'] += result['estimated_bpp'].item() * n
        count += n
    if not count:
        raise ValueError('Empty evaluation loader')
    metrics = {k: v/count for k, v in totals.items()}
    metrics['loss'] = metrics['mse'] + rate_weight * metrics['estimated_bpp']
    return metrics


def load_checkpoint(path, soft_quantize, device='cpu'):
    state = torch.load(path, map_location=device, weights_only=True)
    model = CompressionModel(soft_quantize, **state['model_config']).to(device)
    coder_keys = {k for k in state['state_dict'] if k.startswith('coder_')}
    if coder_keys:
        if coder_keys != {'coder_left', 'coder_up'}:
            raise ValueError('Incomplete arithmetic context checkpoint')
        for key in sorted(coder_keys):
            model.register_buffer(key, state['state_dict'][key].clone().to(device))
    model.load_state_dict(state['state_dict'])
    return model.eval(), state


def fit(model, loaders, config, output_dir, device, *, psnr_fn=None, ssim_fn=None, loss_fn=None):
    output_dir = Path(output_dir)
    # Avoid silently replacing a previous experiment.
    output_dir.mkdir(parents=True, exist_ok=False)
    config = dict(config)
    (output_dir / 'config.json').write_text(json.dumps(config, indent=2))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    epochs, rate_weight = int(config['epochs']), float(config['rate_weight'])
    start, end = float(config['temperature_start']), float(config['temperature_end'])
    if epochs < 1 or min(start, end, config['lr']) <= 0 or rate_weight < 0:
        raise ValueError('Invalid epochs, temperatures, lr, or rate_weight')
    best, history = float('inf'), []
    model.to(device)
    start_time = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(epochs):
        tau = start * (end / start) ** (epoch / max(epochs-1, 1))
        model.train()
        total, count = 0., 0
        for x in loaders['train']:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            result = model(x, temperature=tau)
            if loss_fn is None:
                mse = (result['reconstruction'] - x).square().mean()
                loss = mse + rate_weight * result['estimated_bpp']
            else:
                loss = loss_fn(result, x, rate_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss; check quantization/temperature/lr')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
        val = evaluate(model, loaders['val'], device, rate_weight,
                       psnr_fn=psnr_fn, ssim_fn=ssim_fn)
        row = dict(epoch=epoch+1, temperature=tau, train_loss=total/count,
                   **{f'val_{k}': v for k, v in val.items()})
        history.append(row)
        if val['loss'] < best:
            best = val['loss']
            torch.save(dict(state_dict=model.state_dict(), model_config=model.config,
                            train_config=config, epoch=epoch+1, validation=val), output_dir/'best.pt')
        with (output_dir/'history.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader(); writer.writerows(history)
        torch.save(dict(state_dict=model.state_dict(), model_config=model.config,
                        train_config=config, epoch=epoch+1, validation=val,
                        optimizer=optimizer.state_dict()), output_dir/'last.pt')
        print(f"Epoch {epoch+1}/{epochs}: train={total/count:.5f}, hard val PSNR={val['psnr']:.2f}, SSIM={val['ssim']:.4f}, estimated bpp={val['estimated_bpp']:.4f}")
    with (output_dir/'history.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    resources = dict(device=str(device), training_seconds=time.perf_counter()-start_time,
                     parameter_count=sum(p.numel() for p in model.parameters()),
                     peak_allocated_gpu_mib=(torch.cuda.max_memory_allocated(device)/2**20
                                             if device.type == 'cuda' else None))
    (output_dir/'resources.json').write_text(json.dumps(resources, indent=2))
    print('Run resources:', resources)
    selected, _ = load_checkpoint(output_dir/'best.pt', model.soft_quantize, device)
    return selected, history
