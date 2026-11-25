"""Optional instructor initialization using training images only."""
import torch
import time


def warm_start(model, loader, device, epochs=3, max_vectors=8192, iterations=20):
    """Warm up the continuous autoencoder, then initialize codewords with k-means.

    This does not replace the student's quantizer. Subsequent fit() still trains
    encoder, codebook, decoder, and prior jointly through that implementation.
    """
    started = time.perf_counter()
    model.to(device).train()
    optimizer = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=1e-3)
    losses = []
    for epoch in range(epochs):
        total, count = 0., 0
        for x in loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            y = model.decoder(model.encode(x))[..., :x.shape[-2], :x.shape[-1]]
            loss = (y-x).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach())*len(x); count += len(x)
        losses.append(total/count)
        print(f'Continuous warmup {epoch+1}/{epochs}: training MSE={total/count:.5f}')
    with torch.no_grad():
        vectors, count = [], 0
        for x in loader:
            z = model.encode(x.to(device)).permute(0,2,3,1).reshape(-1, model.codebook.shape[1])
            vectors.append(z); count += len(z)
            if count >= max_vectors: break
        points = torch.cat(vectors)[:max_vectors]
        k = model.config['codebook_size']
        if len(points) < k: raise ValueError('Need at least K training vectors')
        centers = points[torch.randperm(len(points), device=device)[:k]].clone()
        for _ in range(iterations):
            ids = torch.cdist(points, centers).argmin(1)
            for index in range(k):
                selected = points[ids == index]
                if len(selected): centers[index] = selected.mean(0)
                else: centers[index] = points[torch.randint(len(points), (), device=device)]
        model.codebook.copy_(centers)
        distances = torch.cdist(centers, centers).square()
        distances.fill_diagonal_(float('inf'))
        scale = max(float(distances.min(1).values.median()), 1e-6)
    return dict(warmup_epochs=epochs, warmup_seconds=time.perf_counter()-started,
                training_mse=losses, initialization_vectors=len(points),
                temperature_start=scale, temperature_end=scale*0.1)
