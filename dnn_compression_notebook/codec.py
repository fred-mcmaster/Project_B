"""Decodable latent bitstreams: legacy zlib or trained-context arithmetic coding.

ECV1 decodes earlier zlib files. ECV2 chooses the smallest latent payload among
zlib, causal left-context arithmetic, and causal up-context arithmetic. Header
and shared checkpoint bytes are excluded from reported latent-only BPP.
"""
import hashlib
import json
import math
import struct
import zlib
import numpy as np
import torch
from arithmetic import encode as arithmetic_encode, decode as arithmetic_decode


def model_id(model, legacy=False):
    digest = hashlib.sha256(json.dumps(model.config, sort_keys=True).encode())
    for name, value in sorted(model.state_dict().items()):
        if legacy and name.startswith('coder_'):
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _parts(stream):
    if stream[:4] not in (b'ECV1',b'ECV2') or len(stream) < 8:
        raise ValueError('Invalid teaching bitstream')
    length = struct.unpack('>I', stream[4:8])[0]
    if length < 1 or 8+length > len(stream):
        raise ValueError('Invalid bitstream header length')
    return stream[:4], json.loads(stream[8:8+length]), stream[8+length:]


def payload_size(stream):
    """Number of compressed latent bytes, excluding all container/header bytes."""
    return len(_parts(stream)[2])


def _packed_indices(ids, width):
    flat = np.asarray(ids, dtype=np.uint64).reshape(-1)
    bits = ((flat[:, None] >> np.arange(width-1, -1, -1, dtype=np.uint64)) & 1).astype(np.uint8)
    return np.packbits(bits.reshape(-1)).tobytes()


def _prior(model):
    return torch.softmax(model.prior_logits.detach().float().cpu(),dim=0).numpy()


@torch.no_grad()
def compress(model, x):
    if x.ndim != 4 or x.shape[0] != 1 or x.shape[1] != 3:
        raise ValueError('compress expects one RGB image [1,3,H,W]')
    model.eval()
    z = model.encode(x)
    flat = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
    ids = model.hard_indices(flat).cpu().numpy().astype(np.uint64).reshape(z.shape[-2],z.shape[-1])
    width = math.ceil(math.log2(model.config['codebook_size']))
    packed = _packed_indices(ids,width)
    zlib_payload = zlib.compress(packed)
    trained = hasattr(model,'coder_left') and hasattr(model,'coder_up')
    if trained:
        k=model.config['codebook_size']
        prior=_prior(model)
        choices=[zlib_payload,
                 arithmetic_encode(ids,k,prior,'left',model.coder_left.detach().cpu().numpy()),
                 arithmetic_encode(ids,k,prior,'up',model.coder_up.detach().cpu().numpy())]
        coding_mode=min(range(len(choices)),key=lambda i:len(choices[i]))
        payload=choices[coding_mode]
        magic=b'ECV2'
    else:
        payload=zlib_payload
        magic=b'ECV1'
    header=dict(height=x.shape[-2], width=x.shape[-1], zh=z.shape[-2],
                zw=z.shape[-1], bits=width, model_id=model_id(model))
    if trained:
        header['c']=coding_mode
        header['crc32']=zlib.crc32(packed)
    header=json.dumps(header,separators=(',', ':')).encode()
    return magic+struct.pack('>I',len(header))+header+payload


@torch.no_grad()
def decompress(model, stream):
    model.eval()
    magic,header,payload=_parts(stream)
    expected=model_id(model,legacy=(magic==b'ECV1'))
    if header['model_id'] != expected:
        raise ValueError('Decoder checkpoint does not match encoder checkpoint')
    h,w,zh,zw,bits=[header[k] for k in ('height','width','zh','zw','bits')]
    if min(h,w,zh,zw)<=0 or (zh,zw)!=(math.ceil(h/model.factor),math.ceil(w/model.factor)):
        raise ValueError('Invalid shape header')
    k=model.config['codebook_size']
    if bits!=math.ceil(math.log2(k)):
        raise ValueError('Invalid symbol width')
    if magic==b'ECV1' or header['c']==0:
        packed=zlib.decompress(payload)
        count=zh*zw
        if len(packed)!=math.ceil(count*bits/8):
            raise ValueError('Invalid payload size')
        matrix=np.unpackbits(np.frombuffer(packed,dtype=np.uint8))[:count*bits].reshape(count,bits)
        ids=(matrix.astype(np.int64)*(1<<np.arange(bits-1,-1,-1))).sum(1).reshape(zh,zw)
    else:
        if not (hasattr(model,'coder_left') and hasattr(model,'coder_up')):
            raise ValueError('Arithmetic bitstream requires trained context checkpoint')
        if header['c'] not in (1,2):
            raise ValueError('Invalid arithmetic mode')
        axis='left' if header['c']==1 else 'up'
        counts=getattr(model,'coder_'+axis).detach().cpu().numpy()
        ids=arithmetic_decode(payload,zh,zw,k,_prior(model),axis,counts)
    if ids.max()>=k:
        raise ValueError('Invalid codebook index')
    if magic==b'ECV2' and zlib.crc32(_packed_indices(ids,bits))!=header['crc32']:
        raise ValueError('Corrupt latent payload')
    indices=torch.tensor(ids,device=model.codebook.device).reshape(1,zh,zw)
    return model.decode_indices(indices,h,w)
