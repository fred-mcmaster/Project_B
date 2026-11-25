"""Lossless integer arithmetic coder for a raster of VQ codeword indices.

The codec selects the decoded index to the left or above as a causal context.
An optional combined-context mode is available for experiments. Encoder and
decoder start with the same supplied counts and update after each symbol.
When no trained counts are supplied, they start from the global prior.
"""
import numpy as np

BITS = 32
TOP = (1 << BITS) - 1
HALF = 1 << (BITS-1)
QUARTER = HALF >> 1
THREE_QUARTERS = QUARTER * 3
MAX_TOTAL = 1 << 15


def _initial_counts(k, probabilities, starting_counts=None):
    if starting_counts is not None:
        counts = np.asarray(starting_counts, dtype=np.int64)
        if counts.shape != (k+1, k) or (counts < 1).any():
            raise ValueError('Initial context counts must be positive [K+1,K]')
        counts = counts.copy()
        for row in counts:
            while row.sum() >= MAX_TOTAL:
                row[:] = np.maximum(1, (row + 1) // 2)
        return counts
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != (k,) or not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        raise ValueError('Need K finite nonnegative prior probabilities')
    p = p / p.sum()
    base = 1 + np.floor(p * 256).astype(np.int64)
    return np.broadcast_to(base, (k+1, k)).copy()


def _context(output, position, width, k, axis):
    if axis == 'left':
        return int(output[position-1]) if position % width else k
    if axis == 'up':
        return int(output[position-width]) if position >= width else k
    raise ValueError('axis must be left or up')


def _update(counts, context, symbol):
    row = counts[context]
    row[symbol] += 1
    if int(row.sum()) >= MAX_TOTAL:
        row[:] = np.maximum(1, (row + 1) // 2)


def encode(indices, k, probabilities, axis='left', starting_counts=None):
    source = np.asarray(indices, dtype=np.int64)
    if source.ndim != 2 or not source.size or (source < 0).any() or (source >= k).any():
        raise ValueError('Expected nonempty [H,W] indices in [0,K)')
    if axis not in ('left','up','both'):
        raise ValueError('axis must be left, up, or both')
    h, w = source.shape
    symbols = source.reshape(-1)
    if axis == 'both':
        if not isinstance(starting_counts, (tuple,list)) or len(starting_counts) != 2:
            raise ValueError('Both-context mode needs left and up count tables')
        left_counts = _initial_counts(k, probabilities, starting_counts[0])
        up_counts = _initial_counts(k, probabilities, starting_counts[1])
    else:
        counts = _initial_counts(k, probabilities, starting_counts)
    low, high, pending = 0, TOP, 0
    bits = []
    def emit(bit):
        nonlocal pending
        bits.append(bit)
        bits.extend([1-bit] * pending)
        pending = 0
    for pos, symbol in enumerate(symbols):
        if axis == 'both':
            left_context = _context(symbols,pos,w,k,'left')
            up_context = _context(symbols,pos,w,k,'up')
            row = left_counts[left_context] + up_counts[up_context]
        else:
            context = _context(symbols, pos, w, k, axis)
            row = counts[context]
        start = int(row[:symbol].sum())
        total = int(row.sum())
        span = high - low + 1
        high = low + span * (start + int(row[symbol])) // total - 1
        low = low + span * start // total
        while True:
            if high < HALF:
                emit(0)
            elif low >= HALF:
                emit(1)
                low -= HALF; high -= HALF
            elif low >= QUARTER and high < THREE_QUARTERS:
                pending += 1
                low -= QUARTER; high -= QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) | 1
        if axis == 'both':
            _update(left_counts,left_context,int(symbol))
            _update(up_counts,up_context,int(symbol))
        else:
            _update(counts, context, int(symbol))
    pending += 1
    emit(0 if low < QUARTER else 1)
    bits.extend([0] * BITS)  # Decoder can safely initialize/read its final bits.
    return np.packbits(np.array(bits, dtype=np.uint8)).tobytes()


def decode(payload, height, width, k, probabilities, axis='left', starting_counts=None):
    if min(height,width) < 1 or k < 2 or axis not in ('left','up','both') or not payload:
        raise ValueError('Invalid arithmetic stream geometry or mode')
    if axis == 'both':
        if not isinstance(starting_counts, (tuple,list)) or len(starting_counts) != 2:
            raise ValueError('Both-context mode needs left and up count tables')
        left_counts = _initial_counts(k, probabilities, starting_counts[0])
        up_counts = _initial_counts(k, probabilities, starting_counts[1])
    else:
        counts = _initial_counts(k, probabilities, starting_counts)
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    cursor = 0
    def read_bit():
        nonlocal cursor
        bit = int(bits[cursor]) if cursor < len(bits) else 0
        cursor += 1
        return bit
    code = 0
    for _ in range(BITS):
        code = (code << 1) | read_bit()
    low, high = 0, TOP
    result = np.empty(height*width, dtype=np.int64)
    for pos in range(len(result)):
        if axis == 'both':
            left_context = _context(result,pos,width,k,'left')
            up_context = _context(result,pos,width,k,'up')
            row = left_counts[left_context] + up_counts[up_context]
        else:
            context = _context(result, pos, width, k, axis)
            row = counts[context]
        cdf = np.empty(k+1, dtype=np.int64)
        cdf[0] = 0
        np.cumsum(row, out=cdf[1:])
        total = int(cdf[-1])
        span = high - low + 1
        scaled = ((code - low + 1) * total - 1) // span
        symbol = int(np.searchsorted(cdf, scaled, side='right') - 1)
        if not 0 <= symbol < k:
            raise ValueError('Corrupt arithmetic stream')
        result[pos] = symbol
        high = low + span * int(cdf[symbol+1]) // total - 1
        low = low + span * int(cdf[symbol]) // total
        while True:
            if high < HALF:
                pass
            elif low >= HALF:
                low -= HALF; high -= HALF; code -= HALF
            elif low >= QUARTER and high < THREE_QUARTERS:
                low -= QUARTER; high -= QUARTER; code -= QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) | 1
            code = (code << 1) | read_bit()
        if axis == 'both':
            _update(left_counts,left_context,symbol)
            _update(up_counts,up_context,symbol)
        else:
            _update(counts, context, symbol)
    return result.reshape(height,width)
