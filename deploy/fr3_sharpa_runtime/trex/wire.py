"""NumPy-version-independent arrays for the trusted loopback pickle envelope."""
import math
import numpy as np

TAG = '__trex_ndarray_v1__'
MAX_BYTES = 32 * 1024 * 1024


def encode(value):
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.kind not in 'biuf' or array.nbytes > MAX_BYTES:
            raise ValueError('Unsupported wire array type/size')
        return {TAG: True, 'dtype': array.dtype.str,
                'shape': list(array.shape), 'data': array.tobytes()}
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [encode(v) for v in value]
    return value


def decode(value):
    if isinstance(value, dict):
        if value.get(TAG) is True:
            dtype, shape, data = np.dtype(value['dtype']), value['shape'], value['data']
            if (dtype.kind not in 'biuf' or not isinstance(shape, list) or len(shape)>8
                    or any(type(n) is not int or n < 0 for n in shape)
                    or not isinstance(data, bytes) or len(data)>MAX_BYTES
                    or math.prod(shape)*dtype.itemsize != len(data)):
                raise ValueError('Malformed wire array')
            return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list): return [decode(v) for v in value]
    return value
