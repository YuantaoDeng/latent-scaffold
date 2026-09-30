"""Lossless byte-plane integer-delta storage; no floating-point quantization."""
import os,zlib
from pathlib import Path
import torch

MARKER='_lossless_state_v1'
META='_lossless_metadata_v1'
def pack_meta(value):
    if isinstance(value,torch.Tensor):
        t=value.detach().cpu().contiguous();raw=t.reshape(-1).view(torch.uint8).reshape(-1,t.element_size()).T.contiguous().numpy().tobytes()
        blob=zlib.compress(raw,1);assert zlib.decompress(blob)==raw
        return {'_tensor_bytes_v1':blob,'dtype':str(t.dtype).split('.')[-1],'shape':list(t.shape)}
    if isinstance(value,dict):return {k:pack_meta(v) for k,v in value.items()}
    if isinstance(value,list):return [pack_meta(v) for v in value]
    if isinstance(value,tuple):return tuple(pack_meta(v) for v in value)
    return value
def unpack_meta(value):
    if isinstance(value,dict) and '_tensor_bytes_v1' in value:
        dtype=getattr(torch,value['dtype']);size=torch.empty((),dtype=dtype).element_size()
        raw=zlib.decompress(value['_tensor_bytes_v1'])
        if not raw:return torch.empty(value['shape'],dtype=dtype)
        bits=torch.frombuffer(bytearray(raw),dtype=torch.uint8)
        return bits.reshape(size,-1).T.contiguous().reshape(-1).view(dtype).reshape(value['shape'])
    if isinstance(value,dict):return {k:unpack_meta(v) for k,v in value.items()}
    if isinstance(value,list):return [unpack_meta(v) for v in value]
    if isinstance(value,tuple):return tuple(unpack_meta(v) for v in value)
    return value
def save_value(value,path):
    torch.save({META:True,'_value':pack_meta(value)},path,pickle_protocol=4)
def load(path,mmap=True,state_keys=None):
    path=Path(path);value=torch.load(path,map_location='cpu',weights_only=False,mmap=mmap)
    if isinstance(value,dict) and value.get(META):
        value={k:(v if k==MARKER else unpack_meta(v)) for k,v in value.items() if k!=META}
        if '_value' in value:return value['_value']
    if isinstance(value,dict) and MARKER in value:return decode(value,path.parent,state_keys)
    if state_keys is not None and isinstance(value,dict):
        if 'state' in value:value={**value,'state':{k:value['state'][k] for k in state_keys}}
        elif all(isinstance(v,torch.Tensor) for v in value.values()):value={k:value[k] for k in state_keys}
    return value
def decode(value,directory,state_keys=None):
    base=load(directory/value[MARKER]['base'],state_keys=state_keys)
    if 'state' in base:base=base['state']
    state={}
    for name,(kind,blob) in value[MARKER]['tensors'].items():
        if state_keys is not None and name not in state_keys:continue
        ref=base[name];size=ref.element_size()
        encoded=torch.frombuffer(bytearray(zlib.decompress(blob)),dtype=torch.uint8).reshape(size,-1).T.contiguous().reshape(-1)
        if kind=='delta32':
            zig=encoded.view(torch.int32);delta=((zig.to(torch.int64)&0xffffffff)>>1).to(torch.int32)^-(zig&1)
            bits=delta+ref.contiguous().reshape(-1).view(torch.int32)
            state[name]=bits.view(ref.dtype).reshape(ref.shape)
        else:
            bits=encoded^ref.contiguous().reshape(-1).view(torch.uint8)
            state[name]=bits.view(ref.dtype).reshape(ref.shape)
    return {**{k:v for k,v in value.items() if k!=MARKER},'state':state}
def save(value,path,base_path):
    path=Path(path);base_path=Path(base_path)
    base=load(base_path)
    if 'state' in base:base=base['state']
    packed={}
    assert set(base)==set(value['state'])
    for name,v in value['state'].items():
        ref=base[name];assert v.shape==ref.shape and v.dtype==ref.dtype
        if v.dtype==torch.float32:
            delta=v.contiguous().reshape(-1).view(torch.int32)-ref.contiguous().reshape(-1).view(torch.int32)
            encoded=((delta<<1)^(delta>>31)).view(torch.uint8);kind='delta32'
        else:
            encoded=v.contiguous().reshape(-1).view(torch.uint8)^ref.contiguous().reshape(-1).view(torch.uint8);kind='xor'
        raw=encoded.reshape(-1,v.element_size()).T.contiguous().numpy().tobytes()
        blob=zlib.compress(raw,1);assert zlib.decompress(blob)==raw
        packed[name]=(kind,blob)
    payload={**{k:pack_meta(v) for k,v in value.items() if k!='state'},META:True,MARKER:dict(base=os.path.relpath(base_path,path.parent),tensors=packed)}
    restored=decode(payload,path.parent)['state']
    for name,v in value['state'].items():
        assert torch.equal(v.contiguous().reshape(-1).view(torch.uint8),restored[name].contiguous().reshape(-1).view(torch.uint8)),name
    tmp=path.with_suffix('.writing.pt');torch.save(payload,tmp,pickle_protocol=4);tmp.replace(path)
