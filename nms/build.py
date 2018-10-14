import os
import torch
from torch.utils.ffi import create_extension
from IPython import embed

include_dirs = []
library_dirs = []


sources = ['src/nms.cpp']
headers = ['src/nms.h']
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    cuda_home = os.getenv('CUDA_PATH')
    include_dirs.append('%s/include'%cuda_home)

    for base_dir in ['lib/x64', ]:
        absolute_dir = os.path.join(cuda_home, base_dir)
        if os.path.exists(absolute_dir):
            library_dirs.append(absolute_dir)

        #thlibs = 'TH,THC,THCS,THCUNN,THNN,THS'
        thlibs = '_C,caffe2,caffe2_gpu'
        thlibs = thlibs.split(',')
        _pth = r'D:\pylibs\pytorch-scripts\pytorch\torch\lib'
        for e in thlibs:
            _val = '%s/%s.lib'%(_pth, e)
            assert os.path.exists(_val), _val
            library_dirs.append(_val)

    sources += ['src/nms_cuda.cpp', ]
    headers += ['src/nms_cuda.h',]
    defines += [('WITH_CUDA', None)]
    with_cuda = True

this_file = os.path.dirname(os.path.realpath(__file__))
#print(this_file)
extra_objects = [r'D:\dnnLibs\pytorch-mask-rcnn\nms\src\cuda\nms_kernel.cu.obj',]
#extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

#print( library_dirs)
library_dirs=[r'D:\pylibs\pytorch-scripts\pytorch\torch\lib','D:\dnnLibs\pytorch-mask-rcnn\nms\src\cuda',]
libraries=['_C','caffe2', 'caffe2_gpu','cudart'] 
#libraries=['nms_kernel',] 


ffi = create_extension(
    '_ext.nms',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__ ,
    with_cuda=with_cuda,
    include_dirs = include_dirs,
    libraries = libraries,
    extra_objects=extra_objects
)



if __name__ == '__main__':
    ffi.build()
