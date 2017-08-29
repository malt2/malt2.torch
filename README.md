malt2.torch
============
malt-dstorm2 for torch packages

See the [malt website](https://malt2.github.io) for more details about MALT2.

1) Source your torch/cuda/MKL environment:
   on some machines, you might need things something like:
       source [torch-dir]/install/bin/torch-activate
       source /opt/intel/mkl/bin/intel64/mklvars.sh intel64
 

2) Before installing this torch module, you must compile dstorm and liborm)
   First, checkout the parent module:
   
   ```
   git clone https://github.com/malt2/malt2 --recursive
   ```
   and type make. 
   
   
3) Run a quick test.

   - if MPI, then you'll need to run via mpirun, perhaps something like:
         mpirun -np 3 `which th` `pwd -P`/test.lua mpi 2>&1 | tee test-mpi.log

   - if GPU,
         mpirun -np 3 `which th` `pwd -P`/test.lua gpu 2>&1 | tee test-GPU-gpu.log
     - NEW: a WITH_GPU compile can also run with MPI transport
         mpirun -np 3 `which th` `pwd -P`/test.lua mpi 2>&1 | tee test-GPU-mpi.log

   - default transport is set to the "highest" built into libdstorm2:
     - GPU > MPI >  SHM
         mpirun -np 3 `which th` `pwd -P`/test.lua 2>&1 | tee test-best.log

   - a very basic test is to run luajit (or th) and then try, by hand,
         require "torch"
         require "malt2"

4) Run over multiple GPUs.
    MPI only sees the hostname. By default, on everyhost, MPI jobs enumerate the
    GPUs and start running the processes. The only way to change this and run on
    other GPUs in a round-robin fashion is to change this enumeration for every
    rank using CUDA_VISIBLE_DEVICES. An example script is in redirect.sh. 
    To run:
        mpirun -np 2 ./redirect.sh `which th` `pwd`/test.lua

    This script assigns available GPUs in a round-robin fashion. Since MPI requires
    visibility of all other GPUs to correctly access shared memory, this script only
    changes the enumeration order and does not restrict visibility.
    

TODO:
  cuda code has too many kernel invocations (can be streamlined)
  lua interface support for CudaDoubleTensor and DoubleTensor likely incomplete
  other improvements documented in source code
