malt2.torch
============
malt-dstorm2 for torch packages

1) Source your torch/cuda/MKL environment:
   on some machines, you might need things something like:
       source [torch-dir]/install/bin/torch-activate
       source /opt/intel/mkl/bin/intel64/mklvars.sh intel64
   on snake10/snake08, you can try
       module install icc cuda80 luajit

2) Before installing this torch module, you must compile dstorm and liborm)
       cd ../dstorm
       make exports
       # add the suggested export lines to your environment
       ./mkit.sh <type> test
   where TYPE is : 
                 ORM (liborm )
              or MPI (liborm [+orm] + mpi)
              or GPU (liborm + mpi + gpu)
   A side effect is to create ../dstorm-env.{mk|cmake} environment files, so lua capabilities
   can match the libdstorm compile options.

3) You can install the package by opening a terminal, changing directory into the folder and typing:
       luarocks make
   or perhaps force a full recompile with
       rm -rf build && VERBOSE=7 luarocks make malt-2-scm-1.rockspec >& mk.log && echo YAY

5) Run a quick test.

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

6) Run over multiple GPUs.
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
