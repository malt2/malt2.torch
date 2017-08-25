-- require "torch"    -- malt2 pulls this in
require("malt2")
-- require "malt2.dstorm2"
--print_table(package.loaded)
print(dstorm)

-- bump up default transport layer to latest/greatest available
print(" dstorm.WITH_MPI == "..dstorm.WITH_MPI)
print(" dstorm.WITH_GPU == "..dstorm.WITH_GPU)

if dstorm.WITH_GPU > 0 then
  -- could this also be done by require("malt2") ?
  require('cutorch')
  local deviceId = 1
  cutorch.setDevice(deviceId)
  print("cutorch getMemoryUsage: "..cutorch.getMemoryUsage(deviceId))
end

if dstorm.WITH_MPI ~= 0 then
  default_transport="mpi"
end
if dstorm.WITH_GPU ~= 0 then
  default_transport="gpu"
end
print("---------------------------------------------------")
print(" #arg          : "..#arg)
print("  arg[0]       : "..arg[0])
doHelp = 0
if default_transport == nil then
  doHelp = 1
else
  if #arg == 0 then
    print("default_transport: "..tostring(default_transport))
    transportName = default_transport
  elseif #arg > 1 then
    print("  arg[1] (too many args) : "..arg[1])
    doHelp = 1
  else
    print("  arg[1] (OK)  : "..arg[1])
    transportName = arg[1]
  end
end
if doHelp == 1 then
  print ("Usage: ")
  print ("       [mpirun [OPTS]...] {torch_executable} "..arg[0].." [gpu|mpi|shm]")
  assert(false)
end

print("Initializing dstorm for transportName : "..tostring(transportName))
-- d = dstorm() --global dStorm object
dstorm.init(transportName)
print("Initializing dstorm for "..tostring(transportName).." DONE")
dstorm.barrier()
print(" iProc         : " .. dstorm.iProc())
print(" nProc         : " .. dstorm.nProc())
print(" transport     : " .. dstorm.getTransport()) -- as string value ( mpi | gpu )

--
-- dstorm GPU transport needs a CUDA storage for the Dstorm::store(seg,iter,...) call
--
defaultTensorType = "torch.FloatTensor"
if false and dstorm.getTransport() == "gpu" then
  print(" pulling in cutorch now...")
  require("cutorch")
  print(" trying torch.CudaTensor...")
  defaultTensorType = "torch.CudaTensor"
  -- this does not seem to work
end
torch.setdefaulttensortype(defaultTensorType)

require('pl')
if true then -- play with tensor types a little
  print("\n torch.Tensor(3,3) storage demo")
  t=torch.Tensor(3,3):uniform(-1,1)
  print("t=torch.Tensor(3,3) --> "..type(t).."\n"..tostring(t))
  print("t:size():\n"..tostring(t:size()))
  --pretty.dump(t)
  s=t:storage()
  print("s=t:storage()   = "..tostring(s))
  print("s:data()        = "..tostring(s:cdata()))
  print("s:cdata()       = "..tostring(s:cdata()))
  print("s:size()        = "..tostring(s:size()))
  print("s:elementSize() = "..tostring(s:elementSize()))
  print("\n")
  -- conversions:
  --   FloatTensor:cuda() ---> CudaTensor
  --   CudaTensor:float() ---> xfer back to host, as FloatTensor
  -- xxxTensor:storage():data() pointers are different for cuda() or float() tensor conversions
  --
  -- so let's just use cuda() (and try to gc the temporary copy) for GPU transport ...
end

iProc = dstorm.iProc()
nProc = dstorm.nProc()

-- New: dstorm object also has some "static const" capabilities fields:
--   dstorm.WITH_{LIBORM|NOTIFYACK|MPI|GPU}
-- In lua both a 0 and 1 value evaluate as "true", so compare explicitly with 0 for false:
if not dstorm.WITH_NOTIFYACK==0 then
  notify_ack = Seg.SEGSYNC_NOTIFY_ACK
  print("notify_ack available")
else
  -- this script should still work, because we have inserted dstorm.barrier()
  -- "all over the place" (to make log messages appear in better order)
  notify_ack = Seg.SEGSYNC_NONE
  print("notify_ack unavailable")
end
-- Reduce policy
--
--   (if not supported, transports are free to use a higher level of synchronization)
local Red = { ORIG = Seg.REDUCE_AVG_RBUF,
    OBUF = Seg.REDUCE_AVG_RBUF_OBUF,
-- NTACK uses ASYNC consistency setting, but provides guarantee stronger the SYNC.
-- NTACK is "stronger than SYNC": wait for ALL rcvlist, reduce, and ACK
--                                AND guarantees no mixed-version.
-- quick MNIST test shows 13% faster and a bit more accurate
--                  than Red.OBUF with SYNC.
    NTACK = Seg.REDUCE_AVG_RBUF_OBUF + notify_ack
}

dim = 8388700;
-- dim = 500
seg_idx=99
print("Ready to add_segment("..seg_idx..", IoNet.ALL, Seg.FULL+Red.ORIG+notify_ack, dim="..dim..", \"r4\"")
dstorm.barrier()
-- dstorm.add_segment(seg_idx, IoNet.ALL, Seg.FULL + Red.NTACK, dim, "r4")   -- OK
dstorm.add_segment(seg_idx, IoNet.ALL, Seg.FULL + Red.ORIG + notify_ack, dim, "r4") -- fixed, now.
print("iProc="..iProc.." : Back from add_segment")
if iProc == 0 then
  print(" for seg_idx = "..seg_idx.." ...")
  print(" getIoNet           "..dstorm.getIoNet(seg_idx))
  print(" getPolicy          "..dstorm.getPolicy(seg_idx))
  print(" getSegNum          "..dstorm.getSegNum(seg_idx))
  print(" getObuf            "..dstorm.getObuf(seg_idx))
  print(" getIbuf            "..dstorm.getIbuf(seg_idx))
  print(" getRbuf            "..dstorm.getRbuf(seg_idx))
  print(" getNbuf            "..dstorm.getNbuf(seg_idx))
  print(" getBufBytes        "..dstorm.getBufBytes(seg_idx))
  print(" getSegBytes        "..dstorm.getSegBytes(seg_idx))
  -- segfault for cuda here:
  -- print(" getMem             "..type(dstorm.getMem(seg_idx)))       -- lua sees the raw ccptr as "string"
  print(" getMem             "..tostring(dstorm.getMem(seg_idx)))       -- changed to return Uint8 version of intptr_t(sInfo.mem)
  print(" getDatacode        "..dstorm.getDatacode(seg_idx))
  print(" getDatasize        "..dstorm.getDatasize(seg_idx))
  print(" getCnt             "..dstorm.getCnt(seg_idx))
  print(" getSizeofMsgHeader "..dstorm.getSizeofMsgHeader(seg_idx))
  print(" getSeg_id          "..dstorm.getSeg_id(seg_idx))
  print(" getFmtValue        "..dstorm.getFmtValue(seg_idx))
  if dstorm.getValid(seg_idx) then
    print(" segment          is valid")
  else
    print(" segment          is invalid")
  end
end
dstorm:barrier()
-- if iProc==0 then print(ionet.help()) end


--mynet = IoNet.HALTON                  -- no cvgce for 8, 12 machines
-- mynet = IoNet.ALL                     -- nice cvgce
-- mynet = ionet.halton(nProc,nProc/2) -- asim fixme

print("--------------------Segment created-------------------------")

t1 = torch.Tensor(dim)
t1:fill(iProc)
-- print (t1)
--t1:round((dstorm.iProc()+1), (dstorm.iProc()+1))
print (torch.type(t1:storage())) -- Ex. type --> userdata     torch.type --> torch.FloatStorage
dstorm.barrier()

if dstorm.getTransport() == "gpu" then
  print("cutorch getMemoryUsage: "..cutorch.getMemoryUsage(deviceId))
  local t1cuda = t1:cuda()                -- copy host data to gpu (temporarily)
  dstorm.store(seg_idx, t1cuda:storage())
  t1cuda=nil
  print("cutorch getMemoryUsage: "..cutorch.getMemoryUsage(deviceId))
else
  dstorm.store(seg_idx, t1:storage())
end
dstorm.barrier()

--send_bytes = d:push(seg_g, IoNetEnum.CHORD )
send_bytes = dstorm.push(seg_idx )
print("["..iProc.."] ---<store,push>---seg_idx, send_bytes ------------------ "..seg_idx..", "..send_bytes)
dstorm.barrier()

-- print("--------t1 = d:get_copy_tensor(seg_g)---------------------")
-- t1 = d:get_copy_tensor(seg_g,1,1)
-- print(t1)

-- print("["..iProc.."] --------t2 = average(seg_g)------------------------------------------- ")
-- t2 = dstorm.reduce(seg_idx)
-- print(t2)

--     CHANGE in lua return values ... (fix later???)
-- WITH_GPU, returning a TorchTensor from the dstorm.reduce(seg_idx) would be very costly
-- should re-introduce returning a [fast] CudaTensor, or CudaDoubleTensor
-- nreduce = dstorm.reduce(seg_idx)
nreduce, offset, t2 = dstorm.reduce(seg_idx)
dstorm.barrier()
if t2 == nil then
  print("["..iProc.."] ---<reduce>-- nreduce = ----------------------- "..nreduce)
else
  print("["..iProc.."] ---<reduce>-- nreduce, offset = ----------------------- "..nreduce..", "..offset..", "..type(t2).." t2:type="..t2:type())
  --if iProc == 0 then pretty.dump(getmetatable(t2)) end
end

dstorm.barrier()
--print("["..iProc.."] --- reduce output, t2 = ---------" .. tostring(t2))
--dstorm.barrier()
--print("["..iProc.."] --- t2[1] = "..t2[1])   -- torch tensor indices begin at 1
--dstorm.barrier()

-- check result correctness, depending on reduction type
--   Note: assumes full barriers, or SEGSYNC_NOTIFY_ACK to "all rbufs" were reduced
--   FIXME: should be able to handle arbitrary ionet (also a good demo)
if t2 == nil then
  print("reduce is not yet returning any tensor view to lua -- SKIPPING correctness check")
else
  print ("********************Checking for correct reduce values for dim size " .. dim)
  for k = 1, dim do
  redux = t2[k]
  segPolicy = dstorm.getPolicy(seg_idx)
  reduxPolicy = bit.band(segPolicy,Seg.REDUCE_OP_MASK)
  if dstorm.getIoNet(seg_idx) == IoNet.ALL then
    sumOfIprocs = nProc * (nProc-1) / 2  -- iProc in zero..nProc-1
    -- if iProc == 0 then print("nProc = "..nProc.." sumOfIprocs="..sumOfIprocs) end
    if reduxPolicy == Seg.REDUCE_AVG_RBUF then
      expec = (sumOfIprocs - iProc) / (nProc-1)
      -- print(" REDUCE_AVG_RBUF iProc="..iProc.." expect avg of {other iProcs} = "..expec.." and got "..redux .. "for index "..k)
      assert( redux == expec );
      -- print(" Excellent -- REDUCE_AVG_RBUF result was CORRECT");
    else
      if reduxPolicy == Seg.REDUCE_AVG_RBUF_OBUF then
        expec = sumOfIprocs / nProc
        -- print(" REDUCE_AVG_RBUF iProc="..iProc.." expect avg of {all iProcs} = "..expec.." and got "..redux .. "for index "..k)
        assert( redux == expec );
        -- print(" Excellent -- REDUCE_AVG_RBUF_OBUF result was CORRECT");
      else
        print(" OHOH -- I don't know how to test correctness of segment reduce policy "..reduxPolicy)
      end
    end
  else
    print(" OHOH - Please implement result checks for IoNet type #"..dstorm.getIoNet(seg_idx))
  end
  end
end

dstorm.barrier()
-- t2 = dstorm:get_copy_tensor(seg_idx, dstorm.getRbuf(seg_idx))
-- print("["..iProc.."] --- tensor? -------------------------------------- "..t2)

dstorm.delete_segment(seg_idx)
dstorm.barrier()
-- print("--------t3 = t1 + t2----------------------------------------")
-- t3 = t1+t2
-- t3:pprint()
-- print("vec():size():"..t3:vec():size())

-- local copy = d:get_copy_tensor(seg_g,1);
os.exit()
