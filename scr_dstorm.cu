/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */

#include "dstorm_fwd.hpp"
#if WITH_GPU

#include "scr_dstorm.hpp"
#include "segVecGpu.hpp"
#include "lua-i.hpp"

#include "dstorm_any2.hh"       // required to get some inline templates instantiated here
                                // o.w. --> "missing symbol"
#include "segImpl.hh"           // other missing symbols ...
#include "dstorm_msg.hh"        // MsgHeader full declarations

#include "THC/THC.h"                // cutorch tensor support

//#include <unistd.h>             // sleep

using namespace dStorm;
using namespace std;

extern thread_local lua_Interpreter * d_si;
extern thread_local Dstorm * globalDstorm;

#define GD globalDstorm
using namespace dStorm;

int scr_Dstorm::gpu_add_segment_helper( Sint4 const segnum
                                        , Sint4 const ionet
                                        , Sint4 const segpolicy
                                        , Sint4 const nPoints
                                        , ccptr const type)
{
    scr_CNT;
    string t(type);
#define SegVec_t dStorm::seg::VecGpu
    if( t == "r4" )
    {
        cout<<" seg "<<segnum<<" --> "<<SegVec_t<float>::name<<"<float>"<<endl;
        GD->add_segment<SegVec_t<float> >( segnum, (IoNet_t)(ionet), (dStorm::SegPolicy)segpolicy, nPoints );
        // TODO lua side should have some segnum-->SegInfoPOD table
        // this could also be done be dynamic cast mechanism, I suppose.
        // (f_reduce uses the dynamic cast method, but maybe that's not so great)
    }
#if 0
    else if( t == "r8" )
    {

        if( GD->transport != GPU ){
            cout<<" seg "<<segnum<<" --> "<<SegVec_t<double>::name<<"<double>"<<endl;
            GD->add_segment<SegVec_t<double> >( segnum, (IoNet_t)(ionet), (dStorm::SegPolicy)segpolicy, nPoints );
        }
    }
#endif
#undef SegVec_t
    else
    {
        cout<<" BAD TYPE: "<<type<<" -- use r4 or r8"<<endl;cout.flush(); //sleep(1);
        Derr_msg( false, true, "type: "<<t<<" not supported --- use r4 or r8");
        goto ERR_LBL; // return 0; // ERROR
    }
    return 0; // huh? what are we returning... a PAIR of external-memory dpoints... really ???
    scr_ERR( "ERR: dstorm.add_segment( <segnum:int>, <ionet:int>, <segLayout:int>, <nPoints:int>, <point_type:string> ) -- point_type real4 or real8" );
}
template< typename SEG_FMT, typename CONST_ITER>
int scr_Dstorm::gpu_store_helper( SegInfo const& sInfo
                                  , CONST_ITER beg
                                  , uint32_t const cnt
                                  , uint32_t const off
                                  , double wgt )
{
    // Issue: GD->store could have beg iterator pointing to HOST or GPU memory.
    //        Dstorm API has so far assumed that 'beg' is
    //        "the same type of pointer as sInfo.mem"
    //   i.e. for SEG_FMT ~ seg::VecGpu,        'beg' points to GPU memory
    //   and  for SEG_FMT ~ seg::VecDense,      'beg' points to CPU memory
    //
    // Possible change:
    //   GD->store( segnum, HOST_MEMORY_POINTER, memoryType = MEMHOST )
    //   GD->store( segnum, HOST_MEMORY_POINTER, MEM_DEVICE )
    //
    // Current approach is to throw an error (in scr_dstorm.cpp) for unsupported conversions.
    //
    // This also forces lua code to be aware of when :cuda() | :float() | :double()
    // memory transfers are required.
    //
    GD->store<SEG_FMT,CONST_ITER>( sInfo.segNum, beg, cnt, off, wgt );
    return 0;
}

// instantiate 2 particular conversions for storing data into a Seg_VecGpu<T>
// ( FMT = seg::VecGpu<T>
template int scr_Dstorm::gpu_store_helper< seg::VecGpu<float>, float const* > 
( SegInfo const& sInfo, float const* beg, uint32_t const cnt, uint32_t const off, double wgt );
// this does not **seem** to instantiate cuda inlines? missing symbols were explicitly forced into libdstorm
// by explicit instantiation ... uggh.  (see dstorm.cu)

template int scr_Dstorm::gpu_store_helper< seg::VecGpu<float>, double const*> 
( SegInfo const& sInfo, double const* beg, uint32_t const cnt, uint32_t const off, double wgt );

template int scr_Dstorm::gpu_store_helper< seg::VecGpu<double>, float const* > 
( SegInfo const& sInfo, float const* beg, uint32_t const cnt, uint32_t const off, double wgt );

template int scr_Dstorm::gpu_store_helper< seg::VecGpu<double>, double const*> 
( SegInfo const& sInfo, double const* beg, uint32_t const cnt, uint32_t const off, double wgt );

#define TRY_GPU_REDUCE( SEGTYP ) do \
{ \
    typedef SEGTYP SegType; \
    SegType const* seg = dynamic_cast<SegType const*> ( &sInfo ); \
    if( seg != nullptr ){ \
        using dStorm::MsgHeader; \
        using dStorm::mem_as; \
        typedef SegType::Base::Tdata Tdata; /* e.g. float or double */ \
        RED_DBG( "sizeof(Tdata) = "<<sizeof(Tdata) ); \
        d_si->put_Sint4( (Sint4)nReduce ); /*-1U should be returned as a nice lua value*/ \
        if( nReduce == 0U || nReduce == -1U ){ \
            d_si->put_Uint4( (Uint4)0 ); /*mHdr->u.offset*/; \
            RED_DBG( " nReduce="<<nReduce ); \
            THFloatTensor_newWithSize1d(0/*empty*/); \
            return 3; \
        } \
        /* reduce output *always* goes into the iBuf  ~ "input buffer" */ \
        /* SEG_AVG_RBUF_OBUF averages rBufs and oBuf into the oBuf (iBuf == oBuf) */ \
        void * const outseg = seg->ptrBuf( sInfo.ibuf ); \
        TRY_REDUCE_DEBUG; \
        auto const mHdr = mem_as< MsgHeader<SegType::Fmt> * >(outseg); \
        uint_least32_t cnt = mHdr->hdr.u.cnt; \
        Uint4 off = mHdr->hdr.u.off; \
        /*auto base = static_cast<SegType::Base const*>(seg); a SegBase<...>  */ \
        /*Tdata* const data = mem_as<Tdata*>( outseg, sizeof(MsgHeader<SegType::Fmt>) );*/ \
        Tdata* const data = seg->data(sInfo.ibuf); \
        RED_DBG(" mHdr@"<<(void*)mHdr<<" cnt="<<cnt<<" off="<<off<<" sizeof(*mHdr)="<<sizeof(*mHdr) \
                <<"\n\t\t data@"<<(void*)data<<" data[0]="<<data[0]<<" data[1]="<<data[1]<<" data[2]="<<data[2] \
                <<"\n\t\t segnum="<<segnum<<" ibuf="<<sInfo.ibuf<<" obuf="<<sInfo.obuf<<" outseg[ibuf]@"<<(void*const)seg->ptrBuf(sInfo.ibuf)<<" outseg[obuf]@"<<(void*const)seg->ptrBuf(sInfo.obuf) ); \
        /* iBuf same as oBuf if iBuf segment was "removed" */ \
        d_si->put_Uint4( off ); \
        /*int ret = segbuf_torch_tensor1D<Tdata>( data, cnt );*/ /* LEAKS MEMORY -- fixme with lookup table */ \
        /*RED_DBG( "segbuf_torch_tensor1D<Tdata(size "<<sizeof(Tdata)<<")> returned "<<ret );*/ \
        /* Who defined 'real_t' ?? is the float/double from torch ??? */ \
        THFloatStorage * theStorage = THFloatStorage_newWithData(data, cnt); \
        if( theStorage == nullptr ) throw runtime_error("THFloatStorage_newWithData failed"); \
        THFloatStorage_clearFlag( theStorage, TH_STORAGE_FREEMEM ); \
        /* SIMPLE CASE -- 1-D data */ \
        THFloatTensor * theTensor =  THFloatTensor_newWithStorage1d(theStorage, /*storageOffset*/0, cnt, /*stride*/1); \
        THFloatStorage_free( theStorage ); \
        /* At this point, we have 'theTensor'. Now we need to push it onto the lua stack */ \
        luaT_pushudata( d_si->L, (void*)theTensor, "torch.FloatTensor" ); \
        /*luaT_pushudata(L, (void*)imgTensor, "torch.FloatTensor");*/ \
        /*lua_setglobal(L, "lv_imageTensor");*/ \
        return 3; \
    } \
}while(0)
#undef TRY_GPU_REDUCE
/** optional debug macro for TRY_REDUCE */
//#define RED_DBG(X) /*DEBUG*/ do { \
//    std::ostringstream oss; \
//    oss<<"GPU_RED_DBG: r"<<GD->iProc<<": " << X << "\n"; \
//    std::cout<<oss.str(); \
//    std::cout.flush(); \
//}while(0)
#define RED_DBG(X)
/** transfer the mHdr from GPU back to CPU.
 * a) we want to return [at least] the mHdr.u.off value
 * b) we need mHdr.u.cnt to set up some lua CudaXxxTensor return value
 */
template< typename SEGFMT > static typename ::dStorm::MsgHeader<SEGFMT>
copy_gpu_mHdr( ::dStorm::SegInfo const& sInfo )
{
    typedef typename ::dStorm::MsgHeader<SEGFMT> MHdr;
    typedef typename SEGFMT::type SegType;
    /* reduce output *always* goes into the iBuf  ~ "input buffer" */
#if 0 // actually, SegInfo::ptrBuf is all we need
    SegType const* seg = dynamic_cast<SegType const*> ( &sInfo );
    if( seg == nullptr ){
        throw std::runtime_error("Programmer error: bad guess for data type of seg::VecGpu<T>");
    }
    void * const outseg_gpu = seg->ptrBuf( sInfo.ibuf ); // we only need the void * version
#else
    void * const outseg_gpu = sInfo.ptrBuf( sInfo.ibuf ); // we only need the void * version
#endif
    /* SEG_AVG_RBUF_OBUF averages rBufs and oBuf into the oBuf (iBuf == oBuf) */
    //using dStorm::MsgHeader;
    //using dStorm::mem_as;
    //MHdr const* const mHdr_gpu = mem_as< MsgHeader<SegType::Fmt> * >(outseg_gpu);
    MHdr mHdr;
    //memset(&mHdr,0,sizeof(MHdr));
    checkCudaErrors(cudaMemcpy(&mHdr, outseg_gpu, sizeof(MHdr), cudaMemcpyDeviceToHost));
    // mHdr.hdr.a is "auto" stuff, that all MsgHeaders have
    // mHdr.hdr.u is "seg:FMT" stuff, that FMT::type store/push/reduce maintain (in principle)
    RED_DBG( "gpu reduce mHdr: .a={iter,bytes,fmt,pushes}={"
             <<mHdr.hdr.a.iter<<","<<mHdr.hdr.a.bytes<<","<<unsigned(mHdr.hdr.a.fmt)<<","<<unsigned(mHdr.hdr.a.pushes)
             <<"} .u={off,cnt,wgt,sz}={"
             <<mHdr.hdr.u.off<<","<<mHdr.hdr.u.cnt<<","<<mHdr.hdr.u.wgt<<","<<unsigned(mHdr.hdr.u.sz)
             <<"}");
    return mHdr;

}

static THCState* getCutorchState(lua_State* L)
{
	lua_getglobal(L, "cutorch");
        // alternative method: getfield(L,-1,"_state") is already a plain lightuserdata
	lua_getfield(L, -1, "getState");
	lua_call(L, 0, 1);
	THCState *state = (THCState*) lua_touserdata(L, -1);
	lua_pop(L, 2);
	return state;
}

int scr_Dstorm::gpu_reduce_helper( ::dStorm::SegInfo const& sInfo, uint32_t const nReduce )
{
    if( sInfo.fmtValue != seg::VecGpu<float>::value )
        throw std::runtime_error("Error: gpu_reduce_helper should only be called for a FMT ~ seg::VecGpu<T> segment");
    if( d_si->L == nullptr )
        throw std::runtime_error("Error: gpu_reduce_helper with no lua_State???");
    THCState *state = getCutorchState(d_si->L);
    if( state == nullptr )
        throw std::runtime_error("Error: gpu_reduce_helper failed to obtain TCHState (cutorch state, cutorch._state in lua)");


    d_si->put_Sint4( (Sint4)nReduce ); /*-1U should be returned as a nice lua value*/

    if( sInfo.datasize == 4 ){
        if( nReduce == 0U || nReduce == -1U ){
            d_si->put_Uint4( (Uint4)0 ); /*mHdr->u.offset*/;
            RED_DBG( " nReduce="<<nReduce );
            THCudaTensor_newWithSize1d(state,0/*empty*/); // return empty cuda float tensor
            return 3;
        }
        // define seg::FMT
        typedef seg::VecGpu<float> SegFmt;
        // Generic code
        auto mHdr = copy_gpu_mHdr<SegFmt>( sInfo );
        uint_least32_t const cnt = mHdr.hdr.u.cnt;
        Uint4 const off = mHdr.hdr.u.off;
        assert( mHdr.hdr.u.sz == sizeof(float) );
        assert( mHdr.hdr.u.sz == sInfo.datasize );
        //
#if 1
        // Generic types
        typedef SegFmt::type SegType;           // i.e. Seg_VecGpu<float>
        typedef SegType::Base::Tdata Tdata;     // i.e. float
        // alt. with sInfo.ptrIdata() returns void*
        SegType const* seg = dynamic_cast<SegType const*> ( &sInfo );
        if( seg == nullptr ){
            throw std::runtime_error("Programmer error: bad guess for data type of seg::VecGpu<T>");
        }
        Tdata * data = seg->data(sInfo.ibuf); // well it's constant, but cutorch doesn't know that
        static_assert( sizeof(Tdata) == sizeof(float), "oops");
#else
        typedef SegFmt::type SegType;           // i.e. Seg_VecGpu<float>
        typedef SegType::Base::Tdata Tdata;     // i.e. float
        static_assert( sizeof(Tdata) == sizeof(float), "oops");
        float * data = static_cast<float *>(sInfo.ptrIdata());  // THCudaStorage cannot handle float const*
#endif
        d_si->put_Uint4( off );
        THCudaStorage * theStorage = THCudaStorage_newWithData(state, data, cnt);
        if( theStorage == nullptr ) throw runtime_error("THCudaStorage_newWithData failed");
        THCudaStorage_clearFlag(state, theStorage, TH_STORAGE_FREEMEM );
        /* SIMPLE CASE -- 1-D data */
        THCudaTensor * theTensor =  THCudaTensor_newWithStorage1d(state,theStorage, /*storageOffset*/0, cnt, /*stride*/1);
        THCudaStorage_free(state, theStorage );
        /* At this point, we have 'theTensor'. Now we need to push it onto the lua stack */
        luaT_pushudata( d_si->L, (void*)theTensor, "torch.CudaTensor" );
        return 3;

#if 0
#ifndef NDEBUG
    {
        // can't access mHdr on host, so can't check that
        if( sInfo.policy & dStorm::REDUCE_OP_MASK) == dStorm::REDUCE_AVG_RBUF_OBUF )
            assert( sInfo.ibuf == sInfo.obuf );

    }
#endif
    uint_least32_t cnt = mHdr->hdr.u.cnt;
    Uint4 off = mHdr->hdr.u.off;
    /*auto base = static_cast<SegType::Base const*>(seg); a SegBase<...>  */
    typedef SegType::Base::Tdata Tdata; /* e.g. float or double */
    RED_DBG( "sizeof(Tdata) = "<<sizeof(Tdata) );
    /*Tdata* const data = mem_as<Tdata*>( outseg, sizeof(MsgHeader<SegType::Fmt>) );*/
    Tdata* const data = seg->data(sInfo.ibuf);
    RED_DBG(" mHdr@"<<(void*)mHdr<<" cnt="<<cnt<<" off="<<off<<" sizeof(*mHdr)="<<sizeof(*mHdr)
            <<"\n\t\t data@"<<(void*)data<<" data[0]="<<data[0]<<" data[1]="<<data[1]<<" data[2]="<<data[2]
            <<"\n\t\t segnum="<<segnum<<" ibuf="<<sInfo.ibuf<<" obuf="<<sInfo.obuf<<" outseg[ibuf]@"<<(void*const)seg->ptrBuf(sInfo.ibuf)<<" outseg[obuf]@"<<(void*const)seg->ptrBuf(sInfo.obuf) );
    /* iBuf same as oBuf if iBuf segment was "removed" */
    d_si->put_Uint4( off );
    /*int ret = segbuf_torch_tensor1D<Tdata>( data, cnt );*/ /* LEAKS MEMORY -- fixme with lookup table */
    /*RED_DBG( "segbuf_torch_tensor1D<Tdata(size "<<sizeof(Tdata)<<")> returned "<<ret );*/
    /* Who defined 'real_t' ?? is the float/double from torch ??? */
    THFloatStorage * theStorage = THFloatStorage_newWithData(data, cnt);
    if( theStorage == nullptr ) throw runtime_error("THFloatStorage_newWithData failed");
    THFloatStorage_clearFlag( theStorage, TH_STORAGE_FREEMEM );
    /* SIMPLE CASE -- 1-D data */
    THFloatTensor * theTensor =  THFloatTensor_newWithStorage1d(theStorage, /*storageOffset*/0, cnt, /*stride*/1);
    THFloatStorage_free( theStorage );
    /* At this point, we have 'theTensor'. Now we need to push it onto the lua stack */
    luaT_pushudata( d_si->L, (void*)theTensor, "torch.FloatTensor" );
    /*luaT_pushudata(L, (void*)imgTensor, "torch.FloatTensor");*/
    /*lua_setglobal(L, "lv_imageTensor");*/
    return 3;
#endif

    }else if( sInfo.datasize == 8 ){
        if( nReduce == 0U || nReduce == -1U ){
            d_si->put_Uint4( (Uint4)0 ); /*mHdr->u.offset*/;
            RED_DBG( " nReduce="<<nReduce );
            THCudaDoubleTensor_newWithSize1d(state,0/*empty*/); // return empty cuda float tensor
            return 3;
        }
        auto mHdr = copy_gpu_mHdr<seg::VecGpu<double>>( sInfo );
    }else{
        throw std::runtime_error("Unknown numeric type T for seg::VecGpu<T> in gpu_reduce_helper!");
    }
    return 1;
}
#endif // WITH_GPU
