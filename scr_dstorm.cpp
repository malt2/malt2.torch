/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
/** \file
 * torch lua interface to Dstorm (libmalt2)
 *
 * - lua Tensor includes are in someplace like
 *   - /local/kruus/torch/install/include/TH/ (or .../THC/)
 * - Source code for lua tensor types and allocators are in
 *   - /local/kruus/torch/extra/cutorch/lib/THC
 *   - /local/kruus/torch/pkg/torch/lib/TH
 */
//#if ! WITH_GPU
#include "segVecDense.hh"
// NEW: we can no longer use SegVec_t, because lua client is allowed to ask for **ANY** Transport<>
//
//#define SegVec_t dStorm::seg::VecDense
//#else
#include "segVecGpu.hpp"
//#include "segVecGpu.cuh"
//#define SegVec_t dStorm::seg::VecGpu
//#endif

#include "scr_dstorm.hpp"
#include "scr_ionet.hpp"
#include "mtypes.hpp"           // scr_CNT, etc.
#include "lua-i.hpp"
#include "demangle.hpp"
#include <unistd.h>             // sleep

#include "TH/TH.h"
#include "TH/THGeneral.h"
#include "TH/THTensor.h"

#if WITH_GPU
//#include "THC/THCStorage.h"
#include "THC/THCTensor.h"
#endif

#include <segInfo.hpp>
#include <dstorm.hpp>
#include <dstorm_msg.hpp>       // seg::type-metainfo-classes

// for segVecGpu to be complete...
//#include <segInfo.hh>
//#include <segImpl.hh>

using namespace dStorm;
using namespace std;

extern thread_local lua_Interpreter * d_si;
extern thread_local Dstorm * globalDstorm;

#define GD globalDstorm

static Dstorm * mkGlobalDstorm(char const* optString )
{
    char const * defaultOpt =

#if WITH_GPU==1
        "gpu"
#elif WITH_MPI==1
        "mpi"
#else
        "shm" // kinda broken
#endif

        ;
    string opt (optString? optString: defaultOpt);
    printf ("optString ==> %s", optString);

    // any new transport gets an option here
    bool const opt_gpu   = (icompare(opt.substr(0,3),"gpu"));
    bool const opt_mpi   = (icompare(opt.substr(0,5),"mpi"));
    bool const opt_shm   = (icompare(opt.substr(0,3),"shm")) ;
    char const* msg_init_failed = nullptr;
    cout<<" Enter mkGlobalDstorm: optString="<<optString
        <<" opt_gpu,mpi,shm = "<<opt_gpu<<", "<<opt_mpi<<", "<<opt_shm
        <<endl;

    // err if desired transport unavailable
    if(opt_gpu && !WITH_GPU) msg_init_failed="dstorm without GPU support";
    if(opt_mpi && !WITH_MPI) msg_init_failed="dstorm without MPI support";
    if(!(opt_gpu || opt_mpi || opt_shm)){
        msg_init_failed="invalid dstorm option: perhaps try GPU|MPI|SHM";
    }
#if WITH_GPU
    if(msg_init_failed == nullptr && opt_gpu){
        try {
            cout<<" trying GPU transport ..."<<endl;
            // scoped IPC mutex here XXX
            if( globalDstorm == nullptr ){
                globalDstorm = new Dstorm( Transport<GPU>(/*nothing*/) );
            }
        }
        catch( exception & e ) {
            cout<<" mkGlobalDstorm exception: "<<e.what()<<endl;
            cout<<"\nNOTE:\n"
                <<"   - You should use mpirun to initiate MPI tests.\n"
                <<"   - E.g. mpirun [OPTION]... `which th` `pwd test.lua`/test.lua\n"
                <<"lua dstorm object is still UNUSABLE\n"
                <<endl;
            msg_init_failed = "Failed dstorm GPU initialization";
        }
    }
#endif
#if WITH_MPI
    if(msg_init_failed == nullptr && opt_mpi){
        try {
            cout<<" trying MPI transport ..."<<endl;
            // scoped IPC mutex here XXX
            if( globalDstorm == nullptr ){
                globalDstorm = new Dstorm( Transport<dStorm::OMPI>(/*nothing*/) );
            }
        }
        catch( exception & e ) {
            cout<<" mkGlobalDstorm exception: "<<e.what()<<endl;
            cout<<"\nNOTE:\n"
                <<"   - You should use mpirun to initiate MPI tests.\n"
                <<"   - E.g. mpirun [OPTION]... `which th` `pwd test.lua`/test.lua\n"
                <<"lua dstorm object is still UNUSABLE\n"
                <<endl;
            msg_init_failed = "Failed dstorm MPI initialization";
            if(globalDstorm!=nullptr){ free(globalDstorm); globalDstorm=nullptr; }
        }
    }
#endif
    if(msg_init_failed == nullptr && opt_shm ) {
        try{
#if WITH_SHM
            cout<<" trying shm transport ..."<<endl<<endl;
            cout<<" Retrying with Transport = SHM (WARNING: implementation is INCOMPLETE) ... "<<endl;
            cout<<" Using default SHM configuration for now (should have nThreads parameter)"<<endl;
            uint32_t const nThreads = 1U; // threads not yet supported
            globalDstorm = new Dstorm( Transport<SHM>(nThreads) ); // SHARED MEMORY transport
            // debug:
            cout<<" Dstorm constructed, try some debug calls..."<<endl;
            orm_rank_t ii, nn;
            struct Orm *orm = globalDstorm->orm;
            orm->proc_rank( orm, &ii );
            orm->proc_num ( orm, &nn );
            cout<<" proc_rank = "<<ii<<"  proc_num = "<<nn<<endl;
            cout<<" Dstorm constructoed,                     ... mkGlobalDstorm DONE"<<endl;
#else
            throw std::runtime_error("no WITH_LIBORM: Transport<SHM> not possible");
#endif
        }
        catch( exception & e ) {
            cout<<" mkGlobalDstorm exception: "<<e.what()<<endl;
            throw "mkGlobalDstorm over shared memory FAILED";
            if(globalDstorm!=nullptr){ free(globalDstorm); globalDstorm=nullptr; }
        }
    }
    if( msg_init_failed != nullptr || globalDstorm == nullptr ){
        ostringstream oss;
        oss<<" mkGlobalDstorm failure, optString="<<optString;
        if( msg_init_failed != nullptr ) oss<<" Error: "<<msg_init_failed;
        runtime_error e(oss.str());
        cout<<e.what()<<std::endl;
        throw e;
    }
    cout<<" mkGlobalDstorm for "<<(opt_gpu?"GPU":opt_mpi?"MPI":opt_shm?"SHM":"NOTRANSPORT")
        <<" transport Succeeded"<<endl;
    return globalDstorm;
}

/** automate seginfo getter functions (note that macro
 * \em scr_ERR macro <B> returns ZERO for any error </B>. */
#define SEGINFO_GETTER( RET, FNAME, FIELD ) \
    int scr_Dstorm::f_get##FNAME () { \
        scr_TRY("dstorm.get" #FNAME "( <segnum:int> ) --> " #RET) \
        { \
            scr_INT( segnum, ERR_LBL ); \
            scr_STK("dstorm.get" #FNAME "( <segnum:int> ) --> " #RET ); \
            dStorm::SegInfo const& sInfo = GD->getSegInfo( segnum );  \
            d_si->put_##RET( static_cast<RET>( sInfo.FIELD ) ); \
            return 1; \
        }scr_CATCH; \
    }
SEGINFO_GETTER( Uint4, IoNet, ionet );          // put_Uint2/4/8 don't exist
SEGINFO_GETTER( Sint8, Policy, policy );
SEGINFO_GETTER( Uint4, SegNum, segNum );
SEGINFO_GETTER( Uint4, Obuf, obuf );
SEGINFO_GETTER( Uint4, Ibuf, ibuf );
SEGINFO_GETTER( Uint4, Rbuf, rbuf );
SEGINFO_GETTER( Uint4, Nbuf, nbuf );
SEGINFO_GETTER( Uint4, BufBytes, bufBytes );
SEGINFO_GETTER( Uint4, SegBytes, segBytes );

//SEGINFO_GETTER( ccptr, Mem, mem );              // really a void* lightuserdata, or intptr_t
// above is now problematic with ccptr and gpu transport
// OH -- ccptr is handled with lua_push[l]string, and for GPU, we can never access the memory pointer.
//       Does lua_pushstring scan for the terminal '\0' ?
// Let's return a "raw" intptr_t as a Uint8_t
int scr_Dstorm::f_getMem () {
    scr_TRY("dstorm.get" "Mem" "( <segnum:int> ) --> " "lightuserdata")
    {
        scr_INT( segnum, ERR_LBL );
        scr_STK("dstorm.get" "Mem" "( <segnum:int> ) --> " "lightuserdata" );
        dStorm::SegInfo const& sInfo = GD->getSegInfo( segnum );
        d_si->put_lightuserdata( sInfo.mem );
        // lightuserdata itself can DO nothing, but tostring(lightuserdata) works (and prints the ptr value)
        return 1;
    }scr_CATCH;
}

SEGINFO_GETTER( Uint8, Datacode, datacode );    // datacode and fmtValue might change for ease-of-use (no put_Uint8 func)
SEGINFO_GETTER( Uint4, Datasize, datasize );
SEGINFO_GETTER( Uint4, Cnt, cnt );
SEGINFO_GETTER( Uint4, SizeofMsgHeader, sizeofMsgHeader );
SEGINFO_GETTER( Uint4, Seg_id, seg_id );        // really Uint2
SEGINFO_GETTER( Uint4, FmtValue, fmtValue );    // really Uint1
SEGINFO_GETTER( bool, Valid, valid );
#undef SEGINFO_GETTER

int scr_Dstorm::f_init()
{
    scr_CNT;
    char const* optString = nullptr;    // use default setting
    { scr_STR( s, GOT_OPT );
        scr_STK("dstorm.init() -- default transport");
        optString = s;
        goto MK_GLOBAL_DSTORM;
    }
GOT_OPT:
    scr_STK("<dstorm>:init( <options:string> ) -- options = [ shm | none ]");
MK_GLOBAL_DSTORM:
    cout<<" mkGlobalDstorm (C++ object)..."<<std::endl;
    auto obj = mkGlobalDstorm( optString );
    cout<<" mkGlobalDstorm (C++ object)... "<<(obj==nullptr? "FAILED":"DONE")<<endl;
    // milde error return is zero for error, nonzero for OK
    printf ("Created dstorm %p", globalDstorm);
    return (globalDstorm != nullptr);
}

int scr_Dstorm::f_iProc() {
    scr_CNT;
    if(1){
        scr_STK( "dstorm.iProc() -> <this_node:int>" );
        d_si->put_int( GD->get_iProc() );
        return 1;
    }
    //else goto ERR_LBL; // remove unused-label warning.
    //scr_ERR( "dstorm.iProc() -> <this_node:int>" );
    return 0;
}

int scr_Dstorm::f_nProc() {
    // d_si->put_int( this->d.get_nProc() );
    // return 1;
    //    Above would work, but would skip check for all-args-used
    //cout<<" f_nProc... "; cout.flush();
    scr_CNT;
    if(1){
        scr_STK( "dstorm.nProc() -> <#nodes:int>" );
        d_si->put_int( GD->get_nProc() );
        return 1;
    }
    //else goto ERR_LBL; // remove unused-label warning.
    //scr_ERR( "dstorm.nProc() -> <#nodes:int>" );
    return 0;
}

/** actually, even better might be as:
 *    add_segment(segnum, ionet, n, "r4") and RETURN a PAIR of external-memory
 *    dPoints:  ( outgoing "push" buffer, incoming "reduce" buffer )
 *
 * OR (TBD) add_segment( segnum, ionet, <tensor:scr_Tensor(max dimension, max size)> )
 *
 * Segment type (ex. SegVecDense<T>) and arbitrary lua tensor/matrix/vector<T'> types
 * T and T' COULD be supported. Typically expect T == T'.
 *
 * Q: Seginfo.datatcode is rtti (kludgy) is there an enum list for lua supported types?
 */
int scr_Dstorm::f_add_segment() {
    using dStorm::IoNet_t;              // now just some integer type, TBD: dynamically add graph types from lua
    using dStorm::SegPolicy;
    using dStorm::SegPolicyENUM;
    using dStorm::SegPolicyIO;
    using dStorm::name;
    // changed: now have user-defined IoNet_t, not just OldIoNetEnum tags for building ionets
    using std::string;
    // does scr_XXX give use try catch ? No.
    scr_CNT;
    try {
        scr_INT( segnum, ERR_LBL );
        DBG(" segnum="<<segnum);        // segnum does not exist. just see that it's a small thing
        if( segnum > (1<<16)-1 ){ cout<<"out-of-range segment"<<endl; goto ERR_LBL; };
        { scr_INT( ionet, ERR_LBL );
            DBG(" ionet="<<ionet);
            if( ionet < 0 || ionet >= 256 ) goto ERR_LBL;
            { scr_INT( segpolicy, ERR_LBL );
                DBG(" segpolicy="<<segpolicy);
                if( segpolicy < 0 || segpolicy >= dStorm::SEGPOLICY_MAX ) goto ERR_LBL;
                DBG(" segpolicy="<<segpolicy<<" OK!");
                { scr_INT( nPoints, ERR_LBL );
                    DBG(" nPoints="<<nPoints);
                    if( nPoints < 0 || nPoints > 1000000000 ) goto ERR_LBL;
                    { scr_STR( type, ERR_LBL );
                        // The segment type is RUNTIME selected:
                        //   GD->transport : FMT = seg::VecDense vs seg::VecGpu
                        //   "r4" | "r8"   : float/double
#if WITH_GPU
                        if( GD->transport == GPU ){
                            return gpu_add_segment_helper(segnum,ionet,segpolicy,nPoints,type);
                        }
#endif
                        string t(type);
#define SegVec_t dStorm::seg::VecDense
                        if( t == "r4" ){
                            cout<<" seg "<<segnum<<" --> "<<SegVec_t<float>::name<<"<float>"<<endl;
                            GD->add_segment<SegVec_t<float> >( segnum, (IoNet_t)(ionet), (dStorm::SegPolicy)segpolicy, nPoints );
                            // TODO lua side should have some segnum-->SegInfoPOD table
                            // this could also be done be dynamic cast mechanism, I suppose.
                            // (f_reduce uses the dynamic cast method, but maybe that's not so great)
                        }else if( t == "r8" ){
                            if( GD->transport != GPU ){
                                cout<<" seg "<<segnum<<" --> "<<SegVec_t<double>::name<<"<double>"<<endl;
                                GD->add_segment<SegVec_t<double> >( segnum, (IoNet_t)(ionet), (dStorm::SegPolicy)segpolicy, nPoints );
                            }
#undef SegVec_t
                        }else{
                            cout<<" BAD TYPE: "<<type<<" -- use r4 or r8"<<endl;cout.flush(); sleep(1);
                            Derr_msg( false, true, "type: "<<t<<" not supported --- use r4 or r8");
                            goto ERR_LBL; // return 0; // ERROR
                        }
                    }
                }
            }
        }
    }catch(std::exception& e){
        cout<<e.what();
        DBG(e.what());
        Derr_msg( false, true, e.what() );
        goto ERR_LBL;
    }
    // perhaps store segnum --> ?? for easy lookups, NOT using the C++ FMT template parm XXX
    // ( add C++ restriction that segnums be globally unique across all destorm segments
    //          (of any FMT) )
    // create 2 external memory dPoint's and return them XXX TBD
    //return 1;    // what is returned ??? should this be zero ???
    return 0;
    scr_ERR( "ERR: dstorm.add_segment( <segnum:int>, <ionet:int>, <segLayout:int>, <nPoints:int>, <point_type:string> ) -- point_type real4 or real8" );
}

int scr_Dstorm::f_delete_segment() {
    scr_CNT;
    try {
        scr_INT( segnum, ERR_LBL );
        GD->delete_segment(segnum);
        return 0;       // void return, no error code
    }catch(std::exception& e){
        cout<<e.what();
        Derr_msg( false, true, e.what() );
        goto ERR_LBL;
    }
    return 0;
    scr_ERR( " dstorm.delete_segment( <segnum:int> )" );
}

/** await acks, for NOTIFY_ACK \c SegPolicy only.
* \return outstanding acks, zero means \e all previous \c Dstorm::push have completed.
* - \e possible Weak semantics: data has been received
* - \e \b actual Strong semantics: data has been received \e and fully processed by
*   the destination rank's \c reduce operation.
* - This can be used to guarantee \e zero mixed-version vectors for reduce ops.
*
* - for in-place lua vectors pointing directly at EXTERN segment obuf memory,
*   nice clients will \c dstorm.wait(s,timeout_ms) before \e modifying the obuf
*   data.
* - Ex: if fprop and bprop need read-only access, and adjust modifies obuf, ideally you would:
*   - dstorm.push... net:fprop... net:bprop... dstorm.wait... net:adjust
*/
#if WITH_NOTIFYACK
int scr_Dstorm::f_wait(){
   scr_TRY(" dstorm.wait( <s:SegNum> [, <timeout_ms:int[60000]>] ) -> <remainingAcks:int) -- ret zero means pending sends got acked") {
       //DBG(" Next stack item has type "<<MILDE::GAS.d_si->param_type(__stk_cnt));
       { scr_INT( segnum, ERR_LBL );
           //DBG(" Got segnum "<<segnum<<".  Next stack item has type "<<MILDE::GAS.d_si->param_type(__stk_cnt));
           unsigned timeout_ms = 60000; // default timeout is one minute
           {
               scr_INT( timeout_arg, GOT_timeout_ms );
               //DBG(" Got timeout_ms "<<segnum<<".  Next stack item has type "<<MILDE::GAS.d_si->param_type(__stk_cnt));
               timeout_ms = timeout_arg;
           }
GOT_timeout_ms:
           uint32_t remainingAcks = 0U;
           try{
               remainingAcks = GD->wait( segnum, static_cast<orm_timeout_t>( timeout_ms ));
           }catch( std::exception const& e ){
               Derr_msg( false, true, " exception in wait( segnum="<<segnum<<", timeout_ms="<<timeout_ms<<") : "<<e.what());
           }
           d_si->put_int( remainingAcks );
           return 1;
       }
   }scr_CATCH;
}
#else
int scr_Dstorm::f_wait(){
    ; // no-op, only exists in libdstorm if notify-ack protocol is supported
}
#endif



int scr_Dstorm::f_add_ionet() {
    using dStorm::IoNet_t;              // now just some integer type, TBD: dynamically add graph types from lua
    using dStorm::name;
    using std::string;
    scr_TRY( " dstorm.add_ionet( <net:ionet> ) -> <IoNet_t:int> -- grab impl of ionet, replace ionet with IoNet_t handle" ){
        scr_USR( scr_ionet, net, ERR_LBL );
        assert( GD != nullptr );
        if( net->owner != nullptr ){
            if( net->owner != GD ){
                cout<<" add_ionet ERROR: ionet belongs to another Dstorm object already "<<endl;
                Derr_msg( false, true, " add_ionet ERROR: ionet belongs to another Dstorm object already ");
                goto ERR_LBL;
            }
            cout<<" Duplicate dstorm.add_ionet(<ionet>) ignored -- returning existing handle"<<endl;
            assert( net->ionetHandle != (IoNet_t)(-1) );
        }else{ // Dstorm will acquire a unique_ptr to the impl, etc.
            assert( net->ionetImpl != nullptr );
            net->ionetHandle = GD->add_ionet( unique_ptr<mm2::UserIoNet>( net->ionetImpl ));
            net->owner = GD;
            net->ionetImpl = nullptr;
        }
        d_si->put_int( net->ionetHandle );
        return 1;
    }scr_CATCH;
}

template<>
int scr_Dstorm::segbuf_torch_tensor1D<float>( float * const data
                                              /*, uint_least32_t const offset*/
                                              , uint_least32_t const cnt )
{
   try {
       THFloatStorage * theStorage = THFloatStorage_newWithData(data, cnt);
       if( theStorage == nullptr ) throw runtime_error("THFloatStorage_newWithData failed");
       THFloatTensor * theTensor =  THFloatTensor_newWithStorage1d(theStorage, /*storageOffset*/0, cnt, /*stride*/1);
       //??? THFloatTensor * theTensor =
       //        THFloatTensor_setStorage( emptyTensor, theStorage, /*storageOffset*/0,
       //                                  /*size*/NULL, /*stride*/NULL ):
       luaT_pushudata(d_si->L, (void*)theTensor, "torch.FloatTensor"); \
       return 1;
   }catch(std::exception& e){
       cout<<"segbuf_torch_tensor1D("<<(void*)data<<","<<cnt<<"): "<<e.what();
       Derr_msg( false, true, e.what() );
       return 0U;
   }
}
template<>
int scr_Dstorm::segbuf_torch_tensor1D<double>( double * const data
                                              /*, uint_least32_t const offset*/
                                              , uint_least32_t const cnt )
{
   try {
       THDoubleStorage * theStorage = THDoubleStorage_newWithData(data, cnt);
       if( theStorage == nullptr ) throw runtime_error("THFloatStorage_newWithData failed");
       THDoubleTensor * theTensor =  THDoubleTensor_newWithStorage1d(theStorage, /*storageOffset*/0, cnt, /*stride*/1);
       //??? THFloatTensor * theTensor =
       //        THFloatTensor_setStorage( emptyTensor, theStorage, /*storageOffset*/0,
       //                                  /*size*/NULL, /*stride*/NULL ):
       luaT_pushudata(d_si->L, (void*)theTensor, "torch.DoubleTensor");
       return 1;
   }catch(std::exception& e){
       cout<<"segbuf_torch_tensor1D("<<(void*)data<<","<<cnt<<"): "<<e.what();
       Derr_msg( false, true, e.what() );
       return 0U;
   }
}
// force instantiations of some templates into library
template int scr_Dstorm::segbuf_torch_tensor1D<float>( float * const data, uint_least32_t const cnt );
template int scr_Dstorm::segbuf_torch_tensor1D<double>( double * const data, uint_least32_t const cnt );

#if 0
// force instantiations of some templates into library
//template int scr_Dstorm::segbuf_as_flat_tensor<float>( uint32_t const seg, uint32_t const buf );
//template int scr_Dstorm::segbuf_as_flat_tensor<double>( uint32_t const seg, uint32_t const buf );

template<typename T> inline
int scr_Dstorm::segbuf_as_vec( uint32_t const seg, uint32_t const buf )
{
   try {
       // Note: seg invalid will throw
       SegInfo const& sInfo = GD->getSegInfo( seg );        // throw if illegal
       if( sizeof(T) != sInfo.datasize ){
           Derr_msg( false, true, "cannot convert to flat tensor -- data type size mismatch");
           return 0U;
       }
       if(0){ // could restrict to T exactly matching the segment data type
           if( sInfo.datacode != typeid(T).hash_code() ) {
               Derr_msg( false, true, "cannot convert to flat tensor -- data type does not match segment data type");
               return 0U;
           }
       }
       // Note: buf invalid will throw
       T const* const bufBeg = const_cast<T const*>(static_cast<T*>(sInfo.ptrData( buf )));
       T const* const bufEnd = bufBeg + sInfo.cnt;
       dPoint<T> buf( bufBeg, bufEnd, Cat_ptr::EXTERN );    // has nice EXTERN constructor
       typename Array1D<T>::rcptr const& bufrc(buf);
       Array1D<T> x( bufrc );
       scr_Vec_T<T>::new_stack( x );
       return 1;
   }catch(std::exception& e){
       cout<<"segbuf_as_flat_tensor("<<seg<<","<<buf<<"): "<<e.what();
       Derr_msg( false, true, e.what() );
       return 0U;
   }
}


/** export the segment oBuf as a flat milde tensor.
*  So far only checking for float and double segment data types.
*/
int scr_Dstorm::f_obuf_flat_tensor() {
   scr_CNT;
   try {
       // FIXME need to check that the segment is really a "dense something"
       scr_INT( segnum, ERR_LBL );
       SegInfo const& sInfo = GD->getSegInfo( segnum );        // throw if illegal
       // how to get at the "type" of the SegImpl?
       // for now let's do it without dynamic_cast, by looking at the type info
       if( sInfo.datacode == typeid(float).hash_code()
           && segbuf_as_flat_tensor<float>(segnum,sInfo.obuf) == 1 )
           return 1U;
       if( sInfo.datacode == typeid(double).hash_code()
           && segbuf_as_flat_tensor<double>(segnum,sInfo.obuf) == 1 )
           return 1U;
       Derr_msg( false, true, "obuf_flat_tensor: Please support some new segment data type");
       goto ERR_LBL;
   }catch(std::exception& e){
       cout<<e.what();
       Derr_msg( false, true, e.what() );
       goto ERR_LBL;
   }
   scr_ERR( " dstorm.obuf_flat_tensor(seg_idx:u4>) -> <obuf:tensor>" );
}


int scr_Dstorm::f_obuf_vec() {
       scr_CNT;
       try {
           // FIXME need to check that the segment is really a "dense something"
           scr_INT( segnum, ERR_LBL );
           SegInfo const& sInfo = GD->getSegInfo( segnum );        // throw if illegal
           // how to get at the "type" of the SegImpl?
           // for now let's do it without dynamic_cast, by looking at the type info
           if( sInfo.datacode == typeid(float).hash_code()
               && segbuf_as_vec<float>(segnum,sInfo.obuf) == 1 )
               return 1U;
           if( sInfo.datacode == typeid(double).hash_code()
               && segbuf_as_vec<double>(segnum,sInfo.obuf) == 1 )
               return 1U;
           Derr_msg( false, true, "obuf_vec: Please support some new segment data type");
           goto ERR_LBL;
       }catch(std::exception& e){
           cout<<e.what();
           Derr_msg( false, true, e.what() );
           goto ERR_LBL;
       }
       scr_ERR( " dstorm.obuf_vec(seg_idx:u4>) -> <obuf:vec>" );
}
#endif

#if 0 // old code for f_store
#if 0 == -1
#define COPY_CONVERT_BEG_END(Tseg, Targ) do { \
    THFloatStorage* dpT = dynamic_cast<THFloatStorage * > ( dpoint); \
    if (dpT)    {\
        printf (" snarfed a scr_dPoint_T<%s>\n", #Targ) ; \
        printf (" *processing dPoint in store"); \
        GD->store<Tseg>(segnum, result, result+rsize, /*offset=*/0U, wgt ); \
        return 1; \
    } \
}while(0)
#elif 1 == -1
#define COPY_CONVERT_BEG_END(Tseg,Targ) do { \
    scr_dPoint_T<Targ> * dpT = dynamic_cast<scr_dPoint_T<Targ> *>( dpoint ); \
    if( dpT ) { \
        /*cout<<" snarfed a scr_dPoint_T<" << #Targ <<">"<<endl;*/ \
        GD->store<Tseg>(segnum, dpT->d_arr->begin(), dpT->d_arr->end(), /*offset=*/0U, wgt ); \
        return 1; \
    } \
}while(0)
                // d_arr has dim() == size() function for "cnt", so we could also use the next macro:
                // (beg,end could sometimes be slower because it does a std::distance to shunt to the (beg,cnt) version)
#define COPY_CONVERT_BEG_CNT(Tseg,Targ) do { \
    scr_dPoint_T<Targ> * dpT = dynamic_cast<scr_dPoint_T<Targ> *>( dpoint ); \
    if( dpT ) { \
        /*cout<<" snarfed a scr_dPoint_T<" << #Targ <<">"<<endl;*/ \
        GD->store<Tseg>(segnum, dpT->d_arr->begin(), dpT->d_arr->dim(), /*offset=*/0U, wgt ); \
        return 1; \
    } \
}while(0)
#endif
#if 0 // dynamic_cast approach ONLY will work for milde (we have some common base class)
//DBG(" Got scr_dPoint "<<dpoint); //<<".  Next stack item has type "<<d_si->param_type(__stk_cnt));
// \b FIXME I \b assume Tseg is VecDense<float>,
// but arbitrary segment storage format is in fact possible XXX.
// (i.e. need runtime "buffer representation" type for the buffers in a segment.

// if iter converts to cnt, this version calls the WRONG function (has happened!)//
//

//THFloatTensor* dpT = dynamic_cast<THFloatTensor * > ( dpoint); \
//THFloatStorage* dpT = dynamic_cast<THFloatStorage * > ( dpoint); \
// GD->store<Tseg>(segnum, THFloatTensor_data(dpT), THFloatTensor_data(dpT)+*dpT->size, /*offset=*/0U, wgt ); 
//scr_dPoint_T<real4> * dpr4 = dynamic_cast<scr_dPoint_T<real4> *>( dpoint );
//if( dpr4 ){
//    cout<<" snarfed a scr_dPoint_T<real4>"<<endl;
//    GD->store<VecDense<float>>(segnum, dpr4->d_arr->begin(), dpr4->d_arr->end(), /*offset=*/0U );
//    // XXX want lua to store the "VecDense<float>" type during add_segment
//    //GD->store(segnum, dpr4->d_arr->begin(), dpr4->d_arr->end(), /*offset=*/0U );
//    return 0;
//}
// Q: do we want to warn for inefficient (non-memcpy) conversions?
//
// XXX BUG? will converting iterators silently "optimize away" the memcpy
//          if the iterator pointer points at the obuf data already?
//     This should be disallowed (throwing some error)
//
COPY_CONVERT_BEG_END( VecDense<float>, real4 );
COPY_CONVERT_BEG_END( VecDense<float>, real8 );
COPY_CONVERT_BEG_END( VecDense<float>, Sint4 );
COPY_CONVERT_BEG_END( VecDense<float>, Sint8 );
COPY_CONVERT_BEG_END( VecDense<float>, Uint4 );
COPY_CONVERT_BEG_END( VecDense<float>, Uint8 );
COPY_CONVERT_BEG_END( VecDense<float>, Uint2 );
COPY_CONVERT_BEG_END( VecDense<float>, Sint2 );
COPY_CONVERT_BEG_END( VecDense<float>, Uint1 );
COPY_CONVERT_BEG_END( VecDense<float>, Sint1 );

#undef COPY_CONVERT_FORM_BEG_END_ITER
Derr_msg( false, true, " no support for this scr_dPoint_T type " );
goto ERR_LBL;
NO_DPOINT:
// do we need support for other types? (mat, arrayND, tensor,...)
DBG(" failed to find tensor argument");
goto ERR_LBL;
#endif
#endif

int scr_Dstorm::f_store() {
    int const verbose=1;
    scr_CNT;
    try{
        DBG(" Next stack item has type "<<d_si->param_type(__stk_cnt));
        {
            scr_INT( segnum, ERR_LBL );
            //int segnum = luaL_checkint (d_si->L, 1);

            DBG(" Got segnum "<<segnum<<".  Next stack item has type "<<d_si->param_type(__stk_cnt));
            SegInfo const& sInfo =
            GD->getSegInfo( segnum );  // throw if bad segnum
            // segment data type (ex. VecDense<T>) determinable from SegInfo::fmtValue == seg::VecDense<float>::value
            {
                //scr_USR(THFloatTensor, dpoint, NO_THFloatTensor);
                //  ... but it might be various flavors of torch.xxxTensor
                scr_UDATA( udata, NO_UDATA);
                goto HAVE_UDATA;
NO_UDATA:
                throw std::runtime_error(" No userdata [torch tensor data] argument for lua f_store");
HAVE_UDATA:
                int stk_udata = __stk_cnt-1;
                DBG(" stk_udata = "<<stk_udata);

                double wgt;     // NOTE: changed order of arg parse from milde !!!
                {
                    scr_REAL( optwgt, NO_WGT ); // check for bugs here
                    //real8 optwgt = luaL_checknumber (d_si->L, 3);
                    if (optwgt == 0) goto NO_WGT;
                    wgt=optwgt;
                    DBG(" Got wgt "<<wgt<<".  Next stack item has type "<<d_si->param_type(__stk_cnt));
                    goto HAVE_WGT;
NO_WGT:
                    wgt=1.0;
                    DBG(" using default wgt = "<<wgt);
HAVE_WGT:
                    ;
                }

                //
                // Presence and type of lua arguments seems OK so far...
                // Handle different input types:
                //    - milde did this using 'try{ dynamic_cast }'
                //    - torch 'C' code for tensors instead uses macro-generated typenames,
                //      that we'll need to handle case-by-case
                //

                // argument is some torch tensor type.
                // Get the string name of the lua argument
                const char *_type1 = luaT_typename(d_si->L, stk_udata); // ex. "torch.FloatStorage"
                enum UdataType { UD_Unknown, UD_FloatStorage, UD_CudaStorage };
                enum UdataType ud = UD_Unknown;
                //
                // Is lua_checkudata useful to us? It could cut down code. For example
                //   static Foo *checkFoo (lua_State *L, int index)
                //   {
                //     Foo *bar;
                //     luaL_checktype(L, index, LUA_TUSERDATA);                 // <--
                //     bar = (Foo *)luaL_checkudata(L, index, FOO);             // <--
                //     if (bar == NULL) luaL_typerror(L, index, FOO);           // <--
                //     return bar;
                //   }
                //
                if( std::string(_type1) == std::string("torch.FloatStorage") ) ud = UD_FloatStorage;
#if WITH_GPU
                if( std::string(_type1) == std::string("torch.CudaStorage") ) ud = UD_CudaStorage;
                // or maybe torch.CudaDoubleStorage ?
#endif

                using dStorm::seg::VecDense;
                using dStorm::seg::VecGpu;
                switch(ud){
                  case( UD_FloatStorage ):
                      {
                          DBG(" UD_FloatStorage... ");
                          //if( GD->transport == GPU ){
                          //    // GPU transport version of dstorm.store does not yet support host data--> GPU XXX
                          //    throw std::runtime_error("can't yet supply torch.FloatTensor data to GPU segment (TBD)");
                          //}
                          THFloatStorage *dpoint =  (THFloatStorage *) luaT_toudata (d_si->L, stk_udata, _type1);
                          //
                          // Need iterator to data beginning
                          // and either end or data count
                          float *fltBeg = (float *) dpoint->data;
                          unsigned long fltCnt = (unsigned long) dpoint->size;

                          if(verbose){
                              std::ostringstream oss;
                              oss<<"f_store: segnum="<<segnum<<" wgt="<<wgt<<" luaT_typename "<<_type1
                                  <<" luaT_toudata dpoint @ "<<(void*)dpoint<<" udata @ "<</*(void*)*/udata
                                  <<" fltBeg = dpoint->data @ "<<(void*)fltBeg<<"\n";
                              oss<<"f_store: fltBeg[fltCnt="<<fltCnt<<"] = {";
                              if( fltCnt > 4 ){
                                  for(size_t i=0U; i<3U; ++i) oss<<" "<<fltBeg[i]; oss<<" ... "<<fltBeg[fltCnt-1U];
                              }else{
                                  for(size_t i=0U; i<fltCnt; ++i) oss<<" "<<fltBeg[i];
                              }
                              oss<<" }";
                              std::cout<<oss.str()<<std::endl;
                          }
                          // Here is a do-it-yourself way around the (slow) rtti method using dynamic_cast...
                          // We need to switch based on sInfo.fmtValue, which records the magic constexpr seg::FMT::value
                          // maybe char const* sInfo.segVecName ?
                          DBG(" fmtValue = "<<unsigned{sInfo.fmtValue});
                          switch(sInfo.fmtValue){
                            case(VecDense<float>::value):  // identical constexpr seg::VecDense<ANY>::value
                                DBG(" VecDense<T>... ");
                                {
                                    // sInfo.datacode is typeid(typename Base::Tdata).hash_code();
                                    // For now, this corresponds only to 'float' or 'double'.
                                    // So checking sInfo.datasize is easier (but would not support Uint4 vs. float)
                                    if( sInfo.datasize == sizeof(float) ){
                                        // now we know the exact type of the top-level segment
                                        GD->store<VecDense<float>> (segnum, fltBeg, fltBeg+fltCnt, /*offset=*/0U, wgt );
                                    }else if(sInfo.datasize == sizeof(double) ){
                                        if(GD->iProc == 0) cout<<" ** CHECKME ** is Dstorm::store smart enough to type-convert?"<<endl;
                                        GD->store<VecDense<double>>(segnum, fltBeg, fltBeg+fltCnt, /*offset=*/0U, wgt );
                                    }else{
                                        throw(std::runtime_error("scr_Dstorm:f_store: bad sInfo.datasize"));
                                    }
                                }
                                break;
                            case(seg::VecGpu<float>::value):    // identical constexpr seg::VecGpu<ANY>::value
                                DBG(" VecGpu<T>... ");
                                {
                                    // punt to cuda-aware code ...
                                    //typedef seg_VecGpu<float> Seg_Vec;
                                    // WRONG:  gpu_store_helper needs a ptr to GPU memory
                                    throw std::runtime_error("Please do not store a torch FloatTensor into a seg::VecGpu<T>. Use :cuda() to move it to GPU memory");
                                    if( sInfo.datasize == sizeof(float) ){
                                        gpu_store_helper< VecGpu<float>, float const* >(sInfo, fltBeg, fltCnt, /*offset=*/0U, wgt);
                                    }else if(sInfo.datasize == sizeof(double) ){
                                        gpu_store_helper< VecGpu<double>, float const* >(sInfo, fltBeg, fltCnt, /*offset=*/0U, wgt);
                                    }else{
                                        throw(std::runtime_error("scr_Dstorm:f_store: bad sInfo.datasize"));
                                    }
                                }
                                break;
                            default:
                                DBG(" Unknown segment FMT ");
                                {
                                    ostringstream oss;
                                    oss<<"scr_Dstorm:f_store sInfo fmtValue,datacode,datasize = "
                                        <<sInfo.fmtValue<<", "<<sInfo.datacode<<", "
                                        <<sInfo.datasize<<" : unsupported conversion";
                                    DBG(oss.str());
                                    throw std::runtime_error(oss.str());
                                }
                          }
                          return 1;
                      }
                      break;
                  case( UD_CudaStorage ):
                      {
                          DBG(" UD_CudaStorage... ");
                          if( GD->transport != GPU ){
                              // dstorm.store( iter, ...) original code assumes iter points to device data!
                              // but we have incomplete support for torch.cudaTensor(i,j):storage()
                              throw std::runtime_error("can't yet supply torch.CudaTensor data to non-GPU segment");
                          }
#if WITH_GPU
#if 0
                          throw std::runtime_error("lua dstorm:store(seg,tensor) does not yet know how to use a CudaTensor\n");
#else
                          switch(sInfo.fmtValue){
                            case(VecDense<float>::value):  // identical constexpr seg::VecDense<ANY>::value
                                {
                                    DBG(" VecDense<T>... ");
                                    throw std::runtime_error("Please do not store a CudaTensor into a seg::VecDense<T>. Use :float() (or :double()) to copy the data to CPU");
                                }
                                break;
                            case(VecGpu<float>::value):
                                {
                                    DBG(" VecGpu<T>... ");
                                    //
                                    //typedef struct THCudaStorage
                                    //{
                                    //    float *data;
                                    //    long size;
                                    //    int refcount;
                                    //    char flag;
                                    //    THAllocator *allocator;
                                    //    void *allocatorContext;
                                    //    struct THCudaStorage *view;
                                    //} THCudaStorage;
                                    //
                                    THCudaStorage *thc =  (THCudaStorage *) luaT_toudata (d_si->L, stk_udata, _type1);

                                    /*
                                    std::ostringstream oss;
                                    oss<<"\nf_store(segnum="<<segnum<<",cudaTensor) TBD:\n maybe has THCudaStorage ? data@"<<(void*)thc->data
                                        <<" size="<<ptrdiff_t(thc->size)
                                        <<" recount="<<int(thc->refcount)
                                        <<" flag="<<unsigned(thc->flag)
                                        //<<" device="<<thc->device ... oh, no member of this name?
                                        <<std::endl ;
                                    std::cout<<oss.str();
                                    */
                                    //throw std::runtime_error(oss.str());
                                    float *       fltBeg = (float *)       thc->data;
                                    unsigned long fltCnt = (unsigned long) thc->size;

                                    if( sInfo.datasize == sizeof(float) ){
                                        // GPU float* fltBeg --> GPU float* segment oBuf
                                        gpu_store_helper< VecGpu<float>, float const* >(sInfo, fltBeg, fltCnt, /*offset=*/0U, wgt);
                                    }else if(sInfo.datasize == sizeof(double) ){
                                        // GPU float* fltBeg --> GPU double* segment oBuf
                                        gpu_store_helper< VecGpu<double>, float const* >(sInfo, fltBeg, fltCnt, /*offset=*/0U, wgt);
                                    }else{
                                        throw(std::runtime_error("scr_Dstorm:f_store: bad sInfo.datasize"));
                                    }
                                }
                          }
#endif
#endif // WITH_GPU
                      }
                      break;
                  case( UD_Unknown):
                      {
                          std::ostringstream oss;
                          oss<<"f_store: ERROR: stack item 2 luaT_typename "<<_type1
                              <<" unsupported type (Ex. 'torch.FloatStorage' is supported)";
                          std::cout<<oss.str()<<std::endl;
                          throw runtime_error(oss.str());
                      }
                }
            }
        }
    }catch(std::exception& e){
        cout<<"\nf_store exception: "<<e.what()<<endl;
        Derr_msg( true, true, e.what() );
        goto ERR_LBL;
    }

    return 1;
    scr_ERR(" dstorm.store( <segnum:int>, <data:dPoint<T>> [,<scale:1.0>] ) \n-- copy dPoint into oBuf of segment.\n");
}
int scr_Dstorm::f_getTransport() {
    scr_TRY(" dstorm.getTransport() --> <string> MPI or GPU") {
        if (GD == NULL) d_si->put_ccptr("Transport<HUH>");
        TransportEnum t = GD->transport;
        d_si->put_ccptr(  t==OMPI?      "mpi"
                         :t==GPU?       "gpu"
                         :    "Transport<HUH>" );
        return 1;
    }scr_CATCH;
}

int scr_Dstorm::f_push() {
#if WITH_NOTIFYACK
    scr_TRY( "dstorm.push( <segnum:int> [, <notify:int> [, <snd:int> ]] ) -> <err_or_nBytes:int> -- notify: Orm.NTF_*, snd=-1 or sendlist entry" )
#else
    scr_TRY( "dstorm.push( <segnum:int> ) -> <err_or_nBytes:int>" )
#endif
    {
        ssize_t err_or_nBytes = -1; // i.e. error during push
        scr_INT( segnum, ERR_LBL );
        //scr_STK( "dstorm.push( <segnum:int> )" );
        /*SegInfo const& sInfo =*/ GD->getSegInfo( segnum );  // THROW if bad segnum
        {
#if WITH_NOTIFYACK
            DBG(" dstorm.push: WITH_NOTIFYACK code is active");
            uint32_t snd_option = -1U; // --default value, out-edge =  ALL out-edges
            scr_INT( notify, DSTORM_PUSH_1_ARG );
            scr_STK( "dstorm.push( <segnum:int>, <NTF_xxx:int> ) -> <err_or_nBytes:int> -- NTF_DONE: end-of-stream NTF_SELECT: send-to-all");
            DBG(" dstorm.push: segnum:"<<segnum<<" notify:"<<notify);
            using namespace dStorm;
            if( notify != NTF_DONE && notify != NTF_SELECT && notify != NTF_RUNNING ){
                Derr_msg(false,true," Unexpected NTF_xxx enum value for Dstorm::push");
                goto ERR_LBL;
            }
            //DBG(" alt dstorm.push(segnum,notify[,snd]): do we have ok snd?");
            { scr_INT( snd, DSTORM_PUSH_3_ARG );
                scr_STK(" dstorm.push(segnum,notify,snd) -> <err_or_nBytes:int>, snd=-1 or sendlist entry");
                if( snd != -1 ){                                    // ONE out-edge
                    snd_option = static_cast<uint32_t>(snd);
                    uint32_t nOutEdges = GD->segSendVec(segnum).size();
                    if( snd_option >= nOutEdges ){
                        Derr_msg(false,true,"dstorm.push( <segnum:int>, <NTF_xxx:int>, <snd:int> ) -- snd = "<<snd<<" snd_option="<<snd_option<<" must be -1 or a valid out-edge number: 0.."<<nOutEdges);
                        goto ERR_LBL;
                    }
                }
            }
DSTORM_PUSH_3_ARG:
            err_or_nBytes = GD->push(segnum, static_cast<NotifyEnum>(notify), snd_option);
            goto CHK_PUSH_ERROR;
DSTORM_PUSH_1_ARG: // with only SegNum as argument
#endif
            // todo: opt args for subgraph, or min/max "send policy" bounds (for bw control)
            DBG(" dstorm.push: no WITH_NOTIFYACK support");
            //{std::ostringstream oss; oss<<" dstorm.push( segnum="<<segnum<<" )\n"; std::cout<<oss.str(); std::cout.flush();}
            err_or_nBytes = GD->push(segnum);
CHK_PUSH_ERROR: ;
        }
        d_si->put_int( err_or_nBytes );
        if( err_or_nBytes < 0 ){
            cout<<" Dstorm::push error "<<err_or_nBytes<<endl;
            Derr_msg( false, true, " Dstorm::push error "<<err_or_nBytes );
            goto ERR_LBL;
        }
        return 1;
    }scr_CATCH;
}

/** return nonzero if barrier failed (timeout?), exit on other exceptions */
int scr_Dstorm::f_barrier() {
    scr_TRY( "dstorm.barrier( [<timeout_ms:int>] ) -> <status:int> -- return nonzero on status" ) {
        orm_timeout_t timeout_ms = 60000;
        static_assert( sizeof(orm_timeout_t) == 8, "expected orm_timeout_t to be u8");
        {
            scr_INT( timeout_arg, DFLT_timeout );
            scr_STK( "dstorm.barrier( <timeout_ms:int> ) -> <status:int> -- return nonzero on status" );
            static_assert( sizeof(timeout_arg) == 4, "expected scr_INT variable to be i4");
            // and therefore lua's Gaspi.BLOCK value has been downsized from the original ORM_BLOCK value
            timeout_ms = timeout_arg;
            DBG(" barrier( timeout_arg = "<<timeout_ms<<" )");
            goto DO_BARRIER;
        }
DFLT_timeout:
        DBG(" barrier( default_ms = "<<timeout_ms<<" )");
DO_BARRIER:
        int status = 0;
        try{
            DBG(" GD @ "<<(void*)GD<<" -> barrier("<<timeout_ms<<")");
            GD->barrier( timeout_ms );
        }catch( Dstorm::ErrNetwork const& e ){
            cout<<"rank "<<GD->get_iProc()<<" Dstorm::ErrNetwork -- "<<e.what()<<endl;
            status = 1;  // return nonzero for barrier-related failure (including timeout)
        } // other exceptions propagate up
        DBG(" back from GD->barrier, status="<<status);
        d_si->put_int( status );
        return 1;
    }scr_CATCH;
}

/** optional debug macro for TRY_REDUCE */
#define RED_DBG(X) /*DEBUG*/ do { \
    std::ostringstream oss; \
    oss<<"RED_DBG: " << X << "\n"; \
    std::cout<<oss.str(); \
    std::cout.flush(); \
}while(0)
//#define RED_DBG(X)
#if defined(NDEBUG)
#define TRY_REDUCE_DEBUG
#else
#define TRY_REDUCE_DEBUG \
    /* debug... */ \
if( (sInfo.policy & dStorm::REDUCE_OP_MASK) == dStorm::REDUCE_AVG_RBUF_OBUF ) \
assert( sInfo.ibuf == sInfo.obuf ); \
assert( mHdr->hdr.u.off == 0U ); \
assert( mHdr->hdr.u.sz == sizeof(Tdata) ); \
/* mHdr->hdr.u.cnt ~ how many floats */ \
//RED_DBG( " r"<<GD->iProc<<" mHdr->hdr.u.cnt = "<<mHdr->hdr.u.cnt<<" reduce output bufnum = "<<redbuf<<endl; ) \
/* end of debug */
#endif
/** returns 2 stack variables on success, otherwise falls through.
 * NEW: although iBuf may technically be absent, if it is absent, it's value
 *      is equal to oBuf, so iBuf remains always the buffer number of the
 *      reduce "output".
 */
 #define TRY_REDUCE( SEGTYP ) do \
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
                 /*THFloatTensor * foo =*/ THFloatTensor_newWithSize1d(0/*empty*/); \
                 return 3; \
             } \
             /* reduce output *always* goes into the iBuf  ~ "input buffer" */ \
             /* SEG_AVG_RBUF_OBUF averages rBufs and oBuf into the oBuf (iBuf == oBuf) */ \
             void * const outseg = seg->ptrBuf( sInfo.ibuf ); \
             auto const mHdr = mem_as< MsgHeader<SegType::Fmt> * >(outseg); \
             TRY_REDUCE_DEBUG; \
             RED_DBG( "cpu reduce mHdr: .a={iter,bytes,fmt,pushes}={" <<mHdr->hdr.a.iter<<","<<mHdr->hdr.a.bytes<<","<<unsigned(mHdr->hdr.a.fmt)<<","<<unsigned(mHdr->hdr.a.pushes) <<"} .u={off,cnt,wgt,sz}={" <<mHdr->hdr.u.off<<","<<mHdr->hdr.u.cnt<<","<<mHdr->hdr.u.wgt<<","<<unsigned(mHdr->hdr.u.sz) <<"}"); \
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
#if 0
             /* MILDE::dPoint<float> reduction( data, dataEnd, Cat_ptr::EXTERN ); */ \
             THFloatStorage* theStorage = THFloatStorage_newWithData(data, mHdr->hdr.u.cnt); \
             THFloatTensor* scr_dpoint =  THFloatTensor_newWithStorage1d(theStorage, 0, mHdr->hdr.u.cnt, 1);  \
              assert( scr_dpoint != nullptr );
#endif

     /** Success pushes nReduce, offset, <some torch tensor type> onto lua stack.
      * - seg::VecGpu returns nReduce + offset (copy into a luaTensor would be slow)
      *   - should it maybe return nothing?
      *   - should there be another lua call to retrieve the r|o|iBuf data as a torch tensor?
      *   - \ref gpu_reduce_helper is defined in \ref 
      * \return 3 on success, else 0. */
     int scr_Dstorm::f_reduce() {
         scr_CNT;
         try {
             scr_INT( segnum, ERR_LBL );
             SegInfo const& sInfo = GD->getSegInfo( segnum ); // throw if bad segnum
             uint32_t nReduce = GD->reduce(segnum);
             DBG( " nReduce = GD->reduce("<<segnum<<") = "<<nReduce);
#if WITH_GPU
             // Note: GPU transport **Really** should be a SEGMENT property, not a global dstorm thing.
             if( GD->transport == GPU ){
                 //d_si->put_Sint4( (Sint4)nReduce ); /*-1U should be returned as a nice lua value*/ \
                 // gpu_reduce_helper **could** be:
                 // - a no-op and push zero more stuff onto lua stack
                 // - or push more stuff
                 //   - like add a lua arg to force slow copy-out of a torch tensor ?
                 //   - or maybe another way to copy retrieve a segment buffer into a torch tensor ?
                 return gpu_reduce_helper( sInfo, nReduce );
             }
             else
#endif
             {
                 // XXX TODO: should we streamline or optimize for "no input available case", ret==false
                 //           for now, just assert that ret == false iff hdr.cnt == 0

                 // TODO
                 //if( ! (try_reduce< dStorm::user::Seg_VecDense<float>>( segnum, sInfo )
                 //       || try_reduce< dStorm::user::Seg_VecDense<double>>( segnum, sInfo )
                 //      )){ return 3 } else { Fatal Error }
                 //
                 TRY_REDUCE( dStorm::user::Seg_VecDense<float> );

                 //TRY_REDUCE( dStorm::user::Seg_VecDense<double> );
                 // XXX :  no convert from double data to torch float tensor in the macro
             }
         }catch(std::exception& e){
             cout<<e.what();
             Derr_msg( false, true, e.what() );
             goto ERR_LBL;
         }
         Derr_msg( false, true, "dstorm.reduce: unsupported segment type\n" );
         scr_ERR( " dstorm.reduce( <segnum:int> ) -> <nReduce:s4>, <reduction:dPoint> -- if segnum fmt is Seg_VecDense<T>" );
     }
#undef TRY_REDUCE
#undef TRY_REDUCE_DEBUG
#undef RED_DBG

