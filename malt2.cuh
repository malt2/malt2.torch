/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
/** \file lua_dstorm.cuh
 *  (C) NECLA
 * set up lua interface to dstorm 
 */

#ifndef MALT_lua_dstorm.cuh
#define MALT_lua_dstorm.cuh

#include "dstorm.cuh"
#include "dstorm_fwd.cuh"
#include "dstorm_msg.cuh"
#include "segVecGpu.cuh"

#include "THC/THC.h"
#include "THC/THCGeneral.h"
#include "THC/THCTensor.h"
#include "luaT.h"
//#include "TH/THGenerateAllTypes.h"
//#include "TH/THGenerateFloatTypes.h"


#include <string>
#include <exception>
#include <iostream>
#include <limits>
#include <vector>

#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"



using namespace std;
using dStorm::Dstorm;
using dStorm::Transport;

#if WITH_GPU
using dStorm::GPU;
#endif
using dStorm::SegInfo;

#include "mtypes.hpp"
#include "lua-i.hpp"

/** For now every process has its own Dstorm object.
 **
 ** The name \em globalDstorm really means "process-local" for now!
 **
 ** Eventually Dstorm configuration and "const-ish" data
 ** should be moved into shared memory.
 **/
Dstorm * globalDstorm = nullptr;
static lua_Interpreter *d_si = NULL;
bool d_si_alloc = false;





Dstorm * mkGlobalDstorm(char const* optString )
{
    string opt (optString? optString: "gpu");
    printf ("optString ==> %s", optString);
    bool opt_gpu = opt.substr(0,3)=="gpu";
    cout<<" Enter mkGlobalDstorm: optString="<<optString
        <<" opt_gpu="<<opt_gpu<<endl;

    if( opt_gpu) {
        try{
#if WITH_GPU
            cout<<" trying gpu transport ..."<<endl<<endl;
            cout<<" Retrying with Transport = GPU (WARNING: implementation is INCOMPLETE) ... "<<endl;
            cout<<" Using default GPU configuration for now (should have nThreads parameter)"<<endl;
            uint32_t const nThreads = 1U; // threads not yet supported
            globalDstorm = new Dstorm( Transport<GPU>() ); // GPU MEMORY transport
            // debug:
            cout<<" Dstorm constructed, try some debug calls..."<<endl;
            orm_rank_t ii, nn;
            struct Orm *orm = globalDstorm->orm;
            orm->proc_rank( orm, &ii );
            orm->proc_num ( orm, &nn );
            cout<<" proc_rank = "<<ii<<"  proc_num = "<<nn<<endl;
            cout<<" Dstorm constructoed,                     ... mkGlobalDstorm DONE"<<endl;
#else
            throw std::runtime_error("no WITH_GPU: Transport<GPU> not possible");
#endif
        }
        catch( exception & e ) {
            cout<<" mkGlobalDstorm exception: "<<e.what()<<endl;
            throw "mkGlobalDstorm over shared memory FAILED";
        }
    }
    else {
        ostringstream oss;
        oss<<" mkGlobalDstorm invalid option string: "<<optString;
        runtime_error e(oss.str());
        cout<<e.what()<<std::endl;
        throw e;
    }
    return globalDstorm;
}


#include "scr_ionet.cuh"
 /** This wraps the Dstorm object and constructs it with any
      * configuration stuff passed in from lua. */
struct scr_Dstorm
{
    //static int f_new();
    //static int f___gc();
    // we are not an object, dStorm is a singleton (so far), so just provide an 'init' function
    // that sets a global ptr to a Dstorm object
    /** After a lua <TT>local X = require 'foo' .init("gpu")</TT> you may call
      * dstorm functions. Option selects gpu or other transport.
      * gpu is the default (because it should always be supported)
      */
    static int f_init();
    /// \name Dstorm methods
    //@{
    // basic utilities
    static int f_iProc();
    static int f_nProc();
    static int f_add_segment();
    static int f_delete_segment();
    static int f_add_ionet();   // acquire ownership of impl, lua object becomes a handle
    // for distributed calculation
    static int f_store();
    static int f_push();
    static int f_reduce();
    // special utilities
    static int f_barrier();
    static int f_setStreamFunc(); // install a lua REDUCE_STREAM callback function
    //nDeadCount
    //nDead
    static int f_wait();
    //netSendVec
    //netRecvVec
    //segSendVec
    //segRecvVec
    // debug
    //print_XXX functions
    //@}

    /// \name Lua exporters
    /**
      * - obuf exporters always export a lua view of the \b dstorm.store buffer.
      * - ibuf exporters give a lua view of the \b dstorm.reduce buffer.
      *
      *  - if the reduce policy does not use a separate iBuf,
      *    like in Seg.REDUCE_AVG_RBUF_OBUF, which has no separate ibuf,
      *    the ibuf view sees into the obuf, which still holds the dstorm.reduce output
      */
     //@{
    static int f_obuf_flat_tensor();    ///< export the oBuf buffer of a Dstorm segment as a flat lua tensor
    static int f_obuf_vec();            ///< export the oBuf buffer of a Dstorm segment as a lua vec
    static int f_obuf_dpoint();         ///< export the oBuf buffer of a Dstorm segment as a lua dpoint
    static int f_ibuf_flat_tensor();    ///< export the iBuf buffer of a Dstorm segment as a flat lua tensor
    static int f_ibuf_vec();            ///< export the iBuf buffer of a Dstorm segment as a lua vec
    static int f_ibuf_dpoint();         ///< export the iBuf buffer of a Dstorm segment as a lua dpoint
    //@}
    /// \name SegInfo getters
    /// all take a segtag parameter (same you use for add_segment) and return 1|0 for ok|error.
    /// \sa Dstorm::SegInfo for the various SegInfo fields
    //@{
    static int f_getIoNet          ();
    static int f_getSegNum         ();
    static int f_getObuf           ();
    static int f_getIbuf           ();
    static int f_getRbuf           ();
    static int f_getNbuf           ();
    static int f_getBufBytes       ();
    static int f_getSegBytes       ();
    static int f_getMem            ();
    static int f_getDatacode       ();
    static int f_getDatasize       ();
    static int f_getCnt            ();
    static int f_getSizeofMsgHeader();
    static int f_getSeg_id         ();
    static int f_getFmtValue       ();
    static int f_getValid          ();
    //@}
         
    /// \name internal helpers
    //@{
    /** pushes a new tensor for the full-sized region of bufnum in segment segnum
     * onto lua stack.
     * Fails if sizeof(T) doesn't match the SegInfo::datasize
     * \return 1 if successful, 0 if failed.
     */
    template<typename T>
        static int segbuf_as_flat_tensor( uint32_t const seg, uint32_t const buf );
    template<typename T>
        static int segbuf_as_vec( uint32_t const seg, uint32_t const buf );
    template<typename T>
        static int segbuf_as_dpoint( uint32_t const seg, uint32_t const buf );
    //@}
    // Dstorm * d;      // for now, cheat by having a singleton (global) Dstorm object
};

#define GD globalDstorm

/** automate seginfo getter functions (note that macro
 * \em scr_ERR macro <B> returns ZERO for any error </B>. */
#define SEGINFO_GETTER( RET, FNAME, FIELD ) \
    int scr_Dstorm::f_get##FNAME () { \
        scr_CNT; \
        try { \
            scr_INT( segnum, ERR_LBL ); \
            scr_STK("dstorm.get" #FNAME "( <segnum:int> ) --> " #RET ); \
            SegInfo const& sInfo = GD->getSegInfo( segnum );  \
            d_si->put_##RET( static_cast<RET>( sInfo.FIELD ) ); \
            return 1; \
        }catch(std::exception& e){ \
            cout<<e.what(); \
            Derr_msg( false, true, e.what() ); \
            goto ERR_LBL; \
        } \
        scr_ERR( " dstorm.get" #FNAME "( <segnum:int> ) --> " #RET ); \
    }
SEGINFO_GETTER( uint4, IoNet, ionet );          // put_uint2/4/8 don't exist
SEGINFO_GETTER( uint4, SegNum, segNum );
SEGINFO_GETTER( uint4, Obuf, obuf );
SEGINFO_GETTER( uint4, Ibuf, ibuf );
SEGINFO_GETTER( uint4, Rbuf, rbuf );
SEGINFO_GETTER( uint4, Nbuf, nbuf );
SEGINFO_GETTER( uint4, BufBytes, bufBytes );
SEGINFO_GETTER( uint4, SegBytes, segBytes );
SEGINFO_GETTER( ccptr, Mem, mem );              // really a void* lightuserdata, or intptr_t
SEGINFO_GETTER( sint8, Datacode, datacode );    // datacode and fmtValue might change for ease-of-use (no put_uint8 func)
SEGINFO_GETTER( uint4, Datasize, datasize );
SEGINFO_GETTER( uint4, Cnt, cnt );
SEGINFO_GETTER( uint4, SizeofMsgHeader, sizeofMsgHeader );
SEGINFO_GETTER( uint4, Seg_id, seg_id );        // really uint2
SEGINFO_GETTER( uint4, FmtValue, fmtValue );    // really uint1
SEGINFO_GETTER( bool, Valid, valid );
#undef SEGINFO_GETTER

int scr_Dstorm::f_init()
{
    scr_CNT;
    char const* optString = nullptr;    // use default setting "gpu"
    { scr_STR( s, GOT_OPT );
        scr_STK("dstorm.init() -- default transport");
        optString = s;
        goto MK_GLOBAL_DSTORM;
    }
GOT_OPT:
    scr_STK("<dstorm>:init( <options:string> ) -- options = [ gpu | none ]");
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
 * Segment type (ex. SegVecGpu<T>) and arbitrary lua tensor/matrix/vector<T'> types
 * T and T' COULD be supported. Typically expect T == T'.
 *
 * Q: Seginfo.datatcode is rtti (kludgy) is there an enum list for lua supported types?
 */
int scr_Dstorm::f_add_segment() {
    using dStorm::seg::VecGpu;
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
                        string t(type);
                        if( t == "r4" ){
                            GD->add_segment<VecGpu<float> >( segnum, (IoNet_t)(ionet), (dStorm::SegPolicy)segpolicy, nPoints );
                            // XXX LUA side should store mapping segnum --> "r4"(/...) if sucessful
                            // i.e.    this->segFmts[segnum] = std::string(t)
                            // OR we go through a dynamic cast for SegInfo [segnum] to all possible SegImpls (Horrible)
                            // XXX SEG_FULL should be sendable from lua, using table of SegPolicy enum values
                        }else if( t == "r8" ){
                            GD->add_segment<VecGpu<double> >( segnum, (IoNet_t)(ionet), (dStorm::SegPolicy)segpolicy, nPoints );
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
    return 1;
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
int scr_Dstorm::f_wait(){
   using dStorm::seg::VecGpu;
   using namespace std;
   scr_TRY(" dstorm.wait( <s:SegNum>, <timeout_ms:int> ) -> <remainingAcks:int) -- ret zero means all pending sends got acked") {
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

int scr_Dstorm::f_store() {
    using dStorm::seg::VecGpu;
    using namespace std;
    scr_CNT;
    try{
        DBG(" Next stack item has type "<<d_si->param_type(__stk_cnt));
        { //scr_INT( segnum, ERR_LBL );
          int segnum = luaL_checkint (d_si->L, 1);

            DBG(" Got segnum "<<segnum<<".  Next stack item has type "<<d_si->param_type(__stk_cnt));
            /*SegInfo const& sInfo =*/ GD->getSegInfo( segnum );  // throw if bad segnum
            // XXX segment data type (ex. VecGpu<float> should be determinable from SegInfo here XXX
            { //scr_USR( scr_dPoint, dpoint, NO_DPOINT );
              //THFloatTensor *dpoint = (THFloatTensor *) luaL_checkudata (d_si->L, 2, "THFloatTensor") ;
              //THFloatTensor *dpoint = (THFloatTensor *) luaT_toudata(d_si->L, 2, "THFloatTensor") ;
              //THFloatStorage *storage_res =  dpoint->storage;
              //float *result = storage_res->data;
              //THFloatStorage *dpoint =  (THFloatStorage *) luaL_checkudata (d_si->L, 2, "torch.THFloatStorage");
              const char *_type1 = luaT_typename(d_si->L, 2);
              THFloatStorage *dpoint =  (THFloatStorage *) luaT_toudata (d_si->L, 2, _type1);
              float *result = (float *) dpoint->data;
              float *d_result;
              checkCudaErrors(cudaMalloc(d_result, rsize*sizeof(float)));
              checkCudaErrors(cudaMemcpy(d_result, result, rsize*sizeof(float), cudaMemcpyHostToDevice));
              //unsigned long rsize = luaL_checkint (d_si->L, 3);
              unsigned long rsize = (unsigned long) dpoint->size;
              printf ("result %f %f %f ....%f (%lu items) .\n ", result[0], result[1], result[2], result[rsize-1], rsize);
              //scr_USR(THFloatTensor, dpoint, NO_DPOINT);
                double wgt;
                {
                    //scr_REAL( optwgt, NO_WGT ); check for bugs here
                    real8 optwgt = luaL_checknumber (d_si->L, 1);
                    if (optwgt == 0) goto NO_WGT;
                    wgt=optwgt;
                    goto HAVE_WGT;
NO_WGT:
                    wgt=1.0;
HAVE_WGT:
                    ;
                }
                //cout<<" snarfed a scr_dPoint of some sort"<<endl;
                //printf ("dpoint obtained is %p %p size: %d.\n\n\n", dpoint, THFloatTensor_data(dpoint), dpoint->size);
                printf ("dpoint obtained is %p %p size: %lu \n\n\n", dpoint, dpoint->data, rsize);
                DBG(" Got scr_dPoint "<<dpoint); //<<".  Next stack item has type "<<d_si->param_type(__stk_cnt));
                // \b FIXME I \b assume Tseg is VecGpu<float>,
                // but arbitrary segment storage format is in fact possible XXX.
                // (i.e. need runtime "buffer representation" type for the buffers in a segment.

                // if iter converts to cnt, this version calls the WRONG function (has happened!)//
                //

                //THFloatTensor* dpT = dynamic_cast<THFloatTensor * > ( dpoint); \
                //THFloatStorage* dpT = dynamic_cast<THFloatStorage * > ( dpoint); \
                         ///*GD->store<Tseg>(segnum, THFloatTensor_data(dpT), THFloatTensor_data(dpT)+*dpT->size, /*offset=*/0U, wgt ); 
#define COPY_CONVERT_BEG_END(Tseg, Targ) do \
            {\
                THFloatStorage* dpT = dynamic_cast<THFloatStorage * > ( dpoint); \
                if (dpT)    {\
                         printf (" snarfed a scr_dPoint_T<%s>\n", #Targ) ; \
                         printf (" *processing dPoint in store"); \
                         //GD->store<Tseg>(segnum, result, result+rsize, /*offset=*/0U, wgt ); \
                         //GD->store<Tseg>(segnum, result, result+rsize, /*offset=*/0U, wgt ); \
                         return 1; \
                     } \
                 }while(0)

#define COPY_CONVERT_BEG_CNT(Tseg,Targ) do \
                { \
                    THFloatStorage * dpT = dynamic_cast<THFloatStorage  *>( dpoint ); \
                    if( dpT ) { \
                        cout<<" snarfed a scr_dPoint_T<" << #Targ <<">"<<endl; \
                        GD->store<Tseg>(segnum, result, rsize, /*offset=*/0U, wgt ); \
                        return 1; \
                    } \
                }while(0)

#if 0
#define COPY_CONVERT_BEG_END(Tseg,Targ) do \
                { \
                    scr_dPoint_T<Targ> * dpT = dynamic_cast<scr_dPoint_T<Targ> *>( dpoint ); \
                    if( dpT ) { \
                        /*cout<<" snarfed a scr_dPoint_T<" << #Targ <<">"<<endl;*/ \
                        GD->store<Tseg>(segnum, dpT->d_arr->begin(), dpT->d_arr->end(), /*offset=*/0U, wgt ); \
                        return 1; \
                    } \
                }while(0)
                // d_arr has dim() == size() function for "cnt", so we could also use the next macro:
                // (beg,end could sometimes be slower because it does a std::distance to shunt to the (beg,cnt) version)
#define COPY_CONVERT_BEG_CNT(Tseg,Targ) do \
                { \
                    scr_dPoint_T<Targ> * dpT = dynamic_cast<scr_dPoint_T<Targ> *>( dpoint ); \
                    if( dpT ) { \
                        /*cout<<" snarfed a scr_dPoint_T<" << #Targ <<">"<<endl;*/ \
                        GD->store<Tseg>(segnum, dpT->d_arr->begin(), dpT->d_arr->dim(), /*offset=*/0U, wgt ); \
                        return 1; \
                    } \
                }while(0)
                //scr_dPoint_T<real4> * dpr4 = dynamic_cast<scr_dPoint_T<real4> *>( dpoint );
                //if( dpr4 ){
                //    cout<<" snarfed a scr_dPoint_T<real4>"<<endl;
                //    GD->store<VecGpu<float>>(segnum, dpr4->d_arr->begin(), dpr4->d_arr->end(), /*offset=*/0U );
                //    // XXX want lua to store the "VecGpu<float>" type during add_segment
                //    //GD->store(segnum, dpr4->d_arr->begin(), dpr4->d_arr->end(), /*offset=*/0U );
                //    return 0;
                //}
                // Q: do we want to warn for inefficient (non-memcpy) conversions?
                //
                // XXX BUG? will converting iterators silently "optimize away" the memcpy
                //          if the iterator pointer points at the obuf data already?
                //     This should be disallowed (throwing some error)
                //
#endif
                COPY_CONVERT_BEG_END( VecGpu<float>, real4 );
                COPY_CONVERT_BEG_END( VecGpu<float>, real8 );
                COPY_CONVERT_BEG_END( VecGpu<float>, sint4 );
                COPY_CONVERT_BEG_END( VecGpu<float>, sint8 );
                COPY_CONVERT_BEG_END( VecGpu<float>, uint4 );
                COPY_CONVERT_BEG_END( VecGpu<float>, uint8 );
                COPY_CONVERT_BEG_END( VecGpu<float>, uint2 );
                COPY_CONVERT_BEG_END( VecGpu<float>, sint2 );
                COPY_CONVERT_BEG_END( VecGpu<float>, uint1 );
                COPY_CONVERT_BEG_END( VecGpu<float>, sint1 );

#undef COPY_CONVERT_FORM_BEG_END_ITER
                Derr_msg( false, true, " no support for this scr_dPoint_T type " );
                goto ERR_LBL;
NO_DPOINT:
                // do we need support for other types? (mat, arrayND, tensor,...)
                DBG(" failed to find tensor argument");
                goto ERR_LBL;
            }
        }
    }catch(std::exception& e){
        cout<<e.what();
        Derr_msg( false, true, e.what() );
        goto ERR_LBL;
    }

    return 1;
    scr_ERR(" dstorm.store( <segnum:int>, <data:dPoint<T>> [,<scale:1.0>] ) \n-- copy dPoint into oBuf of segment.\n");
}

   int scr_Dstorm::f_push() {
         scr_CNT;
         //char const *errmsg = "unknown error";
         //scr_STK( "dstorm.push( <segnum:int> [, <notify:int> [, <snd:int> ]] ) -> <err_or_nBytes:int> -- notify: Orm.NTF_*, snd=-1 or sendlist entry" );
         try {
             ssize_t err_or_nBytes;
             scr_INT( segnum, ERR_LBL );
             /*SegInfo const& sInfo =*/ GD->getSegInfo( segnum );  // THROW if bad segnum
             {
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
                     scr_STK(" dstorm.push(segnum,notify,snd) -> <err_or_nBytes:int>");
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
                 //scr_STK( "dstorm.push( <segnum:int> )" );
                 // todo: opt args for subgraph, or min/max "send policy" bounds (for bw control)
                 err_or_nBytes = GD->push(segnum);
 CHK_PUSH_ERROR: ;
             }
             d_si->put_int( err_or_nBytes );
             if( err_or_nBytes < 0 ){
                 cout<<" Dstorm::push error "<<err_or_nBytes<<endl;
                 Derr_msg( false, true, " Dstorm::push error "<<err_or_nBytes );
                 goto ERR_LBL;
             }
         }catch(std::exception& e){
             cout<<" push exception: "<<e.what();
             Derr_msg( false, true, e.what() );
             goto ERR_LBL;
         }
         return 1;
         //scr_ERR( "dstorm.push( <segnum:int> [, <notify:int> [, <snd:int> ]] ) -- notify: Orm.NTF_*, snd=-1 or sendlist entry" );
 ERR_LBL: sleep(2); Derr_msg(false,true,"dstorm.push failed ( <segnum:int> [, <notify:int> [, <snd:int> ]] ) -- notify: Orm.NTF_*, snd=-1 or sendlist entry" ); return 0;
     }

/** return nonzero if barrier failed (timeout?), exit on other exceptions */
int scr_Dstorm::f_barrier() {
    scr_CNT;
    orm_timeout_t timeout_ms = 60000;
    static_assert( sizeof(orm_timeout_t) == 8, "expected orm_timeout_t to be u8");
    {
        scr_INT( timeout_arg, DFLT_timeout );
        static_assert( sizeof(timeout_arg) == 4, "expected scr_INT variable to be i4");
        // and therefore lua's Orm.BLOCK value has been downsized from the original ORM_BLOCK value
        timeout_ms = timeout_arg;
        //DBG(" barrier( timeout_arg = "<<timeout_ms<<" )");
        scr_STK( "dstorm.barrier( <timeout_ms:int> ) -> <status:int> -- return nonzero on status" );
        goto DO_BARRIER;
    }
DFLT_timeout:
    //DBG(" barrier( default_ms = "<<timeout_ms<<" )");
    scr_STK( "dstorm.barrier( ) -> <status:bool> -- ret nonzero for timeout, default timeout_ms is 60 000" );
DO_BARRIER:
    int status = 0;
    try{
        GD->barrier( (orm_timeout_t)timeout_ms );
    }catch( Dstorm::ErrNetwork const& e ){
        cout<<e.what();
        status = 1;  // return nonzero for barrier-related failure (including timeout)
    } // other exceptions propagate up
    d_si->put_int( status );
    return 1;
}

/** optional debug macro for TRY_REDUCE */
//#define RED_DBG(X) /*DEBUG*/ X
#define RED_DBG(X)
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
RED_DBG( cout<<" r"<<GD->iProc<<" mHdr->hdr.u.cnt = "<<mHdr->hdr.u.cnt<<" reduce output bufnum = "<<redbuf<<endl; ) \
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
             d_si->put_sint4( (sint4)nReduce ); /*-1U should be returned as a nice lua value*/ \
             if( nReduce == 0U || nReduce == -1U ){ \
                 THFloatTensor_newWithSize1d(0/*empty*/); \
                 return 2; \
             } \
             \
             /* auto base = static_cast<SegType::Base const*>(seg); a SegBase<...>  */ \
             /* iBuf same as oBuf if iBuf segment was "removed" */ \
             void * const outseg = seg->ptrBuf( sInfo.ibuf ); \
             auto const mHdr = mem_as< MsgHeader<SegType::Fmt> * >(outseg); \
             TRY_REDUCE_DEBUG; \
             Tdata* const data = mem_as<Tdata*>( outseg, sizeof(MsgHeader<SegType::Fmt>) ); \
             Tdata* const dataEnd = data +  mHdr->hdr.u.cnt; \
             /* MILDE::dPoint<float> reduction( data, dataEnd, Cat_ptr::EXTERN ); */ \
             RED_DBG(THFloatStorage* theStorage = THFloatStorage_newWithData(data, mHdr->hdr.u.cnt); ) ; \
             RED_DBG(THFloatTensor* scr_dpoint =  \
             THFloatTensor_newWithStorage1d(theStorage, 0, mHdr->hdr.u.cnt, 1); );  \
             RED_DBG( assert( scr_dpoint != nullptr ); ) \
             return 2; \
         } \
     }while(0)
     /** \return 1 on success, else 0.  Success return nReduce, scr_dPoint<T> to lua */
     int scr_Dstorm::f_reduce() {
         scr_CNT;
         try {
             scr_INT( segnum, ERR_LBL );
             /*SegInfo const& sInfo =*/ GD->getSegInfo( segnum );  // throw if bad segnum
             uint32_t nReduce = GD->reduce(segnum);
             SegInfo const& sInfo = GD->getSegInfo( segnum );
             // XXX TODO: should we streamline or optimize for "no input available case", ret==false
             //           for now, just assert that ret == false iff hdr.cnt == 0

             TRY_REDUCE( dStorm::user::Seg_VecGpu<float> );
             TRY_REDUCE( dStorm::user::Seg_VecGpu<double> );
             // Unrecognized segment type --- can we even do completely "unknown" user types?
             // A: possible, if make NO assumptions about header content (i.e. all vectors "full" length)
             //              and can upcast the the SegBase<IMPL,TDATA> type to snarf the Tdata
             // fall off into unsupported type error ...
         }catch(std::exception& e){
             cout<<e.what();
             Derr_msg( false, true, e.what() );
             goto ERR_LBL;
         }
         Derr_msg( false, true, "dstorm.reduce: unsupported segment type\n" );
         scr_ERR( " dstorm.reduce( <segnum:int> ) -> <nReduce:s4>, <reduction:dPoint> -- if segnum fmt is Seg_VecGpu<T>" );
     }
#undef TRY_REDUCE
#undef TRY_REDUCE_DEBUG
#undef RED_DBG


#define LUA_FUN(name) { #name , luaB_##name }
#define WRAP_FUN(name, prefix) \
    static int luaB_##name( lua_State* L ) \
{ \
    return prefix##name(); \
}

/** create static int luaB_FOO( lua_State* ) fns to invoke scr_Dstorm::NAME() */ 
#define FUN(name) WRAP_FUN(name, scr_Dstorm::f_)
    //FUN(__gc)
    FUN(init)
    FUN(iProc)
    FUN(nProc)
    FUN(add_segment)
    FUN(delete_segment)
    FUN(add_ionet) 
    FUN(store)
    FUN(push)
    FUN(reduce)
    FUN(barrier)
    FUN(wait)
    //FUN(obuf_flat_tensor)
    //FUN(obuf_vec)
#if 0
    // special utils
    //nDeadCount
    //nDead
    
    FUN(setStreamFunc)
    //netSendVec
    //netRecvVec
    //segSendVec
    //segRecvVec
    // debug
    //print_XXX functions
    
    
    FUN(obuf_dpoint)
    FUN(ibuf_flat_tensor)
    FUN(ibuf_vec)
    FUN(ibuf_dpoint)
    // SegInfo getters scr_Dstorm::f_getXXX( SegNum const s ) --> ...
    FUN(getIoNet          )
    FUN(getSegNum         )
    FUN(getObuf           )
    FUN(getIbuf           )
    FUN(getRbuf           )
    FUN(getNbuf           )
    FUN(getBufBytes       )
    FUN(getSegBytes       )
    FUN(getMem            )
    FUN(getDatacode       )
    FUN(getDatasize       )
    FUN(getCnt            )
    FUN(getSizeofMsgHeader)
    FUN(getSeg_id         )
    FUN(getFmtValue       )
    FUN(getValid          )
#endif
#undef FUN


static const struct luaL_Reg dstorm_lib_f [] = {
    LUA_FUN(init),
    LUA_FUN(iProc),
    LUA_FUN(nProc),
    LUA_FUN(add_segment),
    LUA_FUN(delete_segment),
    LUA_FUN(add_ionet),
    LUA_FUN(store),
    LUA_FUN(push),
    LUA_FUN(reduce),
    LUA_FUN(barrier),
    LUA_FUN(wait),
    //LUA_FUN(obuf_flat_tensor),
    //LUA_FUN(obuf_vec),
    {0,0}
};
/*
    LUA_FUN(setStreamFunc),
    //
    
    
    LUA_FUN(obuf_dpoint),
    LUA_FUN(ibuf_flat_tensor),
    LUA_FUN(ibuf_vec),
    LUA_FUN(ibuf_dpoint),
    // SegInfo getters scr_Dstorm::f_getXXX( SegNum const s ) --> ...
    LUA_FUN(getIoNet          ),
    LUA_FUN(getSegNum         ),
    LUA_FUN(getObuf           ),
    LUA_FUN(getIbuf           ),
    LUA_FUN(getRbuf           ),
    LUA_FUN(getNbuf           ),
    LUA_FUN(getBufBytes       ),
    LUA_FUN(getSegBytes       ),
    LUA_FUN(getMem            ),
    LUA_FUN(getDatacode       ),
    LUA_FUN(getDatasize       ),
    LUA_FUN(getCnt            ),
    LUA_FUN(getSizeofMsgHeader),
    LUA_FUN(getSeg_id         ),
    LUA_FUN(getFmtValue       ),
    LUA_FUN(getValid          ),
    {0,0}
};

*/



#define showstack(lstack)   \
     for (int i=0;i<lua_gettop(lstack)+1;i+=1)\
     {\
         using namespace std; \
         if (lua_isnumber(lstack,i))\
         {\
             cout << i << ": number : " << lua_tonumber(lstack,i) << endl;\
         }\
         else if (lua_isstring(lstack,i))\
         {\
             cout << i << ": string : " << lua_tostring(lstack,i) << endl;\
         }\
         else if (lua_istable(lstack,i))\
         {\
             cout << i << ": table" << endl;\
         }\
         else if (lua_iscfunction(lstack,i))\
         {\
             cout << i << ": cfunction" << endl;\
         }\
         else if (lua_isfunction(lstack,i))\
         {\
             cout << i << ": function" << endl;\
         }\
         else if (lua_isboolean(lstack,i))\
         {\
             if (lua_toboolean(lstack,i)==true)\
             cout << i << ": boolean : true" << endl;\
             else\
             cout << i << ": boolean : false" << endl;\
         }\
         else if (lua_isuserdata(lstack,i))\
         {\
             cout << i << ": userdata" << endl;\
         }\
         else if (lua_isnil(lstack,i))\
         {\
             cout << i << ": nil" << endl;\
         }\
         else if (lua_islightuserdata(lstack,i))\
         {\
             cout << i << ": light userdata" << endl;\
         }\
     }


static void init_malt2(lua_State *L)
{
using namespace dStorm;
     cout<<"\nExecuting init_dstorm(lua_state *L) ..."<<std::endl;
     cout<<"L@"<<(void*)L<<std::endl;
     //showstack(L);
     //lua_pushstring(L,"IoNet");

 #define LUA_ENUM( E ) lua_pushstring( L, #E ); lua_pushnumber(L,E); lua_rawset(L,-3);
     // ---------------------------- global table: IoNet.*
     // These are the "builtin" ionet handles
     lua_newtable(L);
     LUA_ENUM( ALL );
     LUA_ENUM( SELF );
     LUA_ENUM( CHORD );
     LUA_ENUM( HALTON );
     LUA_ENUM( RANDOM );
     LUA_ENUM( PARA_SERVER );
     //LUA_ENUM( BUTTERFLY );
     LUA_ENUM( IONET_MAX );
     // If lua were to call Dstorm::add_ionet, it would extend the IoNet table
     lua_setglobal(L, "IoNet"); // lua IoNet.ALL/CHORD/...

     // ---------------------------- global table: Consistency.*
     // These are the builtin handling strategies for accepting/rejecting
     // input buffers and using barriers
     lua_newtable(L);
     LUA_ENUM( BSP );    // Bulk Synchronous (add barriers, never stale)
     // Consistency.SYNC easier to remember (BSP/SSP confusing)
     lua_pushstring( L, "SYNC"); lua_pushnumber(L,BSP); lua_rawset(L,-3);
     //
     LUA_ENUM( ASYNC ); // no barriers, no staleness check (check: milde_malt version had some bogus test code here!)
     //
     LUA_ENUM( SSP );   // Bounded Staleness (async + handle "old" messages (reject? wait?) <-- which is it?
     // Consistency.BOUND is easier to remember (BSP/SSP confusing)
     lua_pushstring( L, "BOUND"); lua_pushnumber(L,SSP); lua_rawset(L,-3);
     //
     lua_setglobal(L, "Consistency");    // lua Consistency.SYNC/ASYNC/BOUND

     lua_newtable(L);

 //LUA_ENUM(SEG_FULL);
     //LUA_ENUM(SEG_ONE);
     lua_pushstring( L, "FULL"); lua_pushnumber(L,SEG_FULL); lua_rawset(L,-3);   // lua Seg.FULL is enum value SEG_FULL
     lua_pushstring( L, "ONE" ); lua_pushnumber(L,SEG_ONE);  lua_rawset(L,-3);   // lua Seg.ONE  is enum value SEG_ONE
     LUA_ENUM( REDUCE_AVG_RBUF );        // reduce: iBuf = avg( rBufs )
     LUA_ENUM( REDUCE_AVG_RBUF_OBUF );   // reduce: oBuf = avg( oBuf and rBufs )
     LUA_ENUM( REDUCE_SUM_RBUF );        // reduce: iBuf = sum( rBufs )
     LUA_ENUM( REDUCE_STREAM );          // reduce: iBuf = sum( rBufs )
     LUA_ENUM( REDUCE_NOP );             // reduce: no-op
     // SegImpls vary in what they have coded for.  Requesting too difficult subview
     // handling for the SegImpl should throw during "add_segment".
     LUA_ENUM( RBUF_SUBVIEW_NONE );      // reduce rBufs required to be full-sized
     LUA_ENUM( RBUF_SUBVIEW_HOMOG );     // reduce rBufs subviews must all match (homogenous)
     LUA_ENUM( RBUF_SUBVIEW_HOMOG_OR_NONOVLP );  // reduce rBufs may also be non-overlapping subviews
     LUA_ENUM( RBUF_SUBVIEW_OVLP_RELAXED );  // reduce rBufs may also be non-overlapping subviews
     LUA_ENUM( RBUF_SUBVIEW_ANY );       // reduce rBufs could be any arbitrary subviews
     // If erroneous rBuf subviews are encountered during reduce at runtime...
     LUA_ENUM( SUBVIEW_ERR_THROW );
     LUA_ENUM( SUBVIEW_ERR_WARN );
     LUA_ENUM( SUBVIEW_ERR_IGNORE );
     //
     LUA_ENUM( SEGSYNC_NONE );
     LUA_ENUM( SEGSYNC_NOTIFY );
     LUA_ENUM( SEGSYNC_NOTIFY_ACK );
     lua_setglobal(L,"Seg");
     //luaT_pugpuetatable(L, "THFloatStorage");
     lua_newtable(L);
     lua_pushstring(L,"TEST"); lua_pushnumber(L,ORM_TEST); lua_rawset(L,-3);
     // XXX scr_XXX have NO support for uint64_t, and scr_INT doesn't work for it ???
     lua_pushstring(L,"BLOCK"); lua_pushnumber(L,999999999); lua_rawset(L,-3);
     lua_pushstring(L,"SUCCESS"); lua_pushnumber(L,ORM_SUCCESS); lua_rawset(L,-3);
     lua_pushstring(L,"TIMEOUT"); lua_pushnumber(L,ORM_TIMEOUT); lua_rawset(L,-3);
     LUA_ENUM( NTF_RUNNING );
     LUA_ENUM( NTF_DONE );
     LUA_ENUM( NTF_ACK );
     LUA_ENUM( NTF_SELECT );
     // etc. as convenient
     lua_setglobal(L,"Orm");

#undef LUA_ENUM

     showstack(L);

     luaT_stackdump(L);
     orm_printf ("malt initialized.");

}



int luaopen_malt2(lua_State *L)
{
   if ( d_si == nullptr )
   {
     d_si = new lua_Interpreter( L );
     d_si_alloc = true;
   }

   d_si->register_namespace (NULL, "dstorm", dstorm_lib_f );
   // Set to lua error handler
   // This function is called from a require
   // There are two calling patterns:
   // (1) Lua or QtLua perform require("milde").
   //     The error handler is set here and stays that way.
   // (2) MildeApp (windows) only performs require("milde")
   //     by calling lua through the lua_Interpreter functions.
   //     These functions set the error handler on entry
   //     and restore it on exit. Therefore they will
   //     reset the error handler to its value before
   //     calling lua.

   //set_error_handler( lua_Interpreter::error_handler);   
   init_malt2(L);
}

#endif // MALT_lua_dstorm.cuh

