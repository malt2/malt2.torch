/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#include "scr_dstorm.hpp"

#include "dstorm.hh"
//#include "dstorm_fwd.hpp"
//#include "dstorm_msg.hh"
//#include "segVecDense.hh"
//
//#include "TH/TH.h"
//#include "TH/THGeneral.h"
//#include "TH/THTensor.h"
//#include "luaT.h"
////#include "TH/THGenerateAllTypes.h"
////#include "TH/THGenerateFloatTypes.h"
//
//
//#include <string>
//#include <exception>
//#include <iostream>
//#include <limits>
//#include <vector>
//
//#include "lua.h"
//#include "lualib.h"
//#include "lauxlib.h"
//#include <luaT.h>

using namespace std;
using dStorm::Dstorm;
using dStorm::Transport;

#if WITH_SHM
using dStorm::SHM;
#endif
using dStorm::SegInfo;

#include "mtypes.hpp"
#include "lua-i.hpp"

#include "scr_ionet.hpp"
#include <string>
#include <exception>
#include <iostream>

/** For now every process has its own Dstorm object.
 *
 * The name \em globalDstorm really means "process-local" for now!
 *
 * Eventually Dstorm configuration and "const-ish" data
 * should be moved into shared memory.
 *
 * note: thread_local is ok in shared libs, but mandates -fPIC in all units referencing it
 **/
thread_local Dstorm *          globalDstorm = nullptr;
thread_local lua_Interpreter * d_si = NULL;
static thread_local bool       d_si_alloc = false;

template<typename T> scr_ionet* scr_ionet::new_stack( T *& ionet ) {
    std::cout<<" scr_ionet::new_stack< "<<demangle(typeid(ionet).name())<<" >"<<std::endl;
    return new (d_si->new_user< scr_ionet >( OBJNAME(scr_ionet)))
        scr_ionet( ionet );
}
template scr_ionet* scr_ionet::new_stack(mm2::user::IoNetEmpty         *& x);
template scr_ionet* scr_ionet::new_stack(mm2::user::IoNetSelf          *& x);
template scr_ionet* scr_ionet::new_stack(mm2::user::IoNetAll           *& x);
template scr_ionet* scr_ionet::new_stack(mm2::user::IoNetChord         *& x);
template scr_ionet* scr_ionet::new_stack(mm2::user::IoNetHalton        *& x);
template scr_ionet* scr_ionet::new_stack(mm2::user::IoNetRandom        *& x);
template scr_ionet* scr_ionet::new_stack(mm2::user::IoNetParaServer    *& x);


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
    FUN(getTransport)   // returns lua string ( mpi | gpu )
    //FUN(obuf_flat_tensor)
    //FUN(obuf_vec)
#if 1
    // special utils
    //nDeadCount
    //nDead
    
    //FUN(setStreamFunc)
    //netSendVec
    //netRecvVec
    //segSendVec
    //segRecvVec
    // debug
    //print_XXX functions
    
    
    //FUN(obuf_dpoint)
    //FUN(ibuf_flat_tensor)
    //FUN(ibuf_vec)
    //FUN(ibuf_dpoint)
    // SegInfo getters scr_Dstorm::f_getXXX( SegNum const s ) --> ...
    FUN(getIoNet          )
    FUN(getPolicy         )
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
    LUA_FUN(getTransport),
    //LUA_FUN(obuf_flat_tensor),
    //LUA_FUN(obuf_vec),
#if 1
    //LUA_FUN(setStreamFunc),
    //
    
    
    //LUA_FUN(obuf_dpoint),
    //LUA_FUN(ibuf_flat_tensor),
    //LUA_FUN(ibuf_vec),
    //LUA_FUN(ibuf_dpoint),
    // SegInfo getters scr_Dstorm::f_getXXX( SegNum const s ) --> ...
    LUA_FUN(getIoNet          ),
    LUA_FUN(getPolicy         ),
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
#endif
    {0,0}
};

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
    //cout<<"\nExecuting init_dstorm(lua_state *L) ..."<<std::endl; //debug messages
    //cout<<"L@"<<(void*)L<<std::endl;
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
    //
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
    // SegPolicy are single-bit flags covering different subpolicy options
    LUA_ENUM( SEG_LAYOUT_MASK);
    //LUA_ENUM(SEG_FULL);
    //LUA_ENUM(SEG_ONE);
    lua_pushstring( L, "FULL"); lua_pushnumber(L,SEG_FULL); lua_rawset(L,-3);   // lua Seg.FULL is enum value SEG_FULL
    lua_pushstring( L, "ONE" ); lua_pushnumber(L,SEG_ONE);  lua_rawset(L,-3);   // lua Seg.ONE  is enum value SEG_ONE
    //
    LUA_ENUM( REDUCE_OP_MASK);
    LUA_ENUM( REDUCE_AVG_RBUF );        // reduce: iBuf = avg( rBufs )
    LUA_ENUM( REDUCE_AVG_RBUF_OBUF );   // reduce: oBuf = avg( oBuf and rBufs )
    LUA_ENUM( REDUCE_SUM_RBUF );        // reduce: iBuf = sum( rBufs )
    LUA_ENUM( REDUCE_STREAM );          // reduce: iBuf = sum( rBufs )
    LUA_ENUM( REDUCE_NOP );             // reduce: no-op
    // SegImpls vary in what they have coded for.  Requesting too difficult subview
    // handling for the SegImpl should throw during "add_segment".
    LUA_ENUM( RBUF_SUBVIEW_MASK );
    LUA_ENUM( RBUF_SUBVIEW_NONE );      // reduce rBufs required to be full-sized
    LUA_ENUM( RBUF_SUBVIEW_HOMOG );     // reduce rBufs subviews must all match (homogenous)
    LUA_ENUM( RBUF_SUBVIEW_HOMOG_OR_NONOVLP );  // reduce rBufs may also be non-overlapping subviews
    LUA_ENUM( RBUF_SUBVIEW_OVLP_RELAXED );  // reduce rBufs may also be non-overlapping subviews
    LUA_ENUM( RBUF_SUBVIEW_ANY );       // reduce rBufs could be any arbitrary subviews
    // If erroneous rBuf subviews are encountered during reduce at runtime...
    LUA_ENUM( SUBVIEW_ERR_MASK );
    LUA_ENUM( SUBVIEW_ERR_THROW );
    LUA_ENUM( SUBVIEW_ERR_WARN );
    LUA_ENUM( SUBVIEW_ERR_IGNORE );
    //
    LUA_ENUM( SEGSYNC_MASK );
    LUA_ENUM( SEGSYNC_NONE );
    LUA_ENUM( SEGSYNC_NOTIFY );
    LUA_ENUM( SEGSYNC_NOTIFY_ACK );
    //
    lua_setglobal(L,"Seg");

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
    //
    lua_setglobal(L,"Orm");

    //luaT_pushmetatable(L, "THFloatStorage");
#undef LUA_ENUM

    // uncomment below messages for debug
    // showstack(L);
    //luaT_stackdump(L);
    //printf ("malt initialized.");

}

extern "C" {
    DLL_EXPORT int luaopen_malt2(lua_State *L)
    {
        if ( d_si == nullptr )
        {
            d_si = new lua_Interpreter( L );
            d_si_alloc = true;
        }

        typedef std::map<char const*, int> Settings;
        Settings dstorm_settings;
        dstorm_settings.insert(Settings::value_type("WITH_LIBORM",WITH_LIBORM));
        dstorm_settings.insert(Settings::value_type("WITH_MPI",WITH_MPI));
        dstorm_settings.insert(Settings::value_type("WITH_GPU",WITH_GPU));
        dstorm_settings.insert(Settings::value_type("WITH_NOTIFYACK",WITH_NOTIFYACK));

        d_si->register_namespace (NULL, "dstorm", dstorm_lib_f, dstorm_settings );
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
        return 1;
    }
}//extern "C"
