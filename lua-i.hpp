/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
// FIXME: #include files are MISSING XXX TODO
#include <lua.hpp>        // lua_State, basic lua interface
//#include <lauxlib.h>    // luaL_* functions
#include <luaT.h>     // lua_pushglobaltable ... advanced lua_ functions
//#include <lualib.h>

#include <map>
#include <cstring>      // strcmp
#include <sstream>
#include <iostream>

class lua_Interpreter   {
    public:
    lua_State *L;
    lua_Interpreter(lua_State *inL);
    bool try_ccptr ( int n, ccptr&  x );
    bool try_ccstr( int n, ccstr&  x );
    int  cnt_stack ();
    int  chk_stack ( int nmin, int nmax, ccptr msg );
    void put_var   ( ccptr  x );               // variable name
    void put_bool  ( bool   x );
    void put_sidx  ( sidx   x );               // 32/64bit
    void put_Sint4 ( Sint4  x );
    void put_Uint4 ( Uint4  x );
    void put_Sint8 ( Sint8  x );
    void put_Uint8 ( Uint8  x );
    void put_int   ( int    x );
    void put_real4 ( real4  x );
    void put_real8 ( real8  x );
    void put_double( real8  x );
    void put_ccptr ( ccptr  x );        // ccptr is for C-strings
    void put_ccptr(char* x, int sz);    // or C blobs (maybe with embedded nulls)
    void put_lightuserdata(void *x);    // push a raw [un-managed, no lua meta-table] pointer
    ccstr param_type( int n );

#define Opd(T) \
    bool try_##T (int n, T& x ); 
        Opd(sidx)
        Opd(uidx)
        Opd(Sint1)
        Opd(Sint2)
        Opd(Sint4)
        Opd(Sint8)
        Opd(Uint4)
        Opd(Uint8)
        Opd(int)
        Opd(real4)
        Opd(real8)
#undef Opd
    //bool try_real8 ( int n, real8&  x );
    template < class U > // allocates new user type U with tag T (but does not construct
    inline vptr new_user( ccptr T ) { return put_user( T, sizeof(U) ); }
    //void register_namespace( const char* mid, const char* sname, const luaL_Reg ftable[] );
    /** malt2 extension -- allow some static integer values under [mid.]sname.foo */
    void register_namespace( const char* mid, const char* sname, const luaL_Reg ftable[], std::map<char const*,int> settings=std::map<char const*,int>() );
    vptr put_user ( ccptr  t, int sz );
    vptr  try_user_tp ( int n, ccptr   t, bool& wrong_tp);
    vptr  try_user    ( int n, ccptr   t                )
    { bool unused; return try_user_tp( n, t , unused); }
};



inline ccstr lua_Interpreter::param_type( int n )
{
    std::string type = lua_typename( L, lua_type( L, n ) );

    if ((type == "userdata") && (luaL_getmetafield( L, n, "__classnm" ))) // +1 to stack if successful
    {
        if (lua_isstring( L, -1 ))
            type = lua_tostring( L, -1 );
        lua_pop( L, 1 );
    }

    return type;
}

inline vptr lua_Interpreter::try_user_tp( int n, ccptr   t , bool& wrong_tp)
{
    int addstack = 0;
    wrong_tp=false;

    // get pointer to userdata
    vptr p = lua_touserdata( L, n ); if ( p == 0 ) return 0;
    if ( t == nullptr ) return p; // NULL pointer means we do not care about the type

    // get registry metatable of this userdata (see register_class above)
    lua_getfield( L, LUA_REGISTRYINDEX, t ); addstack++;

    // get metatable from object
    if (! lua_getmetatable( L,  n ) ) { lua_pop( L, addstack ); return 0; }
    addstack++;

    // if they are the same we are good, return (after removing leftover from the stack)
    if ( lua_rawequal( L, -1, -2 ) ) {
        lua_pop( L, addstack );
        return p;
    }

    // Cout() << "TRYUSER: object not of type: " << t << " (stk=" << lua_gettop(L) << "; addstk=" << addstack << ")" << endl;

    //printf ("TRYUSER: object not of type: %s (stk=%p addstk=%p.\n ", t, lua_gettop(L) , addstack);
    // This is part of a scheme to allow inheritance with userdata
    // if a userdata is derived from another, this is indicated by the "__derived_from" field of the metatable
    // which contains the ID of the baseclass. This allows to check whether the passed userdata is a subclass of
    // the desired userdata.
    // The other part of the scheme is executed in the initialization scripts in Lua and consist of copying
    // the methods of the baseclass to the subclasses and adding the field "__derived_from"
    while (1) {
        lua_getfield( L, -1, "__derived_from" );
        addstack++;

        if (! lua_isstring( L, -1 )) break;

        const char *x = lua_tostring( L, -1 );
        // Cout() << "TRYUSER: checking: " << x << " (stk=" << lua_gettop(L) << "; addstk=" << addstack << ")" << endl;
        if ( strcmp( t, x ) == 0 ) {
            lua_pop( L, addstack);
            return p;
        } else { // recursively check
            lua_getfield( L, LUA_REGISTRYINDEX, x ); addstack++;
        }
    }

    //Cout() << "TRYUSER: failed (stk=" << lua_gettop(L) << "; addstk=" << addstack << ")" << endl;
    //printf ("TRYUSER: failed stk=%p, addstk = %p.\n", lua_gettop(L), addstack);
    lua_pop( L, addstack );
    wrong_tp=true;
    return 0;
}

inline vptr lua_Interpreter::put_user  ( ccptr  t, int sz )       // t = tag name of the type
{
    // create userdata block of desired size and put it on the lua stack
    vptr* p = (vptr*) lua_newuserdata( L, sz );

    // get registry metatable of this userdata (see register_class above)
    luaL_getmetatable( L, t );

    if (lua_isnil( L, -1 ))
    {
        lua_pop( L, 2 );
        //DBG("Userdata " + string(t) + " has no metatable in the Lua Registry. Did you forget to load the library?");
    }

    // sets the userdata metatable
    lua_setmetatable ( L, -2 );
    return p;
}


static void get_module(lua_State* L, const char* modname)
{
    lua_getglobal(L,modname);
    if (lua_isnil(L,-1))
    {
        lua_pop(L, 1);
        lua_newtable(L);
        lua_setglobal(L,modname);
        lua_getglobal(L,modname);
    }
}

//#define luaL_newlibtable(L,l)   \
//      lua_createtable(L, 0, sizeof(l)/sizeof((l)[0]) - 1)
#define luaL_newlibtable(L,l)   \
      lua_createtable(L, 0, sizeof(*l)/sizeof((l)[0]) - 1)

#if !defined LUA_VERSION_NUM || LUA_VERSION_NUM==501
/*
** Adapted from Lua 5.2.0
*/
static void luaL_setfuncs (lua_State *L, const luaL_Reg *l, int nup) {
  luaL_checkstack(L, nup+1, "too many upvalues");
  for (; l->name != NULL; l++) {  /* fill the table with given functions */
    int i;
    lua_pushstring(L, l->name);
    for (i = 0; i < nup; i++)  /* copy upvalues to the top */
      lua_pushvalue(L, -(nup+1));
    lua_pushcclosure(L, l->func, nup);  /* closure with those upvalues */
    lua_settable(L, -(nup + 3));
  }
  lua_pop(L, nup);  /* remove upvalues */
}
#endif


// Merges with existing (sub)module or creates a new one.
inline void lua_Interpreter::register_namespace( const char* mid, const char* sname, const luaL_Reg ftable[], std::map<char const*,int> settings )
{
    // put mid (or global) on top of stack, creating mid if it does not exist
    if ( mid )
        get_module       ( L, mid );
    else
        lua_pushglobaltable( L );

    // create mid.sname if it does not exist
    lua_getfield       ( L, -1, sname );
    if ( lua_isnil( L, -1 ) )
    {
        lua_pop          ( L,  1 );
        luaL_newlibtable ( L, ftable );
        lua_setfield     ( L, -2, sname );
        lua_getfield     ( L, -1, sname );
    } else
    {
        Derr_msg( lua_istable( L, -1 ), true, "tried to overwrite non-table member of " << mid << " with " << sname);
    }

    luaL_setfuncs      ( L, ftable, 0 ); // 0 means no up-values

    // Malt2 extension:
    //     allow some static integer values under [mid.]sname.foo
    for( auto const kv: settings ){
        lua_pushnumber(L,kv.second);
        lua_setfield(L,-2,kv.first);
    }

    // return the new module table
    lua_replace        ( L, -2 );
}



inline void lua_Interpreter::put_var   ( ccptr  x )               // variable name
{
    lua_getglobal( L, x );
}

inline void lua_Interpreter::put_bool  ( bool   x )
{
    lua_pushboolean( L, x ? 1 : 0 );
}

inline void lua_Interpreter::put_sidx  ( sidx   x )               // 32/64bit
{
    lua_pushnumber( L, (lua_Number)x );
}

inline void lua_Interpreter::put_Sint4 ( Sint4  x )
{
    lua_pushnumber( L, x );
}

inline void lua_Interpreter::put_Sint8 ( Sint8  x )
{
    lua_pushnumber( L, (lua_Number)x );
}

inline void lua_Interpreter::put_Uint8 ( Uint8  x )
{
    // luajit and lua-5.2 may behave differently
    lua_pushnumber( L, (lua_Number)x );
}

inline void lua_Interpreter::put_Uint4 ( Uint4  x )
{
    lua_pushnumber( L, x );
}

inline void lua_Interpreter::put_int   ( int    x )
{
    put_Sint4( x );
}

inline void lua_Interpreter::put_real4 ( real4  x )
{
    lua_pushnumber( L, x );
}

inline void lua_Interpreter::put_real8 ( real8  x )
{
    lua_pushnumber( L, x );
}

inline void lua_Interpreter::put_double( real8  x )
{
    put_real8( x );
}

inline void lua_Interpreter::put_ccptr ( ccptr  x )
{
    lua_pushstring( L, x );
}

inline void lua_Interpreter::put_ccptr(char* x, int sz)
{
    lua_pushlstring(L, x, sz); // properly handles strings with embedded '\0'
}

inline void lua_Interpreter::put_lightuserdata(void *x)
{
    lua_pushlightuserdata(L, x); // properly handles strings with embedded '\0'
}

inline bool  lua_Interpreter::try_real8 ( int n, real8&  x )
{
    if ( lua_isnumber( L, n ) == 0 ) return false;
    x =  lua_tonumber( L, n );
    return true;
}

inline bool lua_Interpreter::try_ccptr ( int n, ccptr&  x )
{
    if ( lua_isstring( L, n ) == 0 ) return false;
    x =  lua_tostring( L, n );
    return true;
}

inline bool lua_Interpreter::try_ccstr( int n, ccstr&  x )
{
    if ( lua_isstring( L, n ) == 0 ) return false;
    size_t l; // this properly handles strings with embedded '\0'
    ccptr  p = lua_tolstring( L, n, &l );
    x.assign( p, l );
    return true;
}

// stack accessing (no pop!) functions
inline int lua_Interpreter::cnt_stack () // return # args on the stack
{
  return lua_gettop( L );
}

inline int lua_Interpreter::chk_stack ( int nmin, int nmax, ccptr msg )
{
  int n = cnt_stack();
  Derr_msg( n <= nmax && n >= nmin, true, msg );
  return n;
}

inline lua_Interpreter::lua_Interpreter(lua_State *inL)    
{
    L = inL;
}

#define Op(T) \
    inline bool lua_Interpreter::try_##T (int n, T& x ) \
{ \
    real8 _x; \
    if (!try_real8(n, _x)) return false; \
    return safe_numcast( _x, x ); \
}
    Op(sidx)
    Op(uidx)
    Op(Sint1)
    Op(Sint2)
    Op(Sint4)
    Op(Sint8)
//#if !defined(__CUDACC__)
    Op(Uint4)
    Op(Uint8)
//#endif
    Op(int)
Op(real4)
#undef Op
