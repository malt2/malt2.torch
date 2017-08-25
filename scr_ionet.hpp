/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef MILDE_scr_ionet_hpp
#define MILDE_scr_ionet_hpp

#include "dstorm_fwd.hpp"
//#include "ionet/ionet_fwd.hpp"        // darn templated constructor... need impl!
#include "ionet/userIoNet.hpp"
#include "demangle.hpp"

#include <iostream>

#define tT template<typename T>

class scr_ionet;        // <ionet>:foo() object functions
struct script_ionet;    // ionet.foo() static functions (no object on stack)
struct script_DSTORM;   // dstorm.foo() functions
struct lua_Interpreter;

class scr_ionet
{
private:
    static constexpr bool verbose = false;

    /** we initially hold an ionet object */
    mm2::UserIoNet *ionetImpl;

    /** after conversion to a unique pointer, passed to Dstorm,
     * we can transform into a small \c IoNet_t handle (u8), and a Dstorm* */
    dStorm::IoNet_t ionetHandle;
    dStorm::Dstorm *owner;
public:
    friend class script_ionet;
    friend class script_DSTORM;
    friend class scr_Dstorm;

    /** take ownership of a T *& to a mm2::user::IoNetXXX impl. \post x == nullptr. */
    tT scr_ionet( T *&x )
        : ionetImpl(x), ionetHandle( dStorm::IoNet_t(-1) ), owner( nullptr )
    {
        x = nullptr;
        if( ionetImpl != nullptr ) if(verbose) std::cout<<" +scr_ionet( "<<ionetImpl->shortname()<<" *& )";
    }

    ~scr_ionet();

    char const* type();
    char const* name();     // a.k.a. type
    char const* shortname();
    mm2::Tnode verts();
    /** Can this be done without knowing about the lua Dstorm object?
     * I'd like the lua ionet stuff to be as separate as possible.
     *
     * Once registered we can keep a ptr to the Dstorm object
     * (and accept undefined behavior once the Dstorm object is destroyed.) */
    char const* pprint();
#if 0 // TBD
    /** Pass ownership Dstorm object, representing ourself as a Dstorm* and a handle.
     *
     * Once registered we can keep a ptr to the Dstorm object
     * (and accept undefined behavior once the Dstorm object is destroyed.)
     */
    dStorm::IoNet_t register( dStorm::Dstorm *d ){
        ...;
        return this->ionetHandle;
    }
#endif

    //static bool constexpr is_scr_ionet() { return true; }

    tT static scr_ionet* new_stack( T *& ionet );
};


#undef tT
#endif // MILDE_scr_ionet_hpp
