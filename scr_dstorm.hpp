/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
/** \file lua_dstorm.hpp
 *  (C) NECLA
 * set up lua interface to dstorm 
 *
 * TODO this is wrong to have non-inline in the header.
 * Those funcs should move to malt2.cpp
 */
#ifndef TORCH_SCR_DSTORM_HPP
#define TORCH_SCR_DSTORM_HPP

#include "dstorm_env.h"
#if WITH_GPU
#include "mtypes.hpp"           // Sint4, for GPU code
#include "segInfo.hpp"          // could avoid if pass just the segNum as arg to gpu_store_helper
#endif

#include <cstdint>

/** This wraps the Dstorm object and constructs it with any
 * configuration stuff passed in from lua. */
struct scr_Dstorm
{
    //static int f_new();
    //static int f___gc();
    // we are not an object, dStorm is a singleton (so far), so just provide an 'init' function
    // that sets a global ptr to a Dstorm object
    /** After a lua <TT>local X = require 'foo' .init("shm")</TT> you may call
      * dstorm functions. Option selects shm transport (other transports removed).
      * shm is the default (because it should always be supported)
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
    /** lua args <segnum:int>, [opt wgt=1.0,] <lua tensor data> */
    static int f_store();
    static int f_push();
    static int f_reduce();
    // special utilities
    static int f_barrier();
    //static int f_setStreamFunc(); // install a lua REDUCE_STREAM callback function
    //nDeadCount
    //nDead
    static int f_wait();
    //netSendVec
    //netRecvVec
    //segSendVec
    //segRecvVec
    // debug
    //print_XXX functions
    static int f_getTransport(); ///< returns string MPI, GPU  (,SHM?)
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
    //static int f_obuf_dpoint();         ///< export the oBuf buffer of a Dstorm segment as a lua dpoint
    //static int f_ibuf_flat_tensor();    ///< export the iBuf buffer of a Dstorm segment as a flat lua tensor
    //static int f_ibuf_vec();            ///< export the iBuf buffer of a Dstorm segment as a lua vec
    //static int f_ibuf_dpoint();         ///< export the iBuf buffer of a Dstorm segment as a lua dpoint
    //@}
    /// \name SegInfo getters
    /// all take a segtag parameter (same you use for add_segment) and return 1|0 for ok|error.
    /// \sa Dstorm::SegInfo for the various SegInfo fields
    //@{
    static int f_getIoNet          ();
    static int f_getPolicy         ();
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
#if 0 // these were for milde types
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
#endif
#if 1 // these are for torch types
    /** return a non-allocating TH<type>Tensor appropriate to the data in a segment.
     * \p seg   which segment
     * \p buf   which buffer of segment (from getIbuf/Rbuf/Obuf)
     * These are created once, during create_segment and retained.
     * They are deleted (and storage set to nullptr) during delete segment)
     */
    template<typename T>
        static int segbuf_torch_tensor1D( T * const data, uint_least32_t const cnt );
#endif
#if WITH_GPU
    // Everywhere where scr_dstorm.cpp had VecDense hardwired, we now should
    // check if( GD->transport == GPU ) {
    //    ... invoke a gpu-helper using VecGpu
    //    ... and cuda compilation
    //  }
    /** gpu helper. \return \# items pushed onto lua stack.
     * This is called AFTER dStorm::reduce(segnum) has already been called. */
    static int gpu_reduce_helper( ::dStorm::SegInfo const& sInfo, uint32_t const nReduce );

    /** gpu helper. \return \# items pushed onto lua stack.
     * - This should invoke Dstorm::add_segment operation,
     *   to create a seg::VecGpu type of segment.
     * - Sint4 is for scr_INT macro,
     * - ccptr is for scr_STR macro (\ref mtypes.hpp)
     */
    static int gpu_add_segment_helper( Sint4 const segnum
                                       , Sint4 const ionet
                                       , Sint4 const segpolicy
                                       , Sint4 const nPoints
                                       , ccptr const type);

    /** Invoke Dstorm::store operation for a GPU segment.
     * - gpu helper for store of lua tensor data region.
     * - \tparam SEG_VEC is user::Seg_VecGpu<T> for T=float or double
     * - \tparam CONST_ITER typically float* or double*
     * - \return # items [0] pushed onto lua stack
     */
    template< typename SEG_VEC, typename CONST_ITER>
        static int gpu_store_helper( ::dStorm::SegInfo const& sInfo
                                     , CONST_ITER beg
                                     , uint32_t const cnt
                                     , uint32_t const off
                                     , double wgt );
#endif
    //@}
    // Dstorm * d;      // maybe cheat by having a singleton (global) Dstorm object ?
};



#endif // TORCH_SCR_DSTORM_HPP

