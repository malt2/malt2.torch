/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef MTYPES_HPP
#define MTYPES_HPP

#include <cstdint>
#include <limits>       // numeric_limits
#include <cmath>        // isnan
#include <string>

#define STREAMSTR(msg) ((dynamic_cast<std::ostringstream &>(std::ostringstream() << std::string() << msg)).str())
//Uncomment line below to print debug messages
//#define DBG(msg) do { if(1){ std::ostringstream oss; oss << " r"<<GD->iProc<<": "<<msg << endl; printf("%s\n",oss.str().c_str()); }}while(0)
#define DBG(msg) do { if(0){ std::ostringstream oss; oss << " r"<<GD->iProc<<": "<<msg << endl; printf("%s\n",oss.str().c_str()); }}while(0)

#ifdef __FAST_MATH__
#error -ffast-math should not be enabled because it interferes with functions that rely on the IEEE float spec (i.e. isnan)
#endif

#ifdef _WIN32
#include <float.h>
#define isnan(x) _isnan((double)x)
#define isfinite(x) _finite((double)x)
#else
#define isnan(x)    std::isnan(x)
#define isfinite(x) std::isfinite(x)
#endif

// OHOH nvidia Uint4 is a SIMD vector and this conflicts with milde's definition
//      milde Uint4 is really uint_least32_t
//      nvcc  Uint4 is a SIMD vector of 4 32-bit integers (I think)
//      see /usr/local/cuda-8.0/targets/x86_64-linux/include/vector_types.h
// To support cuda, we CANNOT use some of the milde types.
//      real4 and real8 are still OK.
//      Uint1 Uint2 and Uint4 are the names with conflicts
// We can avoid conflicts if we use rewrite the milde names with first letter capitalized.
typedef        float real4; // 32-bit
typedef       double real8; // 64-bit
//typedef  long double realA; // 80-bit
//typedef std::complex<real4>  cplx4; // there is confusion between
//typedef std::complex<real8>  cplx8; // c complex and c++ complex

#ifdef _WIN32
   typedef          __int8      Sint1;
   typedef unsigned __int8      Uint1;
   typedef          __int16     Sint2;
   typedef unsigned __int16     Uint2;
   typedef          __int32     Sint4;
   typedef unsigned __int32     Uint4;
   typedef          __int64     Sint8;
   typedef unsigned __int64     Uint8;
#else
//#if defined(__CUDACC__)
//#include "vector_types.h"       // CUDA-8.0 supplies some of these, don't redefine them
// hopefully they are the same width as the "milde" definitions :)
//
//#else
   typedef  uint_least8_t  Uint1;
   typedef  uint_least16_t Uint2;
   typedef  uint_least32_t Uint4;
//#endif
   typedef    int_least8_t Sint1;
   typedef   int_least16_t Sint2;
   typedef   int_least32_t Sint4;
   typedef   int_least64_t Sint8;
   typedef  uint_least64_t Uint8;
#endif

   typedef void* vptr;
   typedef char* cptr;

#if defined(__x86_64__) || defined (_WIN64)
   typedef Sint8 sidx; // this two types are supposed to reflect architecture
   typedef Uint8 uidx; // 32-bit or 64-bit depending on the compilation mode
#else
   typedef Sint4 sidx; // do not use "(un)signed long" since gcc treats it as
   typedef Uint4 uidx; // unique type even when it is identical to (s|u)int(4|8)
#endif

// Converts x from T1 to T2 only if x can be represented as a T1.
// Only loss of precision is allowed. Anything that would result
// in undefined behaviour such as conversions of a value outside
// the range of T2 will return false.
template<typename T1, typename T2> inline
__attribute__ ((warn_unused_result))
bool safe_numcast(const T1& x, T2& y)
{
    typedef std::numeric_limits<T1> limT1;
    typedef std::numeric_limits<T2> limT2;
    if (limT1::has_quiet_NaN && !limT2::has_quiet_NaN) {
        if (isnan(x)) return false;
    }
    // check for < 0 explicitly to avoid promotion shenanigans below when T2 is unsigned
    if (limT1::is_signed && !limT2::is_signed) {
        if (x < T1(0)) return false;
    }

    if (x < limT2::lowest() || x > (limT2::max)()) return false;
    //if(x < static_cast<T1>(limT2::lowest()) || x > static_cast<T1>(limT2::max())) return false;

    y = (T2)x;
    return true;
}

#if 0 && defined(__CUDACC__)
template<> inline
__attribute__ ((warn_unused_result))
bool safe_numcast(const real8& x, Uint4& y)
{
    typedef std::numeric_limits<T1> limT1;
    typedef std::numeric_limits<T2> limT2;
    if (limT1::has_quiet_NaN && !limT2::has_quiet_NaN) {
        if (isnan(x)) return false;
    }
    // check for < 0 explicitly to avoid promotion shenanigans below when T2 is unsigned
    if (limT1::is_signed && !limT2::is_signed) {
        if (x < 0) return false;
    }
    if (x < limT2::lowest() || x > (limT2::max)()) return false;
    y = (T2)x;
    return true;
}
#endif

#if defined(_WIN32)
# define PRETTY_FUNC   __FUNCTION__
#else
# define PRETTY_FUNC   __PRETTY_FUNCTION__
#endif

int malt2_err_fun( bool Abort, const int Line, const std::string& File, const std::string& Func, const std::string& Msg );
#define Derr_msg(Cond,Abort,Msg) do{ if (!(Cond)) malt2_err_fun( Abort, __LINE__, __FILE__, PRETTY_FUNC, STREAMSTR(Msg) ); }while(0)

//static inline bool icompare_pred(unsigned char a, unsigned char b)
//{ return std::tolower(a) == std::tolower(b); }
// /** case insensitive string compare -- BUGGY do not use -- */
//static inline bool icompare(std::string const& a, std::string const& b)
//{
//    return std::lexicographical_compare(a.begin(), a.end(),
//            b.begin(), b.end(), icompare_pred);
//}
// --------- try another version ...
inline bool caseInsCharCompare(char a, char b) {
   return(toupper(a) == toupper(b));
}
/** \return true iff s1 and s2 are identical when converted \c toupper */
inline bool icompare(const std::string& s1, const std::string& s2) {
   return((s1.size() == s2.size()) &&
          std::equal(s1.begin(), s1.end(), s2.begin(), caseInsCharCompare));
}

typedef std::string ccstr;
typedef const void* cvptr;
typedef const char* ccptr;

#define OBJNAME(otp) demangle(typeid(otp).name()).c_str()
#if 0 // should not be in header anyway
static string mdemangle( ccptr name )
{
    int  ok;
    char* full = abi::__cxa_demangle(name, 0, 0, &ok);
    if ( full == 0 ) return string( name );
    string sym( full );
    ::free(full);
    for ( int i = sym.size(); --i >= 0; ) {
        if ( sym[i] == ' ' ) {
            if ( i > 0 && !isalnum(sym[i-1]) ) sym.erase( i, 1 );
            else sym[i] = '.';
        }
    }
    return sym;
}
#endif

#define scr_TRY( ERrmsg ) char const* const errmsg = ERrmsg; scr_CNT; try

/** standard closing brace, with ERR_LBL: for goto's */
#define scr_CATCH catch(std::exception& e){ \
    std::cout<<" exception: "<<e.what(); \
    Derr_msg( false, true, e.what() ); \
    goto ERR_LBL; \
 } \
scr_ERR( errmsg )

/** abbreviated lua stack check */
#define scr_CHK scr_STK(errmsg)

#define  CastP(x,T,y) /* */ scr_dPoint_T<T>* x = dynamic_cast< /* */ scr_dPoint_T<T>* >( y )
#define cCastP(x,T,y) const scr_dPoint_T<T>* x = dynamic_cast< const scr_dPoint_T<T>* >( y )

#define scr_CNTn(n) int __stk_cnt = n
#define scr_CNT     scr_CNTn(1)

#define scr_BOOLn(  n,var,label) bool   var; if ( ! d_si->try_bool (n, var ) ) goto label
#define scr_IDXn(   n,var,label) sidx   var; if ( ! d_si->try_sidx (n, var ) ) goto label
#define scr_INTn(   n,var,label) Sint4  var; if ( ! d_si->try_Sint4(n, var ) ) goto label
#define scr_UINTn(  n,var,label) Uint4  var; if ( ! d_si->try_Uint4(n, var ) ) goto label
#define scr_REALn(  n,var,label) real8  var; if ( ! d_si->try_real8(n, var ) ) goto label
#define scr_STRn(   n,var,label) ccptr  var; if ( ! d_si->try_ccptr(n, var ) ) goto label
#define scr_XSTRn(  n,var,label) ccstr  var; if ( ! d_si->try_ccstr(n, var ) ) goto label
#define scr_FILEPn( n,var,label) FILE*  var; if ( ! d_si->try_FILEP(n, var ) ) goto label
#define scr_REFIDn( n,var,label) int    var; if ( ! d_si->try_refid(n, var ) ) goto label
#define scr_ARGSn(  n,var,label) Args   var; if ( ! d_si->try_stack(n, var ) ) goto label
#define scr_ARGMAPn(n,var,label) ArgMap var; if ( ! d_si->try_stack(n, var ) ) goto label
#define scr_STRSn(  n,var,label) script_Interpreter::StrSet var; if ( ! d_si->try_stack(n, var ) ) goto label
#define scr_USRn( T,n,var,label) T*   var = (T*)d_si->try_user(n, OBJNAME(T) ); printf ("%p %s %p\n", d_si, OBJNAME(T), var);if ( var == 0 ) goto label
#define scr_UDATAn( n,var,label) void*  var = d_si->try_user(n, nullptr ); if ( var == 0 ) goto label
#define scr_CHRSETn(n,str,var,label) char var; if ( ! d_si->try_chrset( n, str, var ) ) goto label

#define scr_STKn(   n, msg ) Derr_msg( d_si->cnt_stack() == (n), true, "parameter:" << d_si->cnt_stack() << " != " << n << ", " << msg )

#define scr_BOOL(  var,label) scr_BOOLn(  __stk_cnt, var, label ); ++__stk_cnt
#define scr_IDX(   var,label) scr_IDXn (  __stk_cnt, var, label ); ++__stk_cnt
#define scr_INT(   var,label) scr_INTn (  __stk_cnt, var, label ); ++__stk_cnt
#define scr_UINT(  var,label) scr_UINTn(  __stk_cnt, var, label ); ++__stk_cnt
#define scr_REAL(  var,label) scr_REALn(  __stk_cnt, var, label ); ++__stk_cnt
#define scr_STR(   var,label) scr_STRn(   __stk_cnt, var, label ); ++__stk_cnt
#define scr_XSTR(  var,label) scr_XSTRn(  __stk_cnt, var, label ); ++__stk_cnt
#define scr_FILEP( var,label) scr_FILEPn( __stk_cnt, var, label ); ++__stk_cnt
#define scr_REFID( var,label) scr_REFIDn( __stk_cnt, var, label ); ++__stk_cnt
#define scr_ARGS(  var,label) scr_ARGSn(  __stk_cnt, var, label ); ++__stk_cnt
#define scr_ARGMAP(var,label) scr_ARGMAPn(__stk_cnt, var, label ); ++__stk_cnt
#define scr_STRS(  var,label) scr_STRSn(  __stk_cnt, var, label ); ++__stk_cnt
#define scr_USR( T,var,label) scr_USRn( T,__stk_cnt, var, label ); ++__stk_cnt
#define scr_UDATA( var,label) scr_UDATAn( __stk_cnt, var, label ); ++__stk_cnt
#define scr_CHRSET(str,var,label) scr_CHRSETn(__stk_cnt, str, var, label ); ++__stk_cnt

 // tries to get a uidx from the stack (silently converts num_* to uidx when possible)
#define scr_UIDXn( n, var, label ) uidx var; if ( !scr_Num::cast_stack_to_uidx( n, var ) ) goto label
#define scr_UIDX( var, label ) scr_UIDXn( __stk_cnt, var, label ); ++__stk_cnt

#define scr_STK( msg ) scr_STKn( __stk_cnt-1, msg )

#define scr_ERR( msg ) ERR_LBL: Derr_msg( false, true, "parameter:" << __stk_cnt << ", " << msg ); return 0
#endif // MTYPES_HPP
