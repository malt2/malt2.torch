/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */

#include "mtypes.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>

int malt2_err_fun( bool Abort, const int Line, const std::string& File, const std::string& Func, const std::string& Msg )
{
    //std::cout << Msg << File << Line << Func << Abort << std::endl;
    std::ostringstream oss;
    oss << (Abort? "Error: ": "Warning: ") << Msg << File << ":" << Line << " " << Func << std::endl;
    std::cout<<oss.str();
    std::cout.flush();
    if( 1 && Abort ){
        throw std::runtime_error( oss.str() );
    }
    return 0;
}
