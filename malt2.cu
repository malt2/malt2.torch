/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#include <luaT.h>
#include <lua_malt2.cuh>

#include <string>
#include <exception>
#include <iostream>

extern "C" DLL_EXPORT int luaopen_libmalt2(lua_State *L) {
    luaopen_malt2(L);
    return 1;
}
