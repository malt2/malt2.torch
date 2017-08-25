require 'torch'
print(" Good.  malt2 init.lua was able to require 'torch'")

if false then
  -- this will work if I do the RTLD_NOLOAD|RTLD_GLOBAL trick **inside** luaopen_malt2
  --print(" attempting to require 'malt2'")
  require('malt2')
else
  -- NEXT line should be set during cmake with the full absolute path of torch libraries
  --      - preload libmpi into global space too?
  --      - perhaps even before require 'torch' ?
  --local path = "/local/kruus/torch/install/lib/lua/5.1/libmalt2.so"

  -- package.preload is a table of functions associated with module names.
  -- Before searching the filesystem, require checks if package.preload has a
  -- matching key. If so, the function is called and its result used as the
  -- return value of require.
  -- print("malt2 init.lua ...")

  package.preload["malt2"] = function ()
    local malt2_libname = package.searchpath("libmalt2", package.cpath)   -- absolute path to libmalt2.so
    local malt2_luaname = package.searchpath("malt2", package.cpath)      -- absolute path to malt2/init.lua
    print("malt2_libname = "..tostring(malt2_libname))
    print("malt2_luaname = "..tostring(malt2_luaname))
    assert(malt2_libname)
    -- not in luajit, just returns true/false lib_A = assert(package.loadlib(path, "*")) --> load global
    assert(package.loadlib(malt2_libname, "*"))  --> load library with RTLD_GLOBAL
    local luaopen_function = package.loadlib(malt2_libname, "luaopen_malt2")
    print("luaopen_function = "..tostring(luaopen_function))
    local ret = luaopen_function()   -- malt2 luaopen function takes no extra args
    assert(ret)
  end

  if false then
    -- Oh. We are executing INSIDE an existing    require "malt2"   lua command!
    -- print(" attempting to require 'malt2'")
    require("malt2")                   -- luajit error
  else
    -- but can directly invoke it the package.preload function ...
    print(" attempting to load malt2 with RTLD_GLOBAL flags")
    package.preload["malt2"] ()        -- invoke the load-with-RTLD_GLOBAL method
    print("                     ____  \n" .. 
         "    ____ ___  ____ _/ / /_ \n" .. 
         "   / __  __  / __  / / __/ \n" ..
         "  / / / / / / /_/ / / /_   \n" ..
         " /_/ /_/ /_/\\____/_/\\__/   \n") 
  end

end
-- If successful, you can:
print("dstorm object\n"..tostring(dstorm))

require('pl')

dump_lua_G = function()
  -- dump lua globals to file lua_G.txt
  local gg=pretty.write(_G,"  ")
  local fgg=io.open("lua_G.txt","w")
  fgg:write(gg)
  fgg:close()
end
dump_lua_G()
-- is there a way to find libmpi path and ensure it gets loaded globally BEFORE dstorm is initialized ?
-- this would be done *before* invoking luaopen_malt2

-- local malt2 = {}
-- malt2.dstorm = require 'malt2.dstorm'

-- return malt2
