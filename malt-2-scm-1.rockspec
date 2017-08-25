package = "malt-2"
version = "scm-1"

source = {
    url = "file:///net/mlfs01/export/users/asim/malt-2/torch/"
}

description = {
   summary = "malt2 for torch",
   detailed = [[
   	    malt2 torch package
   ]],
   homepage = "https://mlsvn.nec-labs.com/projects/mlsvn/browser/milde_pkg/milde_malt2"
}

dependencies = {
   "torch >= 7.0"
}

build = {
   type = "command",
   build_command = [[
   cmake -E make_directory build;
   cd build;
   cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && cd .. && $(MAKE)
   ]],
   install_command = "cd build && cd .. && $(MAKE) install"
}
