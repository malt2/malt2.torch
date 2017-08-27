
CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
IF(LUAROCKS_PREFIX)
    MESSAGE(STATUS "Installing Torch through Luarocks")
    STRING(REGEX REPLACE "(.*)lib/luarocks/rocks.*" "\\1" CMAKE_INSTALL_PREFIX  "${LUAROCKS_PREFIX}")
    MESSAGE(STATUS "Prefix inferred from Luarocks: ${CMAKE_INSTALL_PREFIX}")
ENDIF()

INCLUDE(../dStorm-env.cmake)
MESSAGE(STATUS "dStorm-env.cmake: LIBDSTORM_FLAGS = ${LIBDSTORM_FLAGS}")
# this adds flags to both C/C++ compiles (could also set CMAKE_CXX_FLAGS and CMAKE_EXE_LINKER_FLAGS separately)
add_definitions(${LIBDSTORM_FLAGS})

FIND_PACKAGE(Torch REQUIRED)
MESSAGE(STATUS "Torch_INSTALL_INCLUDE : ${Torch_INSTALL_INCLUDE}")
MESSAGE(STATUS "Torch_INSTALL_LIB     : ${Torch_INSTALL_LIB}")
# cutorch package provides a library for cuda Tensor functions (for our .cu compilations)
find_library (TORCH_THC THC ${Torch_INSTALL_LIB} NO_DEFAULT_PATH)
MESSAGE(STATUS "TORCH_THC      : ${TORCH_THC}")

SET(BOOST_ROOT /opt/boost)
FIND_PACKAGE(Boost 1.54.0 REQUIRED)
MESSAGE(STATUS "boost: ${Boost_INCLUDEDIR}")
include_directories(${Boost_INCLUDE_DIRS})

FIND_PACKAGE(MPI)
include_directories(${MPI_INCLUDE_PATH})
FIND_PACKAGE(CUDA)
# is FindCUDA from torch? why 2?
INCLUDE(FindCUDA)
MESSAGE(STATUS "CUDA_SDK_SEARCH_PATH  ${CUDA_SDK_SEARCH_PATH}")
MESSAGE(STATUS "CUDA_SDK_ROOT_DIR     ${CUDA_SDK_ROOT_DIR}")
MESSAGE(STATUS "CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR}")
find_path(CUDA_COMMON_INCLUDE_DIR
  helper_cuda.h
  PATHS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES "samples/common/inc"
  DOC "Location of helper_cuda.h"
  NO_DEFAULT_PATH
  )
# Now search system paths
find_path(CUDA_COMMON_INCLUDE_DIR cutil.h DOC "Location of helper_cuda.h")
mark_as_advanced(CUDA_COMMON_INCLUDE_DIR)
include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_COMMON_INCLUDE_DIR})
MESSAGE(STATUS "CUDA_LIBRARIES           : ${CUDA_LIBRARIES}")
#
# --- TODO ---
# if build env is WITH_MPI or WITH_GPU, g++ --> mpic++
# ... and if WITH_GPU also use mpic++ as nvcc's host compiler
# and nvcc should NOT have the -frtti option
#


SET(DSTORM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../dstorm")
SET(src_DIR "${DSTORM_DIR}/..")
SET(ORM_DIR "${src_DIR}/orm")
include_directories(${DSTORM_DIR}/include)
include_directories(${ORM_DIR}/include)
SET(CMAKE_CXX_FLAGS "-std=c++11 -fopenmp -frtti -march=native") 
FILE(GLOB LDSTORM  "${DSTORM_DIR}/lib64/libdstorm2-pic.a")
FILE(GLOB LIBVERBS "/usr/lib/libibverbs.so")
FILE(GLOB LORM     "${ORM_DIR}/liborm-pic.a")
INCLUDE_DIRECTORIES(BEFORE "${CMAKE_CURRENT_BINARY_DIR}/lib/THC")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/lib/THC")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/torch")

SET(CMAKE_EXE_COMPILE_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS}") 

SET(src ${src} mtypes.cpp scr_dstorm.cpp malt2.cpp scr_dstorm.cu)
FILE(GLOB luasrc *.lua)
SET(luasrc ${luasrc} init.lua test.lua)
MESSAGE(STATUS "src   : ${src}")
MESSAGE(STATUS "luasrc: ${luasrc}")
MESSAGE(STATUS "LINK malt2: DSTORM: ${LDSTORM} ORM: ${LORM} ")
MESSAGE(STATUS "LINK MPI: MPI_C_COMPILER:     ${MPI_C_COMPILER}")
MESSAGE(STATUS "LINK MPI: MPI_C_FLAGS:        ${MPI_C_FLAGS}")
MESSAGE(STATUS "LINK MPI: MPI_C_INCLUDE_PATH: ${MPI_C_INCLUDE_PATH}")
MESSAGE(STATUS "LINK MPI: MPI_C_LINK_FLAGS:   ${MPI_C_LINK_FLAGS}")
MESSAGE(STATUS "LINK MPI: MPI_C_LIBRARIES:    ${MPI_C_LIBRARIES}")
MESSAGE(STATUS "LINK MPI: MPI_CXX_COMPILER:   ${MPI_CXX_COMPILER}")
MESSAGE(STATUS "LINK MPI: MPI_CXX_FLAGS:        ${MPI_CXX_FLAGS}")
MESSAGE(STATUS "LINK MPI: MPI_CXX_INCLUDE_PATH: ${MPI_CXX_INCLUDE_PATH}")
MESSAGE(STATUS "LINK MPI: MPI_CXX_LINK_FLAGS:   ${MPI_CXX_LINK_FLAGS}")
MESSAGE(STATUS "LINK MPI: MPI_CXX_LIBRARIES:    ${MPI_CXX_LIBRARIES}")
MESSAGE(STATUS "LINK MPI: MPI_LINK_FLAGS: ${MPI_LINK_FLAGS}")
MESSAGE(STATUS "LINK MPI: MPI_EXTRA_LIBRARY: ${MPI_EXTRA_LIBRARY}")
MESSAGE(STATUS "LINK MPI: MPIEXEC: ${MPIEXEC}")
MESSAGE(STATUS "LINK MPI: MPI_NUMPROC_FLAG: ${MPI_NUMPROC_FLAG}")
# see https://cmake.org/cmake/help/v3.0/module/FindMPI.html
MESSAGE(STATUS "MPI_PREFIX:     ${MPI_PREFIX}")	 # unset
SET(CMAKE_C_COMPILER ${MPI_C_COMPILER})
SET(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

list(GET MPI_C_LIBRARIES 0 FIRST_MPI_LIBRARY)
MESSAGE(STATUS "FIRST_MPI_LIBRARY ${FIRST_MPI_LIBRARY}")
GET_FILENAME_COMPONENT(MPI_LIBDIR "${FIRST_MPI_LIBRARY}" PATH)
MESSAGE(STATUS "MPI_LIBDIR ${MPI_LIBDIR}")

# adapted from mpiT rockspec...
message("MPI_C_FOUND = ${MPI_C_FOUND}")
IF (APPLE=="TRUE")
        message ("APPLE")
	FILE(GLOB LMPI ${MPI_LIBDIR}/*.dylib)
ELSE()
	# ? IF(CMAKE_SYSTEM_NAME STREQUAL Linux)
	IF("${MPI_C_FOUND}" STREQUAL "TRUE")
		message ("LINK ${PLATFORM_ID} ** assuming ** openmpi -- Change if MPICH is your preference")
		# actually liboshmem.so seems *only* to be available on linux OpenMPI
		SET(LMPI "${MPI_LIBDIR}/liboshmem.so")
		FILE(GLOB LMCA "${MPI_LIBDIR}/openmpi/*.so")
        ELSE()
		FILE(GLOB LMPI ${MPI_LIBDIR}/*.so)
        ENDIF()
ENDIF()

#message ("LMPI is ${LMPI}")
message ("LMCA is ${LMCA}")

ADD_TORCH_PACKAGE(malt2 "${src}" "${luasrc}" "MALT-2 dstorm package")

# MPI note: -lopen-pal hint was from /usr/lib/openmpi/lib/pkgconfig/opal.pc  (pkgconfig file)
#SET(OMPI_EXTRA "/usr/lib/openmpi/lib/libopen-pal.so /usr/lib/openmpi/lib/libopen-rte.so")
#  Above did not fix lua + MPI issues (trying MPI recompile with --disable-dlopen)
SET(OMPI_EXTRA)
TARGET_LINK_LIBRARIES(malt2 luaT TH ibverbs ${LDSTORM} ${LORM} ${TORCH_THC} ${MPI_CXX_LIBRARIES} ${MPI_CXX_LIBRARIES} ${LMPI} ${LMCA} ${OMPI_EXTRA} ${LDSTORM} ${LIBVERBS} ${CUDA_LIBRARIES})
#TARGET_LINK_LIBRARIES(malt2 luaT TH ibverbs ${LDSTORM} ${LORM} ${MPI_CXX_LIBRARIES} ${MPI_CXX_LIBRARIES} /usr/lib/openmpi/lib/libopen-pal.so /usr/lib/openmpi/lib/libopen-rte.so ${LIBVERBS})
