# -----------------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2015, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# -----------------------------------------------------------------------------

cmake_minimum_required(VERSION 2.8)
project(nupic_core CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_VERBOSE_MAKEFILE OFF) # toggle for cmake debug


if( POLICY CMP0046 )
  cmake_policy(VERSION 2.8)
  # The new policy for add_dependencies is to now error if a dependency target
  # is not found. The old policy didn't care and continued through configuration
  cmake_policy(SET CMP0046 OLD)
endif()

if( POLICY CMP0054 )
  # The OLD behavior for this policy is to dereference variables and interpret
  # keywords even if they are quoted or bracketed. The NEW behavior is to not
  # dereference variables or interpret keywords that have been quoted or bracketed.
  cmake_policy(SET CMP0054 OLD)
endif()


#
# NuPIC CMake options
#
option(NUPIC_IWYU "Enable include-what-you-use
  (http://include-what-you-use.org/). This requires the iwyu binary to be
  discoverable by CMake's find_program.")
if(${NUPIC_IWYU})
  find_program(iwyu_path NAMES include-what-you-use iwyu)
  if(NOT iwyu_path)
    message(FATAL_ERROR "Could not find the program include-what-you-use")
  endif()
endif()


#
# Set up compile flags for internal sources and for swig-generated sources
#

set(src_compile_flags "${INTERNAL_CXX_FLAGS_OPTIMIZED}")
set(src_swig_generated_file_compile_flags "${EXTERNAL_CXX_FLAGS_OPTIMIZED}")

if(MINGW)
  # This is for GCC 4.8.x
  # http://stackoverflow.com/questions/10660524/error-building-boost-1-49-0-with-gcc-4-7-0
  set(src_compile_flags "${src_compile_flags} -include cmath")
  set(src_swig_generated_file_compile_flags
      "${src_swig_generated_file_compile_flags} -include cmath")
endif()


#
# Compiler definitions specific to nupic.core code
#

string(TOUPPER ${PLATFORM} platform_uppercase)

set(src_compiler_definitions
    ${COMMON_COMPILER_DEFINITIONS}
    ${CAPNP_COMPILER_DEFINITIONS}
    -DNTA_OS_${platform_uppercase}
    -DNTA_ARCH_${BITNESS}
    -DHAVE_CONFIG_H
    -DNTA_INTERNAL
    -DBOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    -DBOOST_NO_WREGEX
    -DNUPIC2
    -DAPR_DECLARE_STATIC
    -DAPU_DECLARE_STATIC)

if(NOT "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7l")
  set(src_compiler_definitions
      ${src_compiler_definitions}
      -DNTA_ASM)
endif()

if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  set(src_compiler_definitions
      ${src_compiler_definitions}
      -DNTA_ASSERTIONS_ON)
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
  set(src_compiler_definitions
      ${src_compiler_definitions}
      -DNTA_COMPILER_GNU)
elseif(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
  set(src_compiler_definitions
      ${src_compiler_definitions}
      -DNTA_COMPILER_CLANG)
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
  set(src_compiler_definitions
      ${src_compiler_definitions}
      -DNTA_COMPILER_MSVC)
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "MinGW")
  set(src_compiler_definitions
      ${src_compiler_definitions}
      -DNTA_COMPILER_GNU
      -D_hypot=hypot)
endif()


add_definitions(${src_compiler_definitions})


#
# Find Python Interp, min. v2.7
# And NumPy core directory
#
find_package(PythonInterp 2.7 REQUIRED)
message(STATUS "CMAKE Found python interpreter ${PYTHON_EXECUTABLE} version=${PYTHON_VERSION_STRING}")
execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
                        "import numpy; import os; import sys; sys.stdout.write(os.path.dirname(numpy.get_include()))"
                         OUTPUT_VARIABLE src_numpy_core)

# Quick way to fixup directory paths to NumPy
get_filename_component(src_numpy_core ${src_numpy_core}/include/.. ABSOLUTE)

message(STATUS "src_numpy_core   = ${src_numpy_core}")


#
# Setup include paths
#
if("${PROJECT_BUILD_ARTIFACTS_DIR}" STREQUAL "")
  set(PROJECT_BUILD_ARTIFACTS_DIR "${PROJECT_BINARY_DIR}/artifacts")
endif()

include_directories(SYSTEM
                    ${src_numpy_core}/include
                    ${PROJECT_BINARY_DIR})

include_directories(${PROJECT_SOURCE_DIR})

#
# Let CMake know where all of the external files are.
#
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}
                      ${REPOSITORY_DIR}/external/common/
                      ${REPOSITORY_DIR}/external/${PLATFORM}${BITNESS}${PLATFORM_SUFFIX}/
                      ${src_numpy_core}/)

message(STATUS "CMAKE_PREFIX_PATH   = ${CMAKE_PREFIX_PATH}")

#
# Extract current git sha and record in nupic/Version.hpp
#
execute_process(COMMAND git rev-parse HEAD
                OUTPUT_VARIABLE NUPIC_CORE_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE)
configure_file(
  "${PROJECT_SOURCE_DIR}/nupic/Version.hpp.in"
  "${PROJECT_BINARY_DIR}/Version.hpp")

#
# Setup gtest
#
set(src_lib_static_gtest gtest)
add_library(${src_lib_static_gtest}
            STATIC
            ${REPOSITORY_DIR}/external/common/src/gtest/gtest-all.cpp)
set_target_properties(${src_lib_static_gtest}
                      PROPERTIES COMPILE_FLAGS "${src_compile_flags}"
                      LINK_FLAGS "${INTERNAL_LINKER_FLAGS_OPTIMIZED}")

#
# Locate Python artifacts
#
find_package(PythonLibs REQUIRED)
include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS})

message(STATUS "PYTHON_EXECUTABLE   = ${PYTHON_EXECUTABLE}")
message(STATUS "PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
message(STATUS "PYTHON_LIBRARIES    = ${PYTHON_LIBRARIES}")


#
# Python support
#
set(src_py_support_files
    nupic/py_support/PyHelpers.cpp
    nupic/py_support/PyCapnp.cpp
)

# List all .capnp files here. The C++ files will be generated and included
# when compiling later on.
set(src_capnp_specs_rel
  nupic/proto/NetworkProto.capnp
  nupic/proto/PyRegionProto.capnp
  nupic/proto/RandomProto.capnp
  nupic/proto/RegionProto.capnp
  nupic/proto/TemporalMemoryProto.capnp
)

# Create custom command for generating C++ code from .capnp schema files.

foreach(spec ${src_capnp_specs_rel})
  list(APPEND src_capnp_specs ${PROJECT_SOURCE_DIR}/${spec})
  list(APPEND src_capnp_generated_hdrs ${PROJECT_BINARY_DIR}/${spec}.h)
  list(APPEND src_capnp_generated_srcs ${PROJECT_BINARY_DIR}/${spec}.c++)
endforeach(spec)
set(src_capnp_all_spec_hdrs_and_srcs
    ${src_capnp_generated_hdrs}
    ${src_capnp_generated_srcs})

create_capnpc_command("${src_capnp_specs}"
                      ${PROJECT_SOURCE_DIR}
                      ${PROJECT_SOURCE_DIR}
                      ${PROJECT_BINARY_DIR}
                      "${src_capnp_all_spec_hdrs_and_srcs}")


message(STATUS "CAPNP_EXECUTABLE          = ${CAPNP_EXECUTABLE}")
message(STATUS "CAPNP_CMAKE_DEFINITIONS   = ${CAPNP_CMAKE_DEFINITIONS}")
message(STATUS "CAPNP_INCLUDE_DIRS        = ${CAPNP_INCLUDE_DIRS}")
message(STATUS "CAPNP_STATIC_LIB_TARGET   = ${CAPNP_STATIC_LIB_TARGET}")


#
# Common system libraries for shared libraries and executables
#
set(src_common_os_libs)

if("${PLATFORM}" STREQUAL "linux")
  list(APPEND src_common_os_libs pthread dl)
elseif("${PLATFORM}" STREQUAL "darwin")
  list(APPEND src_common_os_libs c++abi)
elseif(MSYS OR MINGW)
  list(APPEND src_common_os_libs psapi ws2_32 wsock32 rpcrt4)
elseif("${PLATFORM}" STREQUAL "windows")
  list(APPEND src_common_os_libs oldnames.lib psapi.lib ws2_32.lib)
endif()

message(STATUS "src_common_os_libs        = ${src_common_os_libs}")


#
# Setup nupic_core_solo static library, consisting of our own sources;
#
# this is an intermediate archive that will be merged with external static
# libraries in a subsequent step to produce the nupic_core "combined" static
# library.
#
set(src_lib_static_nupiccore_solo nupic_core_solo)

set(src_nupiccore_srcs
    nupic/engine/Network.cpp
    nupic/types/BasicType.cpp
    nupic/utils/LoggingException.cpp
    nupic/utils/LogItem.cpp
    nupic/utils/Random.cpp)

set(src_lib_static_nupiccore_srcs
    ${src_capnp_generated_srcs}
    ${src_py_support_files}
    ${src_nupiccore_srcs})

set(src_lib_static_nupiccore_compile_flags
    "${src_compile_flags} -I${src_numpy_core}/include")

message(STATUS "src_compile_flags = ${src_compile_flags}")
message(STATUS "src_lib_static_nupiccore_compile_flags = ${src_lib_static_nupiccore_compile_flags}")

add_library(${src_lib_static_nupiccore_solo} STATIC
            ${src_lib_static_nupiccore_srcs})
# nupic_core_solo sources depend on headers installed by these external projects
#add_dependencies(${src_lib_static_nupiccore_solo}
#                 ${YAML_CPP_STATIC_LIB_TARGET}
#                 ${YAML_STATIC_LIB_TARGET}
#                 ${APR1_STATIC_LIB_TARGET}
#                 ${APRUTIL1_STATIC_LIB_TARGET}
#                 ${Z_STATIC_LIB_TARGET})
set_target_properties(${src_lib_static_nupiccore_solo} PROPERTIES COMPILE_FLAGS
                      ${src_lib_static_nupiccore_compile_flags})
if(${NUPIC_IWYU})
  # TODO: Create a target that doesn't include the generated capnp schema files
  # since we don't want to run iwyu on them and iwyu can only be applied to
  # CMake targets (not source files directly).
  set_target_properties(${src_lib_static_nupiccore_solo}
    PROPERTIES CXX_INCLUDE_WHAT_YOU_USE ${iwyu_path})
endif()


#
# Create the nupic_core "combined" static library by merging nupic_core_solo
# with our external static libraries
#

include(src/NupicLibraryUtils) # for MERGE_STATIC_LIBRARIES

set(src_lib_static_nupiccore_combined nupic_core)

set(src_external_static_libs
    ${CAPNP_STATIC_LIB_TARGET}
)

#set(src_external_static_libs
#    ${YAML_CPP_STATIC_LIB_TARGET}
#    ${YAML_STATIC_LIB_TARGET}
#    ${APR1_STATIC_LIB_TARGET}
#    ${APRUTIL1_STATIC_LIB_TARGET}
#    ${Z_STATIC_LIB_TARGET})


set(src_combined_nupiccore_source_archives
    ${src_lib_static_nupiccore_solo}
    ${src_external_static_libs})

# Create a top-level library target for the combined static lib
merge_static_libraries(${src_lib_static_nupiccore_combined}
                       "${src_combined_nupiccore_source_archives}")


#
# Build tests of the nupic_core "combined" static library
#

# Common libs for test executables
set(src_common_test_exe_libs
    ${src_lib_static_nupiccore_combined}
    ${PYTHON_LIBRARIES}
    ${src_common_os_libs})


message(STATUS "src_common_test_exe_libs = ${src_common_test_exe_libs}")


#
# Setup gtests
#
set(src_executable_gtests unit_tests)
add_executable(${src_executable_gtests}
               test/unit/UnitTestMain.cpp
               test/unit/utils/RandomTest.cpp)
target_link_libraries(${src_executable_gtests}
                      ${src_lib_static_gtest}
                      ${src_common_test_exe_libs})
set_target_properties(${src_executable_gtests}
                      PROPERTIES COMPILE_FLAGS ${src_compile_flags}
                                 LINK_FLAGS "${INTERNAL_LINKER_FLAGS_OPTIMIZED}")
add_custom_target(tests_unit
                  COMMAND ${src_executable_gtests}
                  DEPENDS ${src_executable_gtests}
                  COMMENT "Executing test ${src_executable_gtests}"
                  VERBATIM)


#
# Use SWIG to generate Python extensions.
#
if (NUPIC_BUILD_PYEXT_MODULES)
  include(UseSWIG)

  # Set the output location for the language modules that are created.
  set(CMAKE_SWIG_OUTDIR ${PROJECT_BINARY_DIR})

  # Make sure the directory exists for the generated C++ files.
  file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/nupic/bindings)

  # TODO ZZZ set COMPILE_FLAGS on swig targets

  # SWIG options from:
  # https://github.com/swig/swig/blob/master/Source/Modules/python.cxx#L111
  set(src_swig_flags
      -c++
      -features
      autodoc=0,directors=0
      -noproxyimport
      -keyword
      -modern
      -modernargs
      -noproxydel
      -fvirtual
      -fastunpack
      -nofastproxy
      -fastquery
      -outputtuple
      -castmode
      -nosafecstrings
      -w402
      -w503
      -w511
      -w302
      -w362
      -w312
      -w389
      -DSWIG_PYTHON_LEGACY_BOOL
      -I${SWIG_DIR}/python
      -I${SWIG_DIR}
      ${src_compiler_definitions}
  )

  file(GLOB_RECURSE swig_header_deps *.i *.hpp *.h)

  message(STATUS "src_swig_flags = ${src_swig_flags}")

  # Tell swig which command-line options to use, allowing user to override
  set(CMAKE_SWIG_FLAGS ${src_swig_flags} ${CMAKE_SWIG_FLAGS})

  # Set up linker flags for python extension shared libraries
  set(_SRC_SWIG_EXTENSION_LINK_FLAGS "${PYEXT_LINKER_FLAGS_OPTIMIZED}")

  # NOTE Non-Windows Python extensions shouldn't be linking agains libpython;
  # symbols should be available automatically when python loads the extension.
  #
  # NOTE We don't link our non-Windows python extensions with capnproto static
  # lib in order to force runtime linkage to pycapnp's capnproto symbols thus
  # avoiding the conflict of executing the methods in our own
  # compilation of capnproto (different compiler/STL/version, flags, etc.) on
  # objects created by pycapnp's. We force runtime linking to pycapnp's
  # capnproto symbols by preloading the pycapnp extension in our extension
  # python proxy modules, thus avoiding the conflict from executing the methods
  # in our own compilation of capnproto on objects created by pycapnp's.
  set(_SRC_SWIG_LINK_LIBRARIES
      ${src_lib_static_nupiccore_combined}
      ${src_common_os_libs})

  # Common dependencies for our python extensions for use with
  # SWIG_MODULE_name_EXTRA_DEPS
  # Make sure we don't execute the swig executable before it is built
  set(_SRC_SWIG_EXTRA_DEPS Swig)

  # NOTE Windows DLLs are shared executables with their own main; they require
  # all symbols to resolve at link time, so we have to add libpython for this
  # platform
  #
  # NOTE On Windows nupic.bindings builds we presently build self-contained
  # CAPNP_LITE on Windows, and Windows nupic/nupic.bindings presently have no
  # dependency on pycapnp.
  if("${PLATFORM}" STREQUAL "windows")
    list(APPEND _SRC_SWIG_LINK_LIBRARIES
         ${PYTHON_LIBRARIES})
  endif()

  message(STATUS "_SRC_SWIG_EXTRA_DEPS     = ${_SRC_SWIG_EXTRA_DEPS}")
  message(STATUS "_SRC_SWIG_LINK_LIBRARIES = ${_SRC_SWIG_LINK_LIBRARIES}")
  message(STATUS "_SRC_SWIG_EXTENSION_LINK_FLAGS= ${_SRC_SWIG_EXTENSION_LINK_FLAGS}")
  message(STATUS "CMAKE_SWIG_FLAGS        = ${CMAKE_SWIG_FLAGS}")


  function(BUILD_EXTENSION MODULE_NAME)
    # Create a nupic.bindings swig extension target with the given Swig module
    # name. Also, if PY_EXTENSIONS_DIR is specified, request
    # installation of the extension library and python proxy module.
    #
    # The real target name is ${SWIG_MODULE_${MODULE_NAME}_REAL_NAME}.

    set(source_interface_file nupic/bindings/${MODULE_NAME}.i)

    set_source_files_properties(${source_interface_file} PROPERTIES
                                CPLUSPLUS ON
                                SWIG_MODULE_NAME ${MODULE_NAME})

    # Regenerate SWIG bindings if any headers change.
    set(SWIG_MODULE_${MODULE_NAME}_EXTRA_DEPS ${swig_header_deps})

    #
    # Create custom command for generating files from SWIG
    #

    # Note: swig_add_module outputs ${swig_generated_file_fullname}
    swig_add_module(${MODULE_NAME} python ${source_interface_file})

    set_source_files_properties(
      ${swig_generated_file_fullname} PROPERTIES
      GENERATED TRUE
      COMPILE_FLAGS ${src_swig_generated_file_compile_flags})

    swig_link_libraries(${MODULE_NAME} ${_SRC_SWIG_LINK_LIBRARIES})

    set(real_target "${SWIG_MODULE_${MODULE_NAME}_REAL_NAME}")

    set(extra_deps ${_SRC_SWIG_EXTRA_DEPS})
    set(link_flags ${_SRC_SWIG_EXTENSION_LINK_FLAGS})

    # Create an export map and update extra dependencies and link flags. This
    # export map prevents runtime-link preemption of statically-linked
    # libraries, such as -static-libstdc++, and limits the shared object's
    # symbol visibility to only the python extension's init function. NOTE Not
    # sure what, if anything, to do for MSVC at this time.
    set(extension_init_func "init${real_target}")

    if("${PLATFORM}" STREQUAL "darwin")
      set(link_flags
          "${link_flags} -Wl,-exported_symbol,_${extension_init_func}")

    elseif("${PLATFORM}" STREQUAL "linux" OR MINGW)
      set(export_map_file
          "${CMAKE_CURRENT_BINARY_DIR}/${real_target}_generated.expmap")

      list(APPEND extra_deps "${export_map_file}")
      set(link_flags "${link_flags} -Wl,--version-script=${export_map_file}")

      set(export_map_contents "{global: ${extension_init_func}; local: *;};")

      message(STATUS "Writing export map file ${export_map_file} "
              "(${export_map_contents}).")

      file(WRITE ${export_map_file} "${export_map_contents}")
    endif()


    add_dependencies(${real_target} ${extra_deps})
    set_target_properties(${real_target} PROPERTIES
                          LINK_FLAGS "${link_flags}")

    # If a path is specified, copy extensions files to proper location.
    if (PY_EXTENSIONS_DIR)
      install(TARGETS
              ${real_target}
              LIBRARY DESTINATION ${PY_EXTENSIONS_DIR})
      install(FILES
              ${PROJECT_BINARY_DIR}/${MODULE_NAME}.py
              DESTINATION ${PY_EXTENSIONS_DIR})
    endif(PY_EXTENSIONS_DIR)

    message(
      STATUS
      "Created Swig target ${real_target} for swig module ${MODULE_NAME}. "
      "extra_deps=${extra_deps}, link_flags=${link_flags}")
  endfunction(BUILD_EXTENSION)


  # Engine Internal
  build_extension("engine_internal")

  # Math
  build_extension("math")

endif() # NUPIC_BUILD_PYEXT_MODULES


#
# Install targets into CMAKE_INSTALL_PREFIX
#
install(TARGETS
        ${src_lib_static_nupiccore_combined}
        ${src_lib_static_gtest}
        ${src_executable_gtests}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY nupic/ DESTINATION include/nupic
        FILES_MATCHING PATTERN "*.h*"
        PATTERN "*.hpp.in" EXCLUDE)
install(DIRECTORY nupic/ DESTINATION include/nupic
        FILES_MATCHING PATTERN "*.capnp")
install(DIRECTORY ${PROJECT_BINARY_DIR}/nupic/ DESTINATION include/nupic/
        FILES_MATCHING PATTERN "*.capnp.h")

install(DIRECTORY ${REPOSITORY_DIR}/external/common/include/gtest
        DESTINATION include/gtest
        FILES_MATCHING PATTERN "*.h*")

install(FILES ${PROJECT_BINARY_DIR}/Version.hpp
        DESTINATION include/nupic/)

install(DIRECTORY "${REPOSITORY_DIR}/external/common/include/"
        DESTINATION include)

install(DIRECTORY "${src_numpy_core}/lib/"
        DESTINATION lib)
install(DIRECTORY "${src_numpy_core}/include/"
        DESTINATION include)

foreach(directory ${CAPNP_INCLUDE_DIRS})
  install(DIRECTORY "${directory}/capnp"
          DESTINATION include)
  install(DIRECTORY "${directory}/kj"
          DESTINATION include)
endforeach()

install(DIRECTORY nupic/py_support DESTINATION include/nupic
        FILES_MATCHING PATTERN "*.c*")
install(DIRECTORY nupic DESTINATION include/nupic
        FILES_MATCHING PATTERN "*.py")

#
# `make package` results in
# nupic_core-${NUPIC_CORE_VERSION}-${PLATFORM}${BITNESS}${PLATFORM_SUFFIX}.tar.gz binary release
#

set(CPACK_GENERATOR "TGZ")
set(CPACK_PACKAGE_FILE_NAME "nupic_core-${NUPIC_CORE_VERSION}-${PLATFORM}${BITNESS}${PLATFORM_SUFFIX}")
include(CPack)
