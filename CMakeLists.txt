cmake_minimum_required(VERSION 3.15...3.30)
project(pymineflow LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# One extension module: the bindings, the OES kernels and the MineFlow solver. NumPy is accessed
# through pybind11's runtime API (py::array_t), so the wheel does not depend on the NumPy ABI it was
# built against.
pybind11_add_module(_mineflow
  src/pymineflow/_mineflow.cpp
  src/pymineflow/_oes.cpp
  third_party/mineflow/mineflow.cpp)
target_include_directories(_mineflow PRIVATE third_party/mineflow)

# Portable optimization only: no -march=native and no -ffast-math (results must be reproducible).
if(MSVC)
  target_compile_options(_mineflow PRIVATE /O2 /EHsc)
  # std::mutex without a constexpr constructor: otherwise the module crashes on import when another
  # package (e.g. an older pyarrow) has already loaded an older msvcp140.dll into the process.
  target_compile_definitions(_mineflow PRIVATE _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
else()
  target_compile_options(_mineflow PRIVATE -O3)
endif()

install(TARGETS _mineflow LIBRARY DESTINATION pymineflow)
