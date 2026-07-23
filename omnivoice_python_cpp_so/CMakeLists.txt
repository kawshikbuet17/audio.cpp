# This file is added from the main audio.cpp CMakeLists.txt.
# engine_runtime and ggml are therefore already available.

find_package(
    Python3
    REQUIRED
    COMPONENTS Interpreter Development.Module
)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(
    omnivoice_cpp
    binding.cpp
)

target_include_directories(
    omnivoice_cpp
    PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
)

target_link_libraries(
    omnivoice_cpp
    PRIVATE
    engine_runtime
    ggml
)

# Put the generated .so directly in this folder.
set_target_properties(
    omnivoice_cpp
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
