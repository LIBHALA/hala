find_package(Doxygen OPTIONAL_COMPONENTS dot)

if (NOT DOXYGEN_FOUND)
    message(WARNING "HALA could not find Doxygen, skipping documentation")
    return()
endif()

set(DOXYGEN_GENERATE_HTML       "YES")
set(DOXYGEN_DISABLE_INDEX       "NO")
set(DOXYGEN_GENERATE_TREEVIEW   "YES") # left-hand index
set(DOXYGEN_SORT_GROUP_NAMES    "NO")  # use the order in the source files (logical order)
set(DOXYGEN_SORT_BRIEF_DOCS     "NO")
set(DOXYGEN_SORT_MEMBER_DOCS    "NO")
set(DOXYGEN_USE_MATHJAX         "YES")
set(DOXYGEN_COLLABORATION_GRAPH "YES") # messy and not very informative
set(DOXYGEN_INCLUDE_GRAPH       "YES")
set(DOXYGEN_GRAPHICAL_HIERARCHY "YES")
set(DOXYGEN_CLASS_GRAPH         "NO")
set(DOXYGEN_CLASS_DIAGRAMS      "NO")
set(DOXYGEN_MACRO_EXPANSION     "YES")
set(DOXYGEN_EXPAND_ONLY_PREDEF  "YES")
set(DOXYGEN_EXPAND_AS_DEFINED   "REPEAT_N_TIMES")

list(APPEND DOXYGEN_PREDEFINED "__HALA_DOXYGEN_SKIP") # indicate sections to skip from documentation
foreach(_hala_option HALA_ENABLE_LAPACK HALA_ENABLE_CUDA HALA_ENABLE_ROCM
                     HALA_ENABLE_CHOLMOD
                     HALA_ENABLE_SSE HALA_ENABLE_AVX HALA_ENABLE_AVX512
                     __SSE3__ __AVX__ __AVX512F__)
    if (${_hala_option})
        list(APPEND DOXYGEN_PREDEFINED ${_hala_option})
    endif()
endforeach()


set(DOXYGEN_HTML_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/../html/")

set(DOXYGEN_HTML_EXTRA_STYLESHEET  "${CMAKE_CURRENT_SOURCE_DIR}/hala.css")
set(DOXYGEN_HTML_HEADER            "${CMAKE_CURRENT_SOURCE_DIR}/header.html")
set(DOXYGEN_HTML_FOOTER            "${CMAKE_CURRENT_SOURCE_DIR}/footer.html")
set(DOXYGEN_FORMULA_FONTSIZE "16")
set(DOXYGEN_HTML_COLORSTYLE_HUE    "120") # blue-ish pages, with box color rgb(240,243,249) (210, 90, 100)
set(DOXYGEN_HTML_COLORSTYLE_SAT    "50")
set(DOXYGEN_HTML_COLORSTYLE_GAMMA  "200")

set(DOXYGEN_PROJECT_NAME   "HALA: Handy Accelerated Linear Algebra v${HALA_VERSION_MAJOR}.${HALA_VERSION_MINOR}")
set(DOXYGEN_PROJECT_NUMBER "")

set(DOXYGEN_EXAMPLE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../")

set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${CMAKE_CURRENT_SOURCE_DIR}/../README.md")

doxygen_add_docs(hala_doxygen
                 doxygen/motivation.md
                 README.md
                 doxygen/installation.md
                 doxygen/basic_usage.md
                 cmake/hala_config.hpp
                 common/
                 blas/
                 lapack/
                 sparse/
                 gpu/
                 wax/
                 hex/solvers
                 hex/cholmod
                 hex/vex
                 examples/example_gauss_seidel.cpp
                 examples/example_gpu.cpp
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../
                 COMMENT "Building the HALA documentation")

#add_custom_target(hala_documentation ALL DEPENDS hala_doxygen)
