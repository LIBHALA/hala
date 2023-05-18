
add_executable(example_gauss_seidel example_gauss_seidel.cpp)
target_link_libraries(example_gauss_seidel HALA::HALA)

add_executable(example_gpu example_gpu.cpp)
target_link_libraries(example_gpu HALA::HALA)

add_executable(example_bench_mygpu example_bench_mygpu.cpp)
target_link_libraries(example_bench_mygpu HALA::HALA)

add_executable(example_vex example_vex.cpp)
target_link_libraries(example_vex HALA::HALA)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt.cmake ${CMAKE_CURRENT_BINARY_DIR}/configured/CMakeLists.txt)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/configured/CMakeLists.txt
              ${CMAKE_CURRENT_SOURCE_DIR}/example_gauss_seidel.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/example_gpu.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/example_vex.cpp
        DESTINATION share/hala/examples/)
