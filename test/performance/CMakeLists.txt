if(NOT VULKAN_COMPUTE_EXAMPLE_BUILD_BENCHMARKS)
   return()
endif()

add_custom_command(
   OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders
   COMMAND ${CMAKE_COMMAND}
   ARGS -E create_symlink $<TARGET_FILE_DIR:example_filter>/shaders ${CMAKE_CURRENT_BINARY_DIR}/shaders
   DEPENDS example_filter
   COMMENT "link shaders to build tests folder"
)
add_custom_target(link_shaders_bench DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/shaders)

find_package(benchmark)
add_executable(bench_saxpy saxpy_b.cpp)
target_link_libraries(bench_saxpy PRIVATE benchmark::benchmark example_filter)
