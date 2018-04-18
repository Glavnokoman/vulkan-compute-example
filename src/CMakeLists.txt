find_package(Vulkan REQUIRED)

include(CompileShader)
compile_shader(saxpy_shader
   SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/saxpy.comp
   TARGET ${CMAKE_CURRENT_BINARY_DIR}/shaders/saxpy.spv
)

add_library(example_filter STATIC example_filter.cpp vulkan_helpers.cpp)
target_link_libraries(example_filter PUBLIC Vulkan::Vulkan)
add_dependencies(example_filter saxpy_shader)
