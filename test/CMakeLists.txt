# /test cmake file

if(NOT VULKAN_COMPUTE_EXAMPLE_BUILD_TESTS)
   return()
endif()

add_custom_command(
   OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders
   COMMAND ${CMAKE_COMMAND}
   ARGS -E create_symlink $<TARGET_FILE_DIR:example_filter>/shaders ${CMAKE_CURRENT_BINARY_DIR}/shaders
   DEPENDS example_filter
   COMMENT "link shaders to build tests folder"
)
add_custom_target(link_shaders_dir DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/shaders)

find_package(Catch2 REQUIRED)
function(add_catch_test arg_test_name arg_test_src)
	add_executable(${arg_test_name} ${arg_test_src})
	target_link_libraries(${arg_test_name} PRIVATE Catch2::Catch)
   target_include_directories( ${arg_test_name} PRIVATE ${PROJECT_SOURCE_DIR}/src )
	add_test(NAME ${arg_test_name} COMMAND ${arg_test_name} )
   add_dependencies(${arg_test_name} link_shaders_dir)
endfunction()

add_catch_test(test_saxpy saxpy_t.cpp)
target_link_libraries(test_saxpy PRIVATE example_filter)
