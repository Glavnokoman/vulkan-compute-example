# Vulkan Compute Example

Simple (but complete) example of Vulkan use for GPGPU computing.
Saxpy kernel computation on 2d arrays.

Features covered:
- Vulkan boilerplate setup using vulkan-hpp
- data copy between host and device-local memory
- passing array parameters to shader (layout bindings)
- passing non-array parameters to shader (push constants)
- define workgroup dimensions (specialization constants)
- very simple glsl shader (saxpy)
- glsl to spir-v compilation (build time)

This was an attempt to structure the Vulkan compute code in a way that would be easy to modify for each particular use case.
I think I failed here so this example still sucks. But I learned while doing this and as a result there is a [vuh](https://github.com/Glavnokoman/vuh) Vulkan compute library which enables you to do the same but in (literally) 10 lines of code. You're cordially invited to use that instead.

## Dependencies
- c++14 compatible compiler
- cmake
- [vulkan-headers](https://github.com/KhronosGroup/Vulkan-Docs)
- [vulkan-hpp](https://github.com/KhronosGroup/Vulkan-Hpp)
- [glslang](https://github.com/KhronosGroup/glslang)
- [catch2](https://github.com/catchorg/Catch2) (optional)
- [sltbench](https://github.com/ivafanas/sltbench) (optional)
