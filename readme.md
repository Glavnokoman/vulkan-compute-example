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

# Dependencies

- cmake
- [catch2](https://github.com/catchorg/Catch2) (optional)
- [sltbench](https://github.com/ivafanas/sltbench) (optional)
