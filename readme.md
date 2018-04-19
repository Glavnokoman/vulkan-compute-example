# Vulkan Compute Example

Simple (but complete) example of Vulkan use for GPGPU computing.
Saxpy kernel computation on 2d arrays.

Features covered:
- Vulkan boilerplate setup using vulkan-hpp
- compiling glsl shader to spir-v
- data copy between host and device-local memory
- binding array parameters to the shader
- push constants (to define array dimensions and scaling constant)
- specialization constants (to define the workgroup dimensions)

# Dependencies

- cmake
- [catch2](https://github.com/catchorg/Catch2) (optional)
- [sltbench](https://github.com/ivafanas/sltbench) (optional)
