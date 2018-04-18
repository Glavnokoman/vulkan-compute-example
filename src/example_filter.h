#pragma once

#include "vulkan_helpers.h"

/// doc me
struct ExampleFilter {
	static constexpr auto NumDescriptors = uint32_t(2); ///< number of binding descriptors (array input-output parameters)
	
	/// C++ mirror of the shader push constants interface
	struct PushParams {
		uint32_t width;  ///< frame width
		uint32_t height; ///< frame height
		float a;         ///< saxpy (\$ y = y + ax \$) scaling factor
	};
	
public: // data
	vk::Instance instance;             ///< Vulkan instance
	VkDebugReportCallbackEXT debugReportCallback; //
	vk::PhysicalDevice physDevice;     ///< physical device
	vk::Device device;                 ///< logical device providing access to a physical one
	vk::ShaderModule shader;           ///< compute shader
	vk::Queue queue;                   ///< command queue supporting compute operations
	vk::DescriptorSetLayout dscLayout; ///< c++ definition of the shader binding interface
	vk::DescriptorPool dscPool;        ///< descriptors pool
	vk::CommandPool cmdPool;           ///< used to allocate command buffers
	vk::PipelineCache pipeCache;       ///< pipeline cache
	vk::PipelineLayout pipeLayout;     ///< defines shader interface as a set of layout bindings and push constants
	
	// buffers. do not really belong to filter. move out.
	vk::Buffer status;
	vk::DeviceMemory statusMem;
	vk::Buffer range;
	vk::DeviceMemory rangeMem;
	vk::Buffer variance;
	vk::DeviceMemory varianceMem;
	vk::Buffer normals;
	vk::DeviceMemory normalsMem;
	
	vk::Pipeline pipe;                   ///< pipeline to submit compute commands
	mutable vk::CommandBuffer cmdBuffer; ///< used to record commands to be queued
	
public:
	explicit ExampleFilter(const std::string& shaderPath);
	~ExampleFilter() noexcept;
	// auto updateSpecializations()-> void; // to implement
	
	auto bindParameters(vk::Buffer& out, const vk::Buffer& in, const PushParams& p) const-> void;
	auto run() const-> void;
	auto operator()(vk::Buffer& out, const vk::Buffer& in, const PushParams& p ) const-> void;
private: // helpers		
	static auto createInstance(const std::vector<const char*> layers
	                           , const std::vector<const char*> extensions
	                           )-> vk::Instance;
	
	static auto createDescriptorSetLayout(const vk::Device& device)-> vk::DescriptorSetLayout;
	static auto allocDescriptorPool(const vk::Device& device)-> vk::DescriptorPool;
	
	static auto createPipelineLayout(const vk::Device& device
	                                 , const vk::DescriptorSetLayout& dscLayout
	                                 )-> vk::PipelineLayout;
	
	static auto createComputePipeline(const vk::Device& device, const vk::ShaderModule& shader
	                                  , const vk::PipelineLayout& pipeLayout
	                                  , const vk::PipelineCache& cache
	                                  )-> vk::Pipeline;
	
	static auto createDescriptorSet(const vk::Device& device, const vk::DescriptorPool& pool
	                                , const vk::DescriptorSetLayout& layout
	                                , vk::Buffer& out
	                                , const vk::Buffer& in
	                                , uint32_t size
	                                )-> vk::DescriptorSet;
	
	static auto createCommandBuffer(const vk::Device& device, const vk::CommandPool& cmdPool
	                                , const vk::Pipeline& pipeline, const vk::PipelineLayout& pipeLayout
	                                , const vk::DescriptorSet& dscSet
	                                , const PushParams& p
	                                )-> vk::CommandBuffer;
}; // struct MixpixFilter
