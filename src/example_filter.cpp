#include "example_filter.h"

#include "vulkan_helpers.hpp"

#include <vulkan/vulkan.hpp>

#define ARR_VIEW(x) uint32_t(x.size()), x.data()
#define ST_VIEW(s)  uint32_t(sizeof(s)), &s

using namespace vuh;
namespace {
	constexpr uint32_t WORKGROUP_SIZE = 16; ///< compute shader workgroup dimension is WORKGROUP_SIZE x WORKGROUP_SIZE

#ifdef NDEBUG
	constexpr bool enableValidation = false;
#else
	constexpr bool enableValidation = true;
#endif
} // namespace


/// Constructor
ExampleFilter::ExampleFilter(const std::string& shaderPath){
	auto layers = enableValidation ? enabledLayers({"VK_LAYER_LUNARG_standard_validation"})
											 : std::vector<const char*>{};
	auto extensions = enableValidation ? enabledExtensions({VK_EXT_DEBUG_REPORT_EXTENSION_NAME})
												  : std::vector<const char*>{};
	instance = createInstance(layers, extensions);
	debugReportCallback = enableValidation ? registerValidationReporter(instance, debugReporter)
														: nullptr;
	physDevice = instance.enumeratePhysicalDevices()[0]; // just use the first device
	const auto queueFamilyId = getComputeQueueFamilyId(physDevice);
	device = createDevice(physDevice, layers, queueFamilyId); // TODO: when physical device is a discrete gpu, transfer queue needs to be included.
	shader = loadShader(device, shaderPath.c_str());
	queue = device.getQueue(queueFamilyId, 0); // get the first queue in the suitable family

	dscLayout = createDescriptorSetLayout(device);
	dscPool = allocDescriptorPool(device);
	auto commandPoolCI = vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlags(), queueFamilyId);
	cmdPool = device.createCommandPool(commandPoolCI);
	pipeCache = device.createPipelineCache(vk::PipelineCacheCreateInfo());
	pipeLayout = createPipelineLayout(device, dscLayout);

	pipe = createComputePipeline(device, shader, pipeLayout, pipeCache);
	cmdBuffer = vk::CommandBuffer{};
}

/// Destructor
ExampleFilter::~ExampleFilter() noexcept {
	device.destroyPipeline(pipe);
	device.destroyPipelineLayout(pipeLayout);
	device.destroyPipelineCache(pipeCache);
	device.destroyCommandPool(cmdPool);
	device.destroyDescriptorPool(dscPool);
	device.destroyDescriptorSetLayout(dscLayout);
	device.destroyShaderModule(shader);
	device.destroy();

	if(enableValidation){
		// unregister callback.
		auto destroyFn = PFN_vkDestroyDebugReportCallbackEXT(
					vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT"));
		if(destroyFn){
			destroyFn(instance, debugReportCallback, nullptr);
		} else {
			std::cerr << "Could not load vkDestroyDebugReportCallbackEXT\n";
		}
	}

	instance.destroy();
}

///
auto ExampleFilter::bindParameters(vk::Buffer& out, const vk::Buffer& in
                                   , const ExampleFilter::PushParams& p
                                  ) const-> void
{
	auto dscSet = createDescriptorSet(device, dscPool, dscLayout, out, in, p.width*p.height);
	cmdBuffer = createCommandBuffer(device, cmdPool, pipe, pipeLayout, dscSet, p);
}

/// run (sync) the filter on previously bound parameters
auto ExampleFilter::run() const-> void {
	assert(cmdBuffer != vk::CommandBuffer{}); // TODO: this should be a check for a valid command buffer
	auto fence = device.createFence(vk::FenceCreateInfo());
	auto submitInfo = vk::SubmitInfo(0, nullptr, nullptr, 1, &cmdBuffer); // submit a single command buffer

	// submit the command buffer to the queue and set up a fence.
	queue.submit({submitInfo}, fence);
	device.waitForFences({fence}, true, uint64_t(-1));
	device.destroyFence(fence);
}

/// run (sync) the filter
auto ExampleFilter::operator()(vk::Buffer& out, const vk::Buffer& in
                               , const ExampleFilter::PushParams& p
                              ) const-> void
{
	bindParameters(out, in, p);
	run();
}

/// Create vulkan Instance with app specific parameters.
auto ExampleFilter::createInstance(const std::vector<const char*> layers
                                  , const std::vector<const char*> extensions
                                  )-> vk::Instance
{
	auto appInfo = vk::ApplicationInfo("Mixed Pixel Filter", 0, "mixpix_engine"
	                                   , 0, VK_API_VERSION_1_0); // The only important field here is apiVersion
	auto createInfo = vk::InstanceCreateInfo(vk::InstanceCreateFlags(), &appInfo
	                                         , ARR_VIEW(layers), ARR_VIEW(extensions));
	return vk::createInstance(createInfo);
}

/// Specify a descriptor set layout (number and types of descriptors).
auto ExampleFilter::createDescriptorSetLayout(const vk::Device& device)-> vk::DescriptorSetLayout {
	auto bindLayout = std::array<vk::DescriptorSetLayoutBinding, NumDescriptors>{{
	            {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}
	           ,{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}
		                                                                          }};
	auto layoutCI = vk::DescriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags()
	                                                  , ARR_VIEW(bindLayout));
	return device.createDescriptorSetLayout(layoutCI);
}

/// Allocate descriptor pool for a descriptors to all storage buffer in use
auto ExampleFilter::allocDescriptorPool(const vk::Device& device)-> vk::DescriptorPool {
	auto descriptorPoolSize = vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, NumDescriptors);
	auto descriptorPoolCI = vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(), 1
	                                                     , 1, &descriptorPoolSize);
	return device.createDescriptorPool(descriptorPoolCI);
}

/// Pipeline layout defines shader interface as a set of layout bindings and push constants.
auto ExampleFilter::createPipelineLayout(const vk::Device& device
                                         , const vk::DescriptorSetLayout& dscLayout
                                        )-> vk::PipelineLayout
{
	auto pushConstantsRange = vk::PushConstantRange(vk::ShaderStageFlagBits::eCompute
	                                                , 0, sizeof(PushParams));
	auto pipelineLayoutCI = vk::PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags()
	                                                     , 1, &dscLayout, 1, &pushConstantsRange);
	return device.createPipelineLayout(pipelineLayoutCI);
}

/// Create compute pipeline consisting of a single stage with compute shader.
/// Specialization constants specialized here.
auto ExampleFilter::createComputePipeline(const vk::Device& device, const vk::ShaderModule& shader
                                         , const vk::PipelineLayout& pipeLayout
                                         , const vk::PipelineCache& cache
                                         )-> vk::Pipeline
{
	// specialize constants of the shader
	auto specEntries = std::array<vk::SpecializationMapEntry, 2>{
		{{0, 0, sizeof(int)}, {1, 1*sizeof(int), sizeof(int)}}
	};
	auto specValues = std::array<int, 2>{WORKGROUP_SIZE, WORKGROUP_SIZE};
	auto specInfo = vk::SpecializationInfo(ARR_VIEW(specEntries)
	                                       , specValues.size()*sizeof(int), specValues.data());

	// Specify the compute shader stage, and it's entry point (main), and specializations
	auto stageCI = vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags()
	                                                 , vk::ShaderStageFlagBits::eCompute
	                                                 , shader, "main", &specInfo);
	auto pipelineCI = vk::ComputePipelineCreateInfo(vk::PipelineCreateFlags()
	                                                , stageCI, pipeLayout);
	return device.createComputePipeline(cache, pipelineCI, nullptr);
}

/// Create descriptor set. Actually associate buffers to binding points in bindLayout.
/// Buffer sizes are specified here as well.
auto ExampleFilter::createDescriptorSet(const vk::Device& device, const vk::DescriptorPool& pool
                                       , const vk::DescriptorSetLayout& layout
                                       , vk::Buffer& out, const vk::Buffer& in, uint32_t size
                                       )-> vk::DescriptorSet
{
	auto descriptorSetAI = vk::DescriptorSetAllocateInfo(pool, 1, &layout);
	auto descriptorSet = device.allocateDescriptorSets(descriptorSetAI)[0];

	auto outInfo = vk::DescriptorBufferInfo(out, 0, sizeof(float)*size);
	auto inInfo = vk::DescriptorBufferInfo(in, 0, sizeof(float)*size);

	auto writeDsSets = std::array<vk::WriteDescriptorSet, NumDescriptors>{{
	           {descriptorSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &outInfo}
	          ,{descriptorSet, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &inInfo}
		                                                                   }};

	device.updateDescriptorSets(writeDsSets, {});
	return descriptorSet;
}

/// Create command buffer, push the push constants, bind descriptors and define the work batch size.
/// All command buffers allocated from given command pool must be submitted to queues of corresponding 
/// family ONLY.
auto ExampleFilter::createCommandBuffer(const vk::Device& device, const vk::CommandPool& cmdPool
                                       , const vk::Pipeline& pipeline
                                       , const vk::PipelineLayout& pipeLayout
                                       , const vk::DescriptorSet& dscSet
                                       , const ExampleFilter::PushParams& p
                                       )-> vk::CommandBuffer
{
	// allocate a command buffer from the command pool.
	auto commandBufferAI = vk::CommandBufferAllocateInfo(cmdPool, vk::CommandBufferLevel::ePrimary, 1);
	auto commandBuffer = device.allocateCommandBuffers(commandBufferAI)[0];

	// Start recording commands into the newly allocated command buffer.
	auto beginInfo = vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit); // buffer is only submitted and used once
	commandBuffer.begin(beginInfo);

	// Before dispatch bind a pipeline, AND a descriptor set.
	// The validation layer will NOT give warnings if you forget those.
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeLayout
	                                 , 0, {dscSet}, {});

	commandBuffer.pushConstants(pipeLayout, vk::ShaderStageFlagBits::eCompute, 0, ST_VIEW(p));

	// Start the compute pipeline, and execute the compute shader.
	// The number of workgroups is specified in the arguments.
	commandBuffer.dispatch(div_up(p.width, WORKGROUP_SIZE), div_up(p.height, WORKGROUP_SIZE), 1);
	commandBuffer.end(); // end recording commands
	return commandBuffer;
}
