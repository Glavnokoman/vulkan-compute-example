#include "vulkan_helpers.h"
#include "vulkan_helpers.hpp"

#include <iostream>
#include <fstream>

using std::begin;
using std::end;
#define ALL(x) begin(x), end(x)
#define ARR_VIEW(x) uint32_t(x.size()), x.data()

namespace vuh {
	
VKAPI_ATTR VkBool32 VKAPI_CALL debugReporter(
      VkDebugReportFlagsEXT , VkDebugReportObjectTypeEXT, uint64_t, size_t, int32_t
      , const char*                pLayerPrefix
      , const char*                pMessage
      , void*                      /*pUserData*/
      ){
   std::cerr << "[WARNING]: Vulkan says: " << pLayerPrefix << ": " << pMessage << "\n";
   return VK_FALSE;
}

/// Read binary shader file into array of uint32_t. little endian assumed.
/// Padded by 0s to a boundary of 4.
auto readShaderSrc(const char* filename)-> std::vector<char> {
	auto fin = std::ifstream(filename, std::ios::binary);
	if(!fin.is_open()){
		throw std::runtime_error(std::string("could not open file ") + filename);
	}
	auto ret = std::vector<char>(std::istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>());
	
	ret.resize(4*div_up(ret.size(), size_t(4)));
	return ret;
}

/// create shader module, reading spir-v from a file
auto loadShader(const vk::Device& device, const char* filename
                , vk::ShaderModuleCreateFlags flags
                )-> vk::ShaderModule
{
	auto code = readShaderSrc(filename);
	auto shaderCI = vk::ShaderModuleCreateInfo(flags, code.size()
	                                           , reinterpret_cast<uint32_t*>(code.data()));
	return device.createShaderModule(shaderCI);
}

/// filter list of desired extensions to include only those supported by current Vulkan instance
auto enabledExtensions(const std::vector<const char*>& extensions)-> std::vector<const char*> {
	auto ret = std::vector<const char*>{};
	auto instanceExtensions = vk::enumerateInstanceExtensionProperties();
	for(auto e: extensions){
		auto it = std::find_if(ALL(instanceExtensions)
		                       , [=](auto& p){ return strcmp(p.extensionName, e);});
		if(it != end(instanceExtensions)){
			ret.push_back(e);
		} else {
			std::cerr << "[WARNING]: extension " << e << " is not found" "\n";
		}
	}
	return ret;
}

/// filter list of desired extensions to include only those supported by current Vulkan instance
auto enabledLayers(const std::vector<const char*>& layers)-> std::vector<const char*> {
	auto ret = std::vector<const char*>{};
	auto instanceLayers = vk::enumerateInstanceLayerProperties();
	for(auto l: layers){
		auto it = std::find_if(ALL(instanceLayers)
		                       , [=](auto& p){ return strcmp(p.layerName, l);});
		if(it != end(instanceLayers)){
			ret.push_back(l);
		} else {
			std::cerr << "[WARNING] layer " << l << " is not found" "\n";
		}
	}
	return ret;
}

/// Register a callback function for the extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME,
/// so that warnings emitted from the validation layer are actually printed.
auto registerValidationReporter(const vk::Instance& instance, PFN_vkDebugReportCallbackEXT reporter
                                )-> VkDebugReportCallbackEXT
{
	auto ret = VkDebugReportCallbackEXT(nullptr);
	auto createInfo = VkDebugReportCallbackCreateInfoEXT{};
	createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
	createInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT
	      | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
	createInfo.pfnCallback = reporter;
	
	// We have to explicitly load this function
	auto createFN = PFN_vkCreateDebugReportCallbackEXT(
	         instance.getProcAddr("vkCreateDebugReportCallbackEXT"));
	if(createFN){
		createFN(instance, &createInfo, nullptr, &ret);
	} else {
		std::cerr << "Could not load vkCreateDebugReportCallbackEXT\n";
	}
	return ret;
}

/// create logical device to interact with the physical one
auto createDevice(const vk::PhysicalDevice& physicalDevice, const std::vector<const char*>& layers
                  , uint32_t queueFamilyID
                  )-> vk::Device
{
	// When creating the device specify what queues it has
	auto p = float(1.0); // queue priority
	auto queueCI = vk::DeviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), queueFamilyID, 1, &p);
	auto devCI = vk::DeviceCreateInfo(vk::DeviceCreateFlags(), 1, &queueCI, ARR_VIEW(layers));
	
	return physicalDevice.createDevice(devCI, nullptr);
}

/// Create buffer on a device. Does NOT allocate memory.
auto createBuffer(const vk::Device& device, uint32_t bufSize
                  , vk::BufferUsageFlags usage
                  )-> vk::Buffer 
{
	auto bufferCI = vk::BufferCreateInfo(vk::BufferCreateFlags(), bufSize, usage);
	return device.createBuffer(bufferCI);
}

/// @return the index of a queue family that supports compute operations.
/// Groups of queues that have the same capabilities (for instance, they all supports graphics
/// and computer operations), are grouped into queue families.
/// When submitting a command buffer, you must specify to which queue in the family you are submitting to.
auto getComputeQueueFamilyId(const vk::PhysicalDevice& physicalDevice)-> uint32_t {
	auto queueFamilies = physicalDevice.getQueueFamilyProperties();

	// prefer using compute-only queue
	auto queue_it = std::find_if(ALL(queueFamilies), [](auto& f){
		auto maskedFlags = ~vk::QueueFlagBits::eSparseBinding & f.queueFlags;
		return 0 < f.queueCount
		      && !(vk::QueueFlagBits::eGraphics & maskedFlags)
		      && (vk::QueueFlagBits::eCompute & maskedFlags);
	});
	if(queue_it != end(queueFamilies)){
		return uint32_t(std::distance(begin(queueFamilies), queue_it));
	}

	// otherwise use any queue that would work
	queue_it = std::find_if(ALL(queueFamilies), [](auto& f){
		auto maskedFlags = ~vk::QueueFlagBits::eSparseBinding & f.queueFlags;
		return 0 < f.queueCount && (vk::QueueFlagBits::eCompute & maskedFlags);
	});
	if(queue_it != end(queueFamilies)){
		return uint32_t(std::distance(begin(queueFamilies), queue_it));
	}

	throw std::runtime_error("could not find a queue family that supports compute operations");
}

/// Select memory with desired properties.
/// @return id of the suitable memory, -1 if no suitable memory found.
auto selectMemory(const vk::PhysicalDevice& physDev
                  , const vk::Device& device
                  , const vk::Buffer& buf
                  , const vk::MemoryPropertyFlags properties ///< desired memory properties
                  )-> uint32_t
{
	auto memProperties = physDev.getMemoryProperties();
	auto memoryReqs = device.getBufferMemoryRequirements(buf);
	for(uint32_t i = 0; i < memProperties.memoryTypeCount; ++i){
		if( (memoryReqs.memoryTypeBits & (1u << i))
		    && ((properties & memProperties.memoryTypes[i].propertyFlags) == properties))
		{
			return i;
		}
	}
	throw std::runtime_error("failed to select memory with required properties");
}

auto allocMemory(const vk::PhysicalDevice& physDev, const vk::Device& device
                 , const vk::Buffer& buf
                 , uint32_t memory_id
                 )-> vk::DeviceMemory 
{
   auto memoryReqs = device.getBufferMemoryRequirements(buf);
   auto allocInfo = vk::MemoryAllocateInfo(memoryReqs.size, memory_id);
   return device.allocateMemory(allocInfo);
}

} // namespace vuh
