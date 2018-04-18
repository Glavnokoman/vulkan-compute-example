#pragma once

#include "vulkan_helpers.h"

#include <vulkan/vulkan.hpp>

#include <iostream>
#include <fstream>
#include <type_traits>

namespace vuh {

template<class T>
auto div_up(T x, T y)-> T { return (x + y - 1)/y; }

template<class T> using raw_ptr = typename std::add_pointer<T>::type;

/// access to device memory from the host
template<class T>
struct BufferHostView {
	using value_type = T;
	using ptr_type = raw_ptr<T>;
	
	std::unique_ptr<const vk::Device> device;
	const vk::DeviceMemory& devMemory;
	const ptr_type data; ///< points to the first element
	const size_t size;   ///< number of elements
	
	BufferHostView(BufferHostView&&) = default;
	auto operator=(BufferHostView&&)-> BufferHostView& = default;
	
	/// Constructor
	explicit BufferHostView(const vk::Device& device
	                        , const vk::DeviceMemory& devMem
	                        , size_t nelements ///< number of elements
	                        , uint32_t offset = 0
	                        )
	   : device(&device), devMemory(devMem)
	   , data(ptr_type(device.mapMemory(devMem, offset, nelements*sizeof(T))))
	   , size(nelements)
	{}
	
	~BufferHostView() noexcept {
		if(device){
			device->unmapMemory(devMemory);
			device.release();
		}
	}
	
	auto begin()->ptr_type { return data; }
	auto end()-> ptr_type { return data + size; }
}; // BufferHostView

/// Device buffer owning its chunk of memory.
/// @TODO when physical device is discrete GPU non-host-visible buffer should have VK_BUFFER_USAGE_TRANSFER_SRC_BIT and DST bits 
template<class T>
class DeviceBufferOwn {
private:
	vk::Buffer _buf;                        ///< device buffer
	vk::DeviceMemory _mem;                  ///< associated chunk of device memorys
	vk::PhysicalDevice _physdev;            ///< physical device owning the memory
	std::unique_ptr<const vk::Device> _dev; ///< pointer to logical device. no real ownership, just to provide value semantics to the class.
	vk::MemoryPropertyFlags _flags;         ///< Actual flags of allocated memory. Can be a superset of requested flags.
	size_t _size;                           ///< number of elements. actual allocated memory may be a bit bigger than necessary.
public:
	using value_type = T;

	DeviceBufferOwn(DeviceBufferOwn&&) = default;
	auto operator=(DeviceBufferOwn&&)-> DeviceBufferOwn& = default;
	
	/// Constructor
	explicit DeviceBufferOwn(const vk::Device& device, const vk::PhysicalDevice& physDevice
	                         , uint32_t n_elements ///< number of elements of corresponding type
	                         , vk::MemoryPropertyFlags properties=vk::MemoryPropertyFlagBits::eDeviceLocal
	                         , vk::BufferUsageFlags usage=vk::BufferUsageFlagBits::eStorageBuffer
	                         )
	   : DeviceBufferOwn(device, physDevice
	       , createBuffer(device, n_elements*sizeof(T), update_usage(physDevice, properties, usage))
	       , properties, n_elements)
	{}
	
	/// Destructor
	~DeviceBufferOwn() noexcept {
		if(_dev){
			_dev->freeMemory(_mem);
			_dev->destroyBuffer(_buf);
			_dev.release();
		}
	}
	
	template<class C>
	static auto fromHost(C&& c, const vk::Device& device, const vk::PhysicalDevice& physDev
	                     , vk::MemoryPropertyFlags properties=vk::MemoryPropertyFlagBits::eDeviceLocal
								, vk::BufferUsageFlags usage=vk::BufferUsageFlagBits::eStorageBuffer
	                    )-> DeviceBufferOwn 
	{
		auto r = DeviceBufferOwn<T>(device, physDev, uint32_t(c.size()), properties, usage);
		if(r._flags & vk::MemoryPropertyFlagBits::eHostVisible){ // memory is host-visible
			auto hv = r.host_view();
			std::copy(begin(c), end(c), hv.data);
		} else { // memory is not host visible
			auto stage_buf = fromHost(std::forward<C>(c), device, physDev
			                          , vk::MemoryPropertyFlagBits::eHostVisible
			                          , vk::BufferUsageFlagBits::eTransferSrc);
			// copy from staging buffer to device memory
			const auto qf_id = getComputeQueueFamilyId(physDev); // queue family id, TODO: use transfer queue
			auto cmd_pool = device.createCommandPool({vk::CommandPoolCreateFlagBits::eTransient, qf_id});
			auto cmd_buf = device.allocateCommandBuffers({cmd_pool, vk::CommandBufferLevel::ePrimary, 1})[0];
			cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
			auto region = vk::BufferCopy(0, 0, stage_buf.size()*sizeof(T));
			cmd_buf.copyBuffer(stage_buf, r, 1, &region);
			cmd_buf.end();
			auto queue = device.getQueue(qf_id, 0);
			auto submit_info = vk::SubmitInfo(0, nullptr, nullptr, 1, &cmd_buf);
			queue.submit({submit_info}, nullptr);
			queue.waitIdle();
			device.freeCommandBuffers(cmd_pool, 1, &cmd_buf);
			device.destroyCommandPool(cmd_pool);
		}
		return r;
	}

	operator vk::Buffer& () { return *reinterpret_cast<vk::Buffer*>(this + offsetof(DeviceBufferOwn, _buf)); }
	operator const vk::Buffer& () const { return *reinterpret_cast<const vk::Buffer*>(this + offsetof(DeviceBufferOwn, _buf)); }

	/// @return number of items in the buffer
	auto size() const-> size_t {
		return _size;
	}
	
	template<class C>
	auto to_host(C& c)-> void {
		if(_flags & vk::MemoryPropertyFlagBits::eHostVisible){ // memory IS host visible
			auto hv = host_view();
			c.resize(size());
			std::copy(std::begin(hv), std::end(hv), c.data());
		} else { // memory is not host visible
			// copy device memory to staging buffer
			auto stage_buf = DeviceBufferOwn(*_dev, _physdev, size()
			                                 , vk::MemoryPropertyFlagBits::eHostVisible
			                                 , vk::BufferUsageFlagBits::eTransferDst);
			const auto qf_id = getComputeQueueFamilyId(_physdev); // queue family id, TODO: use transfer queue
			auto cmd_pool = _dev->createCommandPool({vk::CommandPoolCreateFlagBits::eTransient, qf_id});
			auto cmd_buf = _dev->allocateCommandBuffers({cmd_pool, vk::CommandBufferLevel::ePrimary, 1})[0];
			cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
			auto region = vk::BufferCopy(0, 0, size()*sizeof(T));
			cmd_buf.copyBuffer(_buf, stage_buf, 1, &region);
			cmd_buf.end();
			auto queue = _dev->getQueue(qf_id, 0);
			auto submit_info = vk::SubmitInfo(0, nullptr, nullptr, 1, &cmd_buf);
			queue.submit({submit_info}, nullptr);
			queue.waitIdle();
			_dev->freeCommandBuffers(cmd_pool, 1, &cmd_buf);
			_dev->destroyCommandPool(cmd_pool);
			
			stage_buf.to_host(c); // copy from staging buffer to host
		}
	}
	
	///
	auto host_view()-> BufferHostView<T> {
		return BufferHostView<T>(*_dev, _mem, size());
	}
	
private: // helpers
	/// Helper constructor
	explicit DeviceBufferOwn(const vk::Device& device, const vk::PhysicalDevice& physDevice
	                         , vk::Buffer buffer
	                         , vk::MemoryPropertyFlags properties
	                         , size_t size
	                         )
	   : DeviceBufferOwn(device, physDevice, buffer, size
	                     , selectMemory(physDevice, device, buffer, properties))
	{}
	
	/// Helper constructor. This one does the actual construction and binding.
	explicit DeviceBufferOwn(const vk::Device& device, const vk::PhysicalDevice& physDevice
	                         , vk::Buffer buf, size_t size
	                         , uint32_t memory_id)
	   : _buf(buf)
	   , _mem(allocMemory(physDevice, device, buf, memory_id))
	   , _physdev(physDevice)
	   , _dev(&device)
	   , _flags(physDevice.getMemoryProperties().memoryTypes[memory_id].propertyFlags)
	   , _size(size)
	{
		device.bindBufferMemory(buf, _mem, 0);
	}
	
	/// crutch to modify buffer usage
	auto update_usage(const vk::PhysicalDevice& physDevice
	                  , vk::MemoryPropertyFlags properties
	                  , vk::BufferUsageFlags usage
	                  )-> vk::BufferUsageFlags 
	{
		if(physDevice.getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu
		   && properties == vk::MemoryPropertyFlagBits::eDeviceLocal
		   && usage == vk::BufferUsageFlagBits::eStorageBuffer)
		{
			usage |= vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
		}
		return usage;
	}
}; // DeviceBufferOwn

} // namespace vuh

