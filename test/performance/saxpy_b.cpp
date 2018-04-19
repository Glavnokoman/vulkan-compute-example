#include <sltbench/Bench.h>

#include <example_filter.h>
#include <vulkan_helpers.hpp>

#include <vector>

namespace {

struct Params{
   uint32_t width;
   uint32_t height;
   float a;
   
   auto operator== (const Params& other) const-> bool {
      return width == other.width && height == other.height && a == other.a;
   }
   auto operator!= (const Params& other) const-> bool { return !(*this == other); }
   
   friend auto operator<< (std::ostream& s, const Params& p)-> std::ostream& {
      s << "{" << p.width << ", " << p.height << ", " << p.a << "}";
      return s;
   }
};

struct DataFixFull {
   ExampleFilter f{"shaders/saxpy.spv"};
   Params p;
   std::vector<float> y;
   std::vector<float> x;
};

struct FixSaxpyFull: private DataFixFull {
   using Type = DataFixFull;
   
   auto SetUp(const Params& p)-> Type& {
      if(p != this->p) {
         this->p = p;
         y = std::vector<float>(p.width*p.height, 3.1f);
         x = std::vector<float>(p.width*p.height, 1.9f);
      }
      return *this;
   }
   
   auto TearDown()-> void {}
}; // class FixSaxpyCopy

//
void saxpy_full(DataFixFull& fix, const Params& p) {
   auto d_y = vuh::DeviceBufferOwn<float>::fromHost(fix.y, fix.f.device, fix.f.physDevice);
   auto d_x = vuh::DeviceBufferOwn<float>::fromHost(fix.x, fix.f.device, fix.f.physDevice);
   
   fix.f(d_y, d_x, {fix.p.width, fix.p.height, fix.p.a});
}

} // namespace

static const auto params = std::vector<Params>({{32u, 32u, 2.f}, {128, 128, 2.f}});

SLTBENCH_FUNCTION_WITH_FIXTURE_AND_ARGS(saxpy_full, FixSaxpyFull, params);

SLTBENCH_MAIN();
