#define CATCH_CONFIG_MAIN
#include <catch/catch.hpp>

#include "approx.hpp"

#include <example_filter.h>
#include <vulkan_helpers.hpp>

#include <iostream>
#include <fstream>

using test::approx;

TEST_CASE("saxpy", "[correctness]"){
	const auto width = 90;
	const auto height = 60;
	const auto a = 2.0f; // saxpy scaling factor
	
	auto y = std::vector<float>(width*height, 0.71f);
	auto x = std::vector<float>(width*height, 0.65f);
	
	ExampleFilter f("shaders/saxpy.spv");
	auto d_y = vuh::DeviceBufferOwn<float>::fromHost(y, f.device, f.physDevice);
	auto d_x = vuh::DeviceBufferOwn<float>::fromHost(x, f.device, f.physDevice);

	f(d_y, d_x, {width, height, a});

	auto out_tst = std::vector<float>{};
	d_y.to_host(out_tst);

	auto out_ref = y;
	for(size_t i = 0; i < y.size(); ++i){
		out_ref[i] += a*x[i];
	}
	
	REQUIRE(out_tst == approx(out_ref).eps(1.e-5).verbose());
}
