// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "open3d/core/SYCLUtils.h"

#include <vector>

#include "open3d/core/MemoryManager.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Timer.h"
#include "tests/Tests.h"
#include "tests/core/CoreTest.h"

namespace open3d {
namespace tests {

TEST(SYCLUtils, SYCLDemo) { EXPECT_EQ(core::sycl_utils::SYCLDemo(), 0); }

TEST(SYCLUtils, PrintAllSYCLDevices) {
    core::sycl_utils::PrintSYCLDevices(/*print_all=*/true);
}

TEST(SYCLUtils, PrintSYCLDevices) {
    core::sycl_utils::PrintSYCLDevices(/*print_all=*/false);
}

TEST(SYCLUtils, SYCLMemoryManager) {
    std::vector<float> src_host_data{0.0, 1.0, 2.0, 3.0};  // 16 bytes.
    const void* src_host_ptr = static_cast<void*>(src_host_data.data());
    size_t byte_size = src_host_data.size() * sizeof(float);

    core::Device host = core::Device("CPU:0");
    core::Device sycl_cpu = core::Device("SYCL_CPU:0");
    core::Device sycl_gpu = core::Device("SYCL_GPU:0");

    void* dst_host_ptr = core::MemoryManager::Malloc(byte_size, host);
    auto reset_dst_host_ptr = [dst_host_ptr]() {
        static_cast<float*>(dst_host_ptr)[0] = 3.14;
        static_cast<float*>(dst_host_ptr)[1] = 3.14;
        static_cast<float*>(dst_host_ptr)[2] = 3.14;
        static_cast<float*>(dst_host_ptr)[3] = 3.14;
    };
    auto is_dst_host_ptr_correct = [dst_host_ptr]() -> bool {
        bool rc = true;
        rc = rc && static_cast<float*>(dst_host_ptr)[0] == 0.0;
        rc = rc && static_cast<float*>(dst_host_ptr)[1] == 1.0;
        rc = rc && static_cast<float*>(dst_host_ptr)[2] == 2.0;
        rc = rc && static_cast<float*>(dst_host_ptr)[3] == 3.0;
        return rc;
    };

    // SYCL_CPU.
    void* sycl_cpu_ptr = core::MemoryManager::Malloc(byte_size, sycl_cpu);
    core::MemoryManager::Memcpy(sycl_cpu_ptr, sycl_cpu, src_host_ptr, host,
                                byte_size);
    reset_dst_host_ptr();
    EXPECT_FALSE(is_dst_host_ptr_correct());
    core::MemoryManager::Memcpy(dst_host_ptr, host, sycl_cpu_ptr, sycl_cpu,
                                byte_size);
    EXPECT_TRUE(is_dst_host_ptr_correct());
    core::MemoryManager::Free(sycl_cpu_ptr, sycl_cpu);

    // SYCL_GPU.
    void* sycl_gpu_ptr = core::MemoryManager::Malloc(byte_size, sycl_gpu);
    core::MemoryManager::Memcpy(sycl_gpu_ptr, sycl_gpu, src_host_ptr, host,
                                byte_size);
    reset_dst_host_ptr();
    EXPECT_FALSE(is_dst_host_ptr_correct());
    core::MemoryManager::Memcpy(dst_host_ptr, host, sycl_gpu_ptr, sycl_gpu,
                                byte_size);
    EXPECT_TRUE(is_dst_host_ptr_correct());
    core::MemoryManager::Free(sycl_gpu_ptr, sycl_gpu);

    // Clean up.
    core::MemoryManager::Free(dst_host_ptr, host);
}

}  // namespace tests
}  // namespace open3d
