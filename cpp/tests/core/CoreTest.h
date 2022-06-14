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

#pragma once

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Device.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/SizeVector.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

class PermuteDtypesWithBool : public testing::TestWithParam<core::Dtype> {
public:
    static std::vector<core::Dtype> TestCases() {
        return {
                core::Bool,  core::UInt8,   core::Int8,    core::UInt16,
                core::Int16, core::UInt32,  core::Int32,   core::UInt64,
                core::Int64, core::Float32, core::Float64,
        };
    }
};

// Select one device for each device type.
class PermuteDevices : public testing::TestWithParam<core::Device> {
public:
    static std::vector<core::Device> TestCases() {
        std::vector<core::Device> cpu_devices =
                core::Device::GetAvailableCPUDevices();
        std::vector<core::Device> cuda_devices =
                core::Device::GetAvailableCUDADevices();

        std::vector<core::Device> devices;
        if (!cpu_devices.empty()) {
            devices.push_back(cpu_devices[0]);
        }
        if (!cuda_devices.empty()) {
            devices.push_back(cuda_devices[0]);
        }

        return devices;
    }
};

class PermuteDevicesWithSYCL : public testing::TestWithParam<core::Device> {
public:
    static std::vector<core::Device> TestCases() {
        std::vector<core::Device> devices = PermuteDevices::TestCases();

        std::vector<core::Device> sycl_cpu_devices =
                core::Device::GetAvailableSYCLCPUDevices();
        std::vector<core::Device> sycl_gpu_devices =
                core::Device::GetAvailableSYCLGPUDevices();

        if (!sycl_cpu_devices.empty()) {
            devices.push_back(sycl_cpu_devices[0]);
        }
        if (!sycl_gpu_devices.empty()) {
            devices.push_back(sycl_gpu_devices[0]);
        }

        return devices;
    }
};

class PermuteDevicePairs
    : public testing::TestWithParam<std::pair<core::Device, core::Device>> {
public:
    static std::vector<std::pair<core::Device, core::Device>> TestCases() {
        std::vector<core::Device> cpu_devices =
                core::Device::GetAvailableCPUDevices();
        std::vector<core::Device> cuda_devices =
                core::Device::GetAvailableCUDADevices();
        cuda_devices.resize(
                std::min(static_cast<size_t>(2), cuda_devices.size()));
        std::vector<core::Device> devices;
        devices.insert(devices.end(), cpu_devices.begin(), cpu_devices.end());
        devices.insert(devices.end(), cuda_devices.begin(), cuda_devices.end());

        std::vector<std::pair<core::Device, core::Device>> device_pairs;
        // Self-pairs.
        for (size_t i = 0; i < devices.size(); i++) {
            device_pairs.push_back({devices[i], devices[i]});
        }
        // Cross-pairs (bidirectional).
        for (size_t i = 0; i < devices.size(); i++) {
            for (size_t j = 0; j < devices.size(); j++) {
                if (i != j) {
                    device_pairs.push_back({devices[i], devices[j]});
                }
            }
        }

        return device_pairs;
    }
};

class PermuteSizesDefaultStrides
    : public testing::TestWithParam<
              std::pair<core::SizeVector, core::SizeVector>> {
public:
    static std::vector<std::pair<core::SizeVector, core::SizeVector>>
    TestCases() {
        return {
                {{}, {}},
                {{0}, {1}},
                {{0, 0}, {1, 1}},
                {{0, 1}, {1, 1}},
                {{1, 0}, {1, 1}},
                {{1}, {1}},
                {{1, 2}, {2, 1}},
                {{1, 2, 3}, {6, 3, 1}},
                {{4, 3, 2}, {6, 2, 1}},
                {{2, 0, 3}, {3, 3, 1}},
        };
    }
};

class TensorSizes : public testing::TestWithParam<int64_t> {
public:
    static std::vector<int64_t> TestCases() {
        std::vector<int64_t> tensor_sizes{
                0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
        // clang-format off
        std::vector<int64_t> large_sizes{
                (1 << 6 ) - 1, (1 << 6 ), (1 << 6 ) + 1,
                (1 << 10) - 6, (1 << 10), (1 << 10) + 6,
                (1 << 15) - 7, (1 << 15), (1 << 15) + 7,
                (1 << 20) - 1, (1 << 20), (1 << 20) + 1,
                (1 << 25) - 2, (1 << 25), (1 << 25) + 2, // ~128MB for float32
        };
        // clang-format on
        tensor_sizes.insert(tensor_sizes.end(), large_sizes.begin(),
                            large_sizes.end());
        return tensor_sizes;
    }
};

}  // namespace tests
}  // namespace open3d
