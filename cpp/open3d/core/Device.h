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

#include <string>
#include <vector>

namespace open3d {
namespace core {

/// Device context specifying device type and device id.
/// For CPU, there is only one device with id 0.
class Device {
public:
    /// Type for device.
    enum class DeviceType {
        CPU = 0,
        CUDA = 1,
        SYCL_CPU = 3,  // SYCL host_selector(), not cpu_selector().
        SYCL_GPU = 4,  // SYCL gpu_selector().
    };

    /// Default constructor -> "CPU:0".
    Device() = default;

    /// Constructor with device specified.
    explicit Device(DeviceType device_type, int device_id);

    /// Constructor from device type string and device id.
    explicit Device(const std::string& device_type, int device_id);

    /// Constructor from string, e.g. "CUDA:0".
    explicit Device(const std::string& type_colon_id);

    bool operator==(const Device& other) const;

    bool operator!=(const Device& other) const;

    bool operator<(const Device& other) const;

    /// Returns true iff device type is CPU.
    bool IsCPU() const;

    /// Returns true iff device type is CUDA.
    bool IsCUDA() const;

    /// Returns true iff device type is SYCL_CPU or SYCL_GPU.
    bool IsSYCL() const;

    /// Returns true iff device type is SYCL_CPU.
    bool IsSYCLCPU() const;

    /// Returns true iff device type is SYCL_GPU.
    bool IsSYCLGPU() const;

    /// Returns string representation of device, e.g. "CPU:0", "CUDA:0".
    std::string ToString() const;

    /// Get device description.
    std::string GetDescription() const;

    /// Returns type of the device, e.g. DeviceType::CPU, DeviceType::CUDA.
    DeviceType GetType() const;

    /// Returns the device index (within the same device type).
    int GetID() const;

    /// Returns true if the device is available.
    bool IsAvailable() const;

    /// Returns a vector of available devices.
    static std::vector<Device> GetAvailableDevices();

    /// Returns a vector of available CPU device.
    static std::vector<Device> GetAvailableCPUDevices();

    /// Returns a vector of available CUDA device.
    static std::vector<Device> GetAvailableCUDADevices();

    /// Returns a vector of available SYCL device.
    static std::vector<Device> GetAvailableSYCLDevices();

    /// Returns a vector of available SYCL_CPU device.
    static std::vector<Device> GetAvailableSYCLCPUDevices();

    /// Returns a vector of available SYCL_GPU device.
    static std::vector<Device> GetAvailableSYCLGPUDevices();

    /// Print all available devices.
    static void PrintAvailableDevices();

protected:
    DeviceType device_type_ = DeviceType::CPU;
    int device_id_ = 0;
};

}  // namespace core
}  // namespace open3d

namespace std {
template <>
struct hash<open3d::core::Device> {
    std::size_t operator()(const open3d::core::Device& device) const {
        return std::hash<std::string>{}(device.ToString());
    }
};
}  // namespace std
