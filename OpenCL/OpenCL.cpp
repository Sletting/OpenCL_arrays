#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY
#include <CL/cl2.hpp>
#include <CL/opencl.hpp>
#include <fstream>
#include <iostream>

cl::Program CreateProgram(const std::string& file) 
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    _ASSERT(platforms.size() > 0);

    auto platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    _ASSERT(devices.size() > 0);

    auto device = devices.front();
    auto vendor = device.getInfo<CL_DEVICE_VENDOR>();
    auto version = device.getInfo<CL_DEVICE_VERSION>();

    std::ifstream kernelfile(file);
    std::string src(std::istreambuf_iterator<char>(kernelfile), (std::istreambuf_iterator<char>()));

    cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));
    cl::Context context(device);
    cl::Program program(context, sources);
    auto err = program.build("-cl-std=CL1.2");

    return program;
}

void helloWorld()
{
    auto program = CreateProgram("kernelfile.cl");
    auto context = program.getInfo<CL_PROGRAM_CONTEXT>();
    auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
    auto& device = devices.front();

    char buf[16];
    cl::Buffer memBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(buf));
    cl::Kernel kernel(program, "kernelfile");
    kernel.setArg(0, memBuf);

    cl::CommandQueue queue(context, device);
    queue.enqueueTask(kernel);
    queue.enqueueReadBuffer(memBuf, CL_TRUE, 0, sizeof(buf), buf);

    std::cout << buf;
}

void ProcessArray()
{
    auto program = CreateProgram("ProcessArray.cl");
    auto context = program.getInfo<CL_PROGRAM_CONTEXT>();
    auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
    auto& device = devices.front();

    std::vector<int> vec(1024);
    // Use CPU to fill.
    //std::fill(vec.begin(), vec.end(), 1);

    cl::Buffer inBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(int) * vec.size(), vec.data());
    cl::Buffer outBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(int) * vec.size());
    cl::Kernel kernel(program, "ProcessArray");
    kernel.setArg(0, inBuf);
    kernel.setArg(1, outBuf);
    cl::CommandQueue queue(context, device);
  
    // Use GPU to fill buffer. Skip first 10 and fill rest with 3.
    queue.enqueueFillBuffer(inBuf, 3, sizeof(int) * 10, sizeof(int) * (vec.size() - 10));
    
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vec.size()));
    queue.enqueueReadBuffer(outBuf, CL_FALSE, 0, sizeof(int) * vec.size(), vec.data());
    
    cl::finish();

    std::cin.get();
}


int main()
{
    ProcessArray();
}