
#include <iostream>
#include <algorithm>
#include <chrono>
#include <vector>
#include <CL/cl.h>

const std::string negative_kernel = "__kernel \n"
                                    "void calculate_negative( \n"
                                    "               __global uchar * result, \n"
                                    "               int VECTOR_SIZE) \n"
                                    "{ \n"
                                    "   int index = get_global_id(0); \n"
                                    "   if(index >= VECTOR_SIZE) return; \n"
                                    "   result[index] = 255 - result[index]; \n"
                                    "} \n";

const std::string median_kernel = "void merge_series(uchar* arr, uchar* temp_array, int s1start, int s1end, int s2start, int s2end) { \n"
                                  "    if(s1end==s2end) \n"
                                  "        return; \n"
                                  "    int i=s1start,i2=s1start; \n"
                                  "    while (s1start<=s1end || s2start<=s2end) { \n"
                                  "        int next_index = (s1start > s1end) ? s2start++ : ((s2start > s2end) ? s1start++ : ((arr[s1start] > arr[s2start]) ? s2start++ : s1start++)); \n"
                                  "        temp_array[i++] = arr[next_index]; \n"
                                  "    } \n"
                                  "    for (; i2 <= s2end; i2++) { \n"
                                  "        arr[i2] = temp_array[i2]; \n"
                                  "    } \n"
                                  "} \n"
                                  " \n"
                                  "void merge_sort(uchar* arr, uchar* temp, int size) { \n"
                                  "    int series_size=1; \n"
                                  "    int s1s,s1e,s2s,s2e; \n"
                                  "    while (series_size < size) { \n"
                                  "        for (int i = 0; i < size; i += 2 * series_size) { \n"
                                  "            s1s = i; \n"
                                  "            s1e = min(i + series_size - 1 , size - 1); \n"
                                  "            s2s = min(i + series_size, size - 1); \n"
                                  "            s2e = min(i + 2 * series_size - 1, size - 1); \n"
                                  "            merge_series(arr, temp, s1s, s1e, s2s, s2e); \n"
                                  "        } \n"
                                  "        series_size *= 2; \n"
                                  "    } \n"
                                  "} \n"
                                  " \n"
                                  "__kernel \n"
                                  "void calculate_median( \n"
                                  "               __global uchar * input, \n"
                                  "               __global uchar * output, \n"
                                  "               int width, \n"
                                  "               int height) \n"
                                  "{ \n"
                                  "   int posX = get_global_id(0); \n"
                                  "   int posY = get_global_id(1); "
                                  "   if(posX > height || posY > width) return; \n"
                                  "   uchar windowR[225]; \n"
                                  "   uchar windowG[225]; \n"
                                  "   uchar windowB[225]; \n"
                                  "   uchar tempArray[225]; \n"
                                  "   uchar windowSize = 0; \n"
                                  "   for (int windowPosX = posX - 7; windowPosX <= posX + 7; windowPosX++) { \n"
                                  "       for (int windowPosY = posY - 7; windowPosY <= posY + 7; windowPosY++) { \n"
                                  "           if (windowPosX >= 0 && windowPosY >= 0 && windowPosX < height && windowPosY < width) { \n"
                                  "               windowR[windowSize] = input[windowPosX * width * 3 + windowPosY * 3]; \n"
                                  "               windowG[windowSize] = input[windowPosX * width * 3 + windowPosY * 3 + 1]; \n"
                                  "               windowB[windowSize] = input[windowPosX * width * 3 + windowPosY * 3 + 2]; \n"
                                  "               windowSize++; \n"
                                  "           } \n"
                                  "       } \n"
                                  "   } \n"
                                  "   merge_sort(windowR, tempArray, windowSize); \n"
                                  "   merge_sort(windowG, tempArray, windowSize); \n"
                                  "   merge_sort(windowB, tempArray, windowSize); \n"
                                  "   output[posX * width * 3 + posY * 3] = windowR[windowSize / 2]; \n"
                                  "   output[posX * width * 3 + posY * 3 + 1] = windowG[windowSize / 2]; \n"
                                  "   output[posX * width * 3 + posY * 3 + 2] = windowB[windowSize / 2]; \n"
                                  "} \n";

cl_device_id create_device() {
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_int err = 0;
    err |= clGetPlatformIDs(1, &platform_id, NULL);
    err |= clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (err) throw;
    return device_id;
}

cl_context create_context(cl_device_id device) {
    cl_int err = 0;
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err) throw;
    return context;
}

cl_program build_program(cl_context context, const std::string &src) {
    cl_int err = 0;
    const char *src_text = src.data();
    size_t src_length = src.size();
    cl_program program = clCreateProgramWithSource(context, 1, &src_text, &src_length, &err);
    err |= clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err) throw;
    return program;
}

int align_to_fit(int global_size, int local_group_size) {
    return (global_size + local_group_size - 1) / local_group_size * local_group_size;
}

void
invoke_negative_kernel(cl_kernel kernel, cl_command_queue queue, cl_mem filled_buffer, cl_uchar *result, int size) {
    cl_int err = 0;
    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &filled_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_int), &size);
    size_t local_size[1] = {256};
    size_t size_horizontal = align_to_fit(size, local_size[0]);
    size_t global_size[1] = {size_horizontal};
    err |= clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_size, local_size, 0, NULL, NULL);
    if (err) throw;
    auto t1 = std::chrono::high_resolution_clock::now();
    clFinish(queue);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << ms_int.count() << " microseconds" << std::endl;
    err |= clEnqueueReadBuffer(queue, filled_buffer, CL_TRUE, 0, sizeof(cl_uchar) * size, result, 0, NULL, NULL);
    if (err) throw;
    clFinish(queue);
}

std::vector<unsigned char>
invoke_opencl_negative(cl_command_queue queue, cl_context context, std::vector<unsigned char> input) {
    cl_int err = 0;
    cl_program program = build_program(context, negative_kernel);
    cl_kernel kernel = clCreateKernel(program, "calculate_negative", &err);
    if (err) throw;
    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uchar) * input.size(),
                                   input.data(), NULL);
    invoke_negative_kernel(kernel, queue, buffer, input.data(), input.size());
    clReleaseKernel(kernel);
    clReleaseMemObject(buffer);
    clReleaseProgram(program);
    return input;
}

void
invoke_median_kernel(cl_kernel kernel, cl_command_queue queue, cl_mem input_buffer, cl_mem output_buffer,
                     cl_uchar *result, int width, int height) {
    cl_int err = 0;
    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_int), &width);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_int), &height);
    size_t local_size[2] = {256, 1};
    size_t size_horizontal = align_to_fit(height, local_size[0]);
    size_t size_vertical = align_to_fit(width, local_size[1]);
    size_t global_size[2] = {size_horizontal, size_vertical};
    auto t1 = std::chrono::high_resolution_clock::now();
    err |= clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
    if (err) throw;
    clFinish(queue);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << ms_int.count() << " microseconds" << std::endl;
    err |= clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, sizeof(cl_uchar) * width * height * 3, result, 0, NULL,
                               NULL);
    if (err) throw;
    clFinish(queue);
}

std::vector<unsigned char>
invoke_opencl_median(cl_command_queue queue, cl_context context, std::vector<unsigned char> input, int width,
                     int height) {
    cl_int err = 0;
    cl_program program = build_program(context, median_kernel);
    cl_kernel kernel = clCreateKernel(program, "calculate_median", &err);
    if (err) throw;
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(cl_uchar) * input.size(), input.data(), NULL);
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_uchar) * input.size(), NULL, NULL);
    invoke_median_kernel(kernel, queue, input_buffer, output_buffer, input.data(), width, height);
    clReleaseKernel(kernel);
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseProgram(program);
    return input;
}
