#pragma GCC target("avx512f")
#pragma GCC target("avx512bw")
#pragma GCC target("avx512vl")

#include <iostream>
#include <chrono>
#include <lodepng.h>
#include <algorithm>
#include <vector>
#include <x86intrin.h>
#include <CL/cl.h>
#include "sequential.cpp"
#include "openmp.cpp"
#include "vectorized.cpp"
#include "opencl.cpp"

std::vector<unsigned char> decodeImageVector(const std::string &fileName, unsigned &width, unsigned &height) {
    std::vector<unsigned char> result;
    unsigned error = lodepng::decode(result, width, height, fileName, LCT_RGB);
    if (error) {
        std::cout << "Image decoding error " << error << ": " << lodepng_error_text(error) << std::endl;
    }
    return result;
}

void encodeImageVector(const std::string &fileName, const std::vector<unsigned char> &pixels, unsigned width,
                       unsigned height) {
    unsigned error = lodepng::encode(fileName, pixels, width, height, LCT_RGB);
    if (error) {
        std::cout << "Image encoding error " << error << ": " << lodepng_error_text(error) << std::endl;
    }
}

std::vector<unsigned char> measureExecutionTimeMsNegative(std::vector<unsigned char> image,
                                                          void (*filter)(std::vector<unsigned char> &)) {
    auto t1 = std::chrono::high_resolution_clock::now();
    filter(image);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << ms_int.count() << " microseconds" << std::endl;
    return image;
}

std::vector<unsigned char> measureExecutionTimeMsMedian(std::vector<unsigned char> imageToCheck, int width, int height,
                                                        void (*filter)(std::vector<unsigned char> &,
                                                                       std::vector<unsigned char> &, int, int)) {
    std::vector<unsigned char> output(imageToCheck.size());
    auto t1 = std::chrono::high_resolution_clock::now();
    filter(output, imageToCheck, width, height);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << ms_int.count() << " microseconds" << std::endl;
    return output;
}

void singleIteration(std::vector<std::string>& filePaths, bool saveFiles, int iterationNumber, cl_command_queue queue,
                     cl_context context) {
    std::cout << "Iteration #" << iterationNumber << std::endl;
    for (std::string &path: filePaths) {
        unsigned width, height;
        std::vector<unsigned char> image = decodeImageVector(path, width, height);
        std::cout << "  " << path << ": " << std::endl;
        std::cout << "    Negative: " << std::endl;
        std::cout << "        Sequential: ";
        std::vector<unsigned char> negative_seq = measureExecutionTimeMsNegative(image, sequentialNegativeFilter);
        std::cout << "        OpenMP: ";
        std::vector<unsigned char> negative_omp = measureExecutionTimeMsNegative(image, openMpNegativeFilter);
        std::cout << "        Vectorized: ";
        std::vector<unsigned char> negative_vec = measureExecutionTimeMsNegative(image, vectorizedNegativeFilter);
        std::cout << "        OpenCL: ";
        std::vector<unsigned char> negative_ocl = invoke_opencl_negative(queue, context, image);
        std::cout << "    Median: " << std::endl;
        std::cout << "        Sequential: ";
        std::vector<unsigned char> median_seq = measureExecutionTimeMsMedian(image, width, height,
                                                                             sequentialMedianFilter);
        std::cout << "        OpenMP: ";
        std::vector<unsigned char> median_omp = measureExecutionTimeMsMedian(image, width, height, openMpMedianFilter);
        std::cout << "        Vectorized: ";
        std::vector<unsigned char> median_vec = measureExecutionTimeMsMedian(image, width, height,
                                                                             vectorizedMedianFilter);
        std::cout << "        OpenCL: ";
        std::vector<unsigned char> median_ocl = invoke_opencl_median(queue, context, image, width, height);
        if (saveFiles) {
            std::string cutPath = path.substr(0, path.length() - 4);
            encodeImageVector(cutPath + "_neg_seq.png", negative_seq, width, height);
            encodeImageVector(cutPath + "_neg_omp.png", negative_omp, width, height);
            encodeImageVector(cutPath + "_neg_vec.png", negative_vec, width, height);
            encodeImageVector(cutPath + "_neg_ocl.png", negative_ocl, width, height);
            encodeImageVector(cutPath + "_med_seq.png", median_seq, width, height);
            encodeImageVector(cutPath + "_med_omp.png", median_omp, width, height);
            encodeImageVector(cutPath + "_med_vec.png", median_vec, width, height);
            encodeImageVector(cutPath + "_med_ocl.png", median_ocl, width, height);
        }
    }
}

int main() {
    std::string fileName = R"(C:\Users\Ruslan\CLionProjects\ParallelProgramming\Pictures\300x300.png)";
    cl_device_id device = create_device();
    cl_context context = create_context(device);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, NULL);
    std::cout << "Enter amount of files: ";
    int amount;
    std::cin >> amount;
    std::cout << "Enter PNG file paths: " << std::endl;
    std::vector<std::string> filePaths(amount);
    for (int i = 0; i < amount; i++) {
        std::cin >> filePaths[i];
    }
    int iterationCount;
    std::cout << "Enter iteration count: " << std::endl;
    std::cin >> iterationCount;
    std::cout << "Save files on first iteration? y/n" << std::endl;
    char saveInput;
    bool saveOnFirstIter = false;
    std::cin >> saveInput;
    if(saveInput == 'y') {
        saveOnFirstIter = true;
    }
    std::cout << std::endl << "### Benchmark started" << std::endl;
    for (int i = 0; i < iterationCount; i++) {
        singleIteration(filePaths, (saveOnFirstIter && i==0), i + 1, queue, context);
    }
    std::cout << "### Benchmark finished" << std::endl;
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}
