#pragma GCC target("avx512f")
#pragma GCC target("avx512bw")
#pragma GCC target("avx512vl")

#include <algorithm>
#include <vector>
#include <x86intrin.h>

void vectorizedNegativeFilter(std::vector<unsigned char> &pixelsRGB) {
    __m512i maskReg = _mm512_set1_epi8(255);
    int i;
    for (i = 0; i < pixelsRGB.size() - 64; i += 64) {
        __m512i reg = _mm512_loadu_si512((__m512i *) &pixelsRGB[i]);
        __m512i subtractReg = _mm512_sub_epi8(maskReg, reg);
        _mm512_storeu_si512((__m512i *) &pixelsRGB[i], subtractReg);
    }
    for (; i < pixelsRGB.size(); i++) {
        pixelsRGB[i] = 255 - pixelsRGB[i];
    }
}

unsigned char threeElementMedian(const unsigned char &firstElement, const unsigned char &secondElement,
                                 const unsigned char &thirdElement) {
    return std::max(std::min(firstElement, secondElement),
                    std::min(std::max(firstElement, secondElement), thirdElement));
}

unsigned char fiveElementMedian(const unsigned char &firstElement, const unsigned char &secondElement,
                                const unsigned char &thirdElement, const unsigned char &fourthElement,
                                const unsigned char &fifthElement) {
    unsigned char temp1 = std::max(std::min(firstElement, secondElement), std::min(thirdElement, fourthElement));
    unsigned char temp2 = std::min(std::max(firstElement, secondElement), std::max(thirdElement, fourthElement));
    return threeElementMedian(fifthElement, temp1, temp2);
}

unsigned char vectorized25MedianChannel(std::vector<unsigned char> &windowChannel) {
    for (int i = 0; i < 5; i++) {
        windowChannel[i] = fiveElementMedian(windowChannel[i * 5], windowChannel[i * 5 + 1], windowChannel[i * 5 + 2],
                                             windowChannel[i * 5 + 3], windowChannel[i * 5 + 4]);
    }
    return fiveElementMedian(windowChannel[0], windowChannel[1], windowChannel[2], windowChannel[3], windowChannel[4]);
}

unsigned char vectorized75MedianChannel(std::vector<unsigned char> &windowChannel) {
    //0-24
    __m256i part1 = _mm256_loadu_epi8((__m256i *) &windowChannel[0]);
    //25-49
    __m256i part2 = _mm256_loadu_epi8((__m256i *) &windowChannel[25]);
    //50-74
    __m256i part3 = _mm256_loadu_epi8((__m256i *) &windowChannel[50]);
    //Three parts median
    __m256i temp1 = _mm256_min_epu8(part1, part2);
    __m256i temp2 = _mm256_max_epu8(part1, part2);
    __m256i temp3 = _mm256_min_epu8(temp2, part3);
    __m256i median = _mm256_max_epu8(temp1, temp3);
    //Write 25-element median back to vector
    _mm256_storeu_epi8((__m256i *) &windowChannel[0], median);
    return vectorized25MedianChannel(windowChannel);
}

unsigned char vectorized225MedianChannel(std::vector<unsigned char> &windowChannel) {
    //0-63
    __m512i part1 = _mm512_loadu_epi8((__m512i *) &windowChannel[0]);
    //64-74
    __m128i part1Remainder = _mm_loadu_epi8((__m128i *) &windowChannel[64]);
    //75-138
    __m512i part2 = _mm512_loadu_epi8((__m512i *) &windowChannel[75]);
    //139-149
    __m128i part2Remainder = _mm_loadu_epi8((__m128i *) &windowChannel[139]);
    //150-213
    __m512i part3 = _mm512_loadu_epi8((__m512i *) &windowChannel[150]);
    //214-224
    __m128i part3Remainder = _mm_loadu_epi8((__m128i *) &windowChannel[214]);
    //First three parts median
    __m512i temp1 = _mm512_min_epu8(part1, part2);
    __m512i temp2 = _mm512_max_epu8(part1, part2);
    __m512i temp3 = _mm512_min_epu8(temp2, part3);
    __m512i median = _mm512_max_epu8(temp1, temp3);
    //First three parts remainder median
    __m128i remainderTemp1 = _mm_min_epu8(part1Remainder, part2Remainder);
    __m128i remainderTemp2 = _mm_max_epu8(part1Remainder, part2Remainder);
    __m128i remainderTemp3 = _mm_min_epu8(remainderTemp2, part3Remainder);
    __m128i medianRemainder = _mm_max_epu8(remainderTemp1, remainderTemp3);
    //Write 75-element median back to the vector
    //0-63
    _mm512_storeu_epi8((__m512i *) &windowChannel[0], median);
    //64-74
    _mm_storeu_epi8((__m128i *) &windowChannel[64], medianRemainder);
    return vectorized75MedianChannel(windowChannel);
}

void
vectorizedMedianFilter(std::vector<unsigned char> &output, std::vector<unsigned char> &input, int width, int height) {
    int i, j;
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            std::vector<unsigned char> windowR(231);
            std::vector<unsigned char> windowG(231);
            std::vector<unsigned char> windowB(231);
            unsigned char windowSize = 0;
            for (int xPos = i - 7; xPos <= i + 7; xPos++) {
                for (int yPos = j - 7; yPos <= j + 7; yPos++) {
                    if (xPos >= 0 && xPos < height && yPos >= 0 && yPos < width) {
                        windowR[windowSize] = input[xPos * width * 3 + yPos * 3];
                        windowG[windowSize] = input[xPos * width * 3 + yPos * 3 + 1];
                        windowB[windowSize] = input[xPos * width * 3 + yPos * 3 + 2];
                        windowSize++;
                    }
                }
            }
            if (windowSize == 225) {
                output[i * width * 3 + j * 3] = vectorized225MedianChannel(windowR);
                output[i * width * 3 + j * 3 + 1] = vectorized225MedianChannel(windowG);
                output[i * width * 3 + j * 3 + 2] = vectorized225MedianChannel(windowB);
            } else {
                std::sort(windowR.begin(), std::next(windowR.begin(), windowSize));
                output[i * width * 3 + j * 3] = windowR[windowSize / 2];
                std::sort(windowG.begin(), std::next(windowG.begin(), windowSize));
                output[i * width * 3 + j * 3 + 1] = windowG[windowSize / 2];
                std::sort(windowB.begin(), std::next(windowB.begin(), windowSize));
                output[i * width * 3 + j * 3 + 2] = windowB[windowSize / 2];
            }
        }
    }
}