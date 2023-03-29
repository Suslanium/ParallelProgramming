
#include <vector>
#include <algorithm>

void openMpNegativeFilter(std::vector<unsigned char> &pixelChannels) {
    int i;
#pragma omp parallel for shared(pixelChannels) private(i) default(none)
    for (i = 0; i < pixelChannels.size(); i++) {
        pixelChannels[i] = 255 - pixelChannels[i];
    }
}

void openMpMedianFilter(std::vector<unsigned char> &output, std::vector<unsigned char> &input, int width, int height) {
    int i, j;
#pragma omp parallel for shared(input, output, width, height) private(i, j) default(none) collapse(2)
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            std::vector<unsigned char> windowR(225);
            std::vector<unsigned char> windowG(225);
            std::vector<unsigned char> windowB(225);
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
            std::sort(windowR.begin(), std::next(windowR.begin(), windowSize));
            output[i * width * 3 + j * 3] = windowR[windowSize / 2];
            std::sort(windowG.begin(), std::next(windowG.begin(), windowSize));
            output[i * width * 3 + j * 3 + 1] = windowG[windowSize / 2];
            std::sort(windowB.begin(), std::next(windowB.begin(), windowSize));
            output[i * width * 3 + j * 3 + 2] = windowB[windowSize / 2];
        }
    }
}
