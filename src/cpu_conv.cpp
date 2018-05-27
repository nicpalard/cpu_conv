#include <iostream>
#include <math.h>

#include "../include/ImageUtils.hpp"

float* create_gaussian_kernel(float sigma, size_t kernel_size)
{
    float* kernel = new float[kernel_size * kernel_size];
    float mean = kernel_size/2;
    float sum = 0.0;
    for (int i = 0; i < kernel_size * kernel_size; ++i)
    {
        int x = i % kernel_size;
        int y = (i - x) / kernel_size % kernel_size;
        kernel[i] = exp( -0.5 * (pow((x-mean)/sigma, 2.0) + pow((y-mean)/sigma,2.0)) ) / (2 * M_PI * sigma * sigma);
        // Accumulate the kernel values
        sum += kernel[i];
    }
    return kernel;
}

float* cpu_1chan_conv(float* data, uint width, uint height, float* mask, uint mask_width, uint mask_height)
{
    float* out = new float[width * height];
    uint mask_hw = floor(mask_width / 2);
    uint mask_hh = floor(mask_height / 2);

    float sum = 0.0;
    float mask_sum = 0.0;
    for (uint x = 0 ; x < width ; ++x)
    {
        for (uint y = 0 ; y < height ; ++y)
        {
            for (uint mx = 0 ; mx < mask_width ; ++mx)
            {
                for (uint my = 0 ; my < mask_height ; ++my)
                {
                    int px = x + mx - mask_hw;
                    int py = y + my - mask_hh;
                    if (px < 0 || px >= width || py < 0 || py >= height)
                        continue;

                    float m_value = mask[mx + my * mask_width];
                    sum += m_value * data[px + py * width];
                    mask_sum += m_value;
                }
            }
            out[x + y * width] = sum / mask_sum;
        }
    }

    return out;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        return EXIT_FAILURE;
    }

    uint mask_width = 5, mask_height = 5;
    float* mask = create_gaussian_kernel(2, mask_width * mask_height);

    float* data = generate_random_image(2048, 2048, 1);
    float* out_data = cpu_1chan_conv(data, 2048, 2048, mask, mask_width, mask_height);
}
