#include <iostream>
#include <iomanip>

#include <chrono>
#include <math.h>

#include "../include/conv_common.h"

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<double> fsec;
typedef std::chrono::milliseconds ms;

float* cpu_1chan_conv(float* data, uint width, uint height, float* mask, uint mask_width, uint mask_height, struct benchmark& benchmark)
{
    auto total_start = Time::now();
    float* out = new float[width * height];
    uint mask_hw = floor(mask_width / 2);
    uint mask_hh = floor(mask_height / 2);

    auto compute_start = Time::now();
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
    auto compute_end = Time::now();
    auto total_end = Time::now();

    benchmark.compute_time = fsec(compute_end - compute_start).count() * 1e3;
    benchmark.transfer_time = 0.0;
    benchmark.total_time = fsec(total_end - total_start).count() * 1e3;
    return out;
}

int main(int argc, char** argv)
{
    std::fstream csvfile;
    csvfile.open("bench.csv", std::fstream::in | std::fstream::out | std::fstream::app);
    if (!csvfile.is_open())
    {
        std::cerr << "Could not opencv bench.csv, benchmark results are not going to be saved" << std::endl;
    }

    uint mask_width = 5, mask_height = 5;
    float* mask = create_gaussian_kernel(2, mask_width * mask_height);

    //--------------- BENCH SETUP ----------------
    std::cout << std::endl
                << "** Starting benchmark **" << std::endl
                << "CPU Convolution" << std::endl
                << "---------------------------------------------" << std::endl
                << "Size\t\tSize (MB)\tCTime (ms)\tTTime (ms)\tTotal (ms)\tBandwidth (MB/s)" << std::endl
                << std::fixed << std::setprecision(3) << std::setfill('0');
    //--------------- BENCH SETUP ----------------

    if (csvfile.is_open())
        csvfile << "Size,Size (MB),Compute Time (ms),Transfer Time (ms),Total Time (ms),Bandwidth (MB/s)" << std::endl;

    struct benchmark benchmark;
    for (int N = 128 ; N <= 8192 ; N+=N)
    {
        float* data = generate_random_image(2048, 2048);
        unsigned int iterations = 10;
        double compute_time = 0, transfer_time = 0, total_time = 0;
        for (int i = 0 ; i < iterations ; ++i)
        {
            float* result = cpu_1chan_conv(data, N, N, mask, mask_width, mask_height, benchmark);
            compute_time += benchmark.compute_time;
            transfer_time += benchmark.transfer_time;
            total_time += benchmark.total_time;
        }

        double realsize = (N * N * sizeof(float)) / 1e6;
        compute_time /= (double)iterations;
        transfer_time /= (double)iterations;
        total_time /= (double)iterations;
        double bandwidth = realsize / (total_time / 1e3);
        std::cout << N << "x" << N << "\t"
                        << realsize << "\t\t"
                        << compute_time << "\t\t"
                        << transfer_time << "\t\t"
                        << total_time << "\t\t"
                        << bandwidth << std::endl;

        if (csvfile.is_open())
        {
            csvfile << N << "x" << N << ","
                    << realsize << ","
                    << compute_time << ","
                    << transfer_time << ","
                    << total_time << ","
                    << bandwidth << std::endl;
        }

    }
    csvfile.close();
    return EXIT_SUCCESS;
}
