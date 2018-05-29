#ifndef _CONV_COMMON_H_
#define _CONV_COMMON_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

struct benchmark {
    float compute_time;
    float transfer_time;
    float total_time;
};

inline float* create_gaussian_kernel(float sigma, size_t kernel_size)
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

inline void print_array(float* array , uint size)
{
  for (uint y = 0 ; y < size ; ++y)
  {
	  std::cout << "array[" << y << "] = " << array[y] << std::endl;
  }
}

inline void print_2d_array(int** array , uint width, uint height)
{
  for (uint y = 0 ; y < height ; ++y)
  {
	for(uint x = 0 ; x < width ; ++x)
	{
	  std::cout << "array[" << y << "]" << "[" << x << "] = " << array[y][x] << std::endl;
	}
  }
}

inline unsigned char* load_ppm(char* file_name, uint &width, uint &height)
{
  std::ifstream file(file_name);
  std::string magic_number;
  uint max_value;
  file >> magic_number >> width >> height >> max_value;
  if (magic_number != "P6")
	throw std::invalid_argument("Current PPM Image format is " + magic_number  + " and must be P6");
  if (max_value > 255)
	throw std::invalid_argument("Max value is " + std::to_string(max_value) + "but it should not be > 255");

  size_t size = width * height * 3;
  unsigned char* data = new unsigned char[size];
  file.read((char*)(&data[0]), size);
  return data;
}

inline void write_ppm(char* file_name, unsigned char* data, int width, int height)
{
  std::ofstream file(file_name);
  file << "P6" << "\n"
	   << width << "\n"
	   << height << "\n"
	   << 255 << "\n";
  size_t size = width * height * 3;
  file.write((char*)(&data[0]), size);
}

inline int** random_convolution_kernel(uint width, uint height)
{
  int** conv = new int*[height];
  for(uint i = 0 ; i < height ; ++i)
    conv[i] = new int[width];

  for (uint y = 0 ; y < height ; ++y)
  {
	for(uint x = 0 ; x < width ; ++x)
	{
	  conv[y][x] = rand() % 10 - 5;
	}
  }

  return conv;
}

inline float* uchar_to_float(unsigned char* data, uint size)
{
  float* float_data = new float[size];
  for (uint i = 0 ; i < size ; ++i)
  {
	float_data[i] = static_cast<float>(data[i]);
  }
  return float_data;
}

inline unsigned char* float_to_uchar(float* data, uint size)
{
  unsigned char* uchar_data = new unsigned char[size];
  for (uint i = 0 ; i < size ; i++)
  {
	float value = data[i];
	if (value < 0) value = 0;
	if (value > 255) value = 255;
	uchar_data[i] = static_cast<float>(data[i]);
  }
  return uchar_data;
}

inline unsigned char* rgb_to_gray(unsigned char* data, uint width, uint height)
{
  unsigned char* data_gray = new unsigned char[width * height];
  int cpt = 0;
  for (uint i = 0 ; i < width * height * 3; i+=3)
  {
	data_gray[cpt++] = (data[i] + data[i+1] + data[i+2]) / 3;
  }
  
  return data_gray;
}

inline unsigned char* gray_to_rgb(unsigned char* data, uint width, uint height)
{
  unsigned char* data_rgb = new unsigned char[width * height * 3];
  int cpt = 0;
  for (uint i = 0 ; i < width * height; ++i)
  {
	data_rgb[cpt++] = data[i];
	data_rgb[cpt++] = data[i];
	data_rgb[cpt++] = data[i];
  }
  
  return data_rgb;
}

inline float* generate_random_image(uint width, uint height)
{
  float* img = new float[width * height];
  for (int i = 0 ; i < width * height ; ++i)
  {
	img[i] = rand()%256;
  }
  return img;
}

#endif
