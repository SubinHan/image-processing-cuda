#include "HsvOld.cuh"
#include <algorithm>

__device__ __host__ double rgb_max(double a, double b, double c) {
    return (a > b) ? (a > c ? a : c) : (b > c ? b : c);
}


__device__ __host__ double rgb_min(double a, double b, double c) {
    return ((a < b) ? (a < c ? a : c) : (b < c ? b : c));
}

void extract_color_cpu(
	uint8_t* image,
	int width,
	int height,
	int bpp,
	uint8_t* red,
	uint8_t* green,
	uint8_t* blue)
{
    for (int p = 0; p < width * height; p++) {
        const int offset = p * bpp;

        double r = (double)image[offset + 0] / 255.0;
        double g = (double)image[offset + 1] / 255.0;
        double b = (double)image[offset + 2] / 255.0;

        const double max = rgb_max(r, g, b);
        const double min = rgb_min(r, g, b);

        const double delta = max - min;

       double v = max * 100.0;

       double h;
        if (max == min)  //색상(Hue) 0 ~ 360, fmod(double a, double b) -> a % b 의 결과
            h = 0;
        else if (max == r)
            h = fmod((60 * ((g - b) / delta) + 360), 360.0);
        else if (max == g)
            h = fmod((60 * ((b - r) / delta) + 120), 360.0);
        else if (max == b)
            h = fmod((60 * ((r - g) / delta) + 240), 360.0);

        double s;
        if (max == 0) //채도(Saturation) 0 ~ 100,  100에 가까울수록 원색
            s = 0;
        else
            s = delta / max * 100.0;

        if ((h > 30.0 && h < 330.0) || s < 10) {
            double gray = 0.2126 * image[p * bpp] + 0.7152 * image[p * bpp + 1] + 0.0722 * image[p * bpp + 2];
            red[p * bpp + 0] = (int)gray;
            red[p * bpp + 1] = (int)gray;
            red[p * bpp + 2] = (int)gray;
        }
        else
        {
            red[p * bpp + 0] = image[p * bpp + 0];
            red[p * bpp + 1] = image[p * bpp + 1];
            red[p * bpp + 2] = image[p * bpp + 2];
        }

        if ((h > 150.0 || h < 90.0) || s < 10) {
            double gray = 0.2126 * image[p * bpp] + 0.7152 * image[p * bpp + 1] + 0.0722 * image[p * bpp + 2];
            green[p * bpp + 0] = (int)gray;
            green[p * bpp + 1] = (int)gray;
            green[p * bpp + 2] = (int)gray;
        }
        else
        {
            green[p * bpp + 0] = image[p * bpp + 0];
            green[p * bpp + 1] = image[p * bpp + 1];
            green[p * bpp + 2] = image[p * bpp + 2];
        }

        if ((h > 270.0 || h < 180.0) || s < 10) {
            double gray = 0.2126 * image[p * bpp] + 0.7152 * image[p * bpp + 1] + 0.0722 * image[p * bpp + 2];
            blue[p * bpp + 0] = (int)gray;
            blue[p * bpp + 1] = (int)gray;
            blue[p * bpp + 2] = (int)gray;
        }
        else
        {
            blue[p * bpp + 0] = image[p * bpp + 0];
            blue[p * bpp + 1] = image[p * bpp + 1];
            blue[p * bpp + 2] = image[p * bpp + 2];
        }
    }
}

void extract_color_gpu(
    uint8_t* image,
    int width,
    int height,
    int bpp,
    uint8_t* red,
    uint8_t* green,
    uint8_t* blue)
{
    const int buffer_size = sizeof(uint8_t) * width * height * bpp;
    const int image_size = width * height;

    uint8_t* d_image;
    cudaMalloc((void**)&d_image, buffer_size);
    cudaMemcpy(d_image, image, buffer_size, cudaMemcpyHostToDevice);

    uint8_t* d_red, *d_green, *d_blue;
    cudaMalloc((void**)&d_red, buffer_size);
    cudaMalloc((void**)&d_green, buffer_size);
    cudaMalloc((void**)&d_blue, buffer_size);

    int blocksx = ceil((image_size) / 256.0f);
    dim3 threads(256);
    dim3 grid(blocksx);

    extract_color_gpu0 << <grid, threads >> > (d_image, width, height, bpp, d_red, d_green, d_blue);

    cudaMemcpy(red, d_red, buffer_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(green, d_green, buffer_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(blue, d_blue, buffer_size, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_red);
    cudaFree(d_green);
    cudaFree(d_blue);
}

__global__ void extract_color_gpu0(
	uint8_t* image,
	int width,
	int height,
	int bpp,
	uint8_t* red,
	uint8_t* green,
	uint8_t* blue)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = gridDim.x * blockDim.x * ty + tx;

    const int size = width * height;

    if (tid >= size)
        return;

    const int offset = tid * bpp;

    double r = (double)image[offset + 0];
    double g = (double)image[offset + 1];
    double b = (double)image[offset + 2];
    r = r / 255.0;
    g = g / 255.0;
    b = b / 255.0;

    const double max = rgb_max(r, g, b); // maximum of r, g, b
    const double min = rgb_min(r, g, b); // minimum of r, g, b

    const double delta = max - min;

    double v = max * 100.0 / 100.0; //명도(Value) 0 ~ 100  100에 가까울수록 밝은색

    double h;
    if (max == min)  //색상(Hue) 0 ~ 360, fmod(double a, double b) -> a % b 의 결과
        h = 0;
    else if (max == r)
        h = fmod((60 * ((g - b) / delta) + 360), 360.0);
    else if (max == g)
        h = fmod((60 * ((b - r) / delta) + 120), 360.0);
    else if (max == b)
        h = fmod((60 * ((r - g) / delta) + 240), 360.0);

    double s;
    if (max == 0) //채도(Saturation) 0 ~ 100,  100에 가까울수록 원색
        s = 0;
    else
        s = delta / max * 100.0;

    if ((h > 30.0 && h < 330.0) || s < 10) {
        double gray = 0.2126 * image[tid * bpp] + 0.7152 * image[tid * bpp + 1] + 0.0722 * image[tid * bpp + 2];
        red[tid * bpp + 0] = (int)gray;
        red[tid * bpp + 1] = (int)gray;
        red[tid * bpp + 2] = (int)gray;
    }
    else
    {
        red[tid * bpp + 0] = image[tid * bpp + 0];
        red[tid * bpp + 1] = image[tid * bpp + 1];
        red[tid * bpp + 2] = image[tid * bpp + 2];
    }

    if ((h > 150.0 || h < 90.0) || s < 10) {
        double gray = 0.2126 * image[tid * bpp] + 0.7152 * image[tid * bpp + 1] + 0.0722 * image[tid * bpp + 2];
        green[tid * bpp + 0] = (int)gray;
        green[tid * bpp + 1] = (int)gray;
        green[tid * bpp + 2] = (int)gray;
    }
    else
    {
        green[tid * bpp + 0] = image[tid * bpp + 0];
        green[tid * bpp + 1] = image[tid * bpp + 1];
        green[tid * bpp + 2] = image[tid * bpp + 2];
    }

    if ((h > 270.0 || h < 180.0) || s < 10) {
        double gray = 0.2126 * image[tid * bpp] + 0.7152 * image[tid * bpp + 1] + 0.0722 * image[tid * bpp + 2];
        blue[tid * bpp + 0] = (int)gray;
        blue[tid * bpp + 1] = (int)gray;
        blue[tid * bpp + 2] = (int)gray;
    }
    else
    {
        blue[tid * bpp + 0] = image[tid * bpp + 0];
        blue[tid * bpp + 1] = image[tid * bpp + 1];
        blue[tid * bpp + 2] = image[tid * bpp + 2];
    }
}