#pragma once

#include <string>

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION   
#endif

#ifndef STB_IMAGE_WIRTE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif

/**
 * @brief 
 * The struct Pixel represents a pixel of color images.
 * If the image is grayscale, then red, green, blue contains intensity value.
 * intensity = 0.2126 * red + 0.7152 * green + 0.0722 * blue
*/
struct Pixel {
    uint8_t red;
    uint8_t green;
    uint8_t blue;
    uint8_t intensity;
};

class Image
{
public:
	Image() = delete;
	Image(std::string path);
    Image(const int width, const int height);
    ~Image();
    int get_width();
    int get_height();
    Pixel get_pixel_at(int x, int y) noexcept;
    void set_pixel_at(int x, int y, Pixel pixel) noexcept;
    void write(std::string path);
};

//Example Code:
//int main()
//{
//    Image image("image_path.png");
//    Pixel pixel = image.get_pixel_at(0, 0);
//    printf("%d %d %d %d", pixel.red, pixel.green, pixel.blue, pixel.alpha);
//    pixel.red = 0;
//    pixel.green = 0;
//    pixel.blue = 0;
//    pixel.intensity = 0;
//    image.write("output1.png");
//    // output1.png should be the same as the given image.
// 
//    image.set_pixel_at(0, 0, pixel);
//    image.write("output2.png");
//    // output2.png should be the image that 0,0 pixel is black and the remains are the same as the given image.
//
//    Image image2(32, 32);
//    image2.set_pixel_at(0, 0, pixel);
//    image2.write("output3.png");
//    // output3.png should be the image that 0,0 pixel is black and the remains are white.
//}