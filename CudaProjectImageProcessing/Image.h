#pragma once

#include <string>

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

	Pixel(uint8_t red, uint8_t green, uint8_t blue) : red(red), green(green), blue(blue)
	{
		intensity = 0.2126 * red + 0.7152 * green + 0.0722 * blue;
	}
	Pixel(uint8_t intensity = 0) : Pixel(intensity, intensity, intensity) {}

};

class Image
{
public:
	static constexpr int DEFAULT_INTENSITY = 0;

private:
	uint8_t* image;
	int width;
	int height;
	int bpp;

public:
	Image() = delete;
	Image(const Image& copy) : width(copy.width), height(copy.height), bpp(copy.bpp)
	{
		image = new uint8_t[width * height * bpp];
		memcpy(image, copy.image, sizeof(uint8_t) * width * height * bpp);
	}
	Image(std::string path);
	Image(int width, int height, int bpp = 3);
	~Image();
	int get_width() const;
	int get_height() const;
	Pixel get_pixel_at(int x, int y) const;
	void set_pixel_at(int x, int y, Pixel pixel);
	void write(std::string path) const;
	uint8_t* get_raw_data() const;
	int get_bpp() const;

	Image& operator=(const Image& ref)
	{
		delete image;
		width = ref.width;
		height = ref.height;
		bpp = ref.bpp;
		
		image = new uint8_t[width * height * bpp];
		memcpy(image, ref.image, sizeof(uint8_t) * width * height * bpp);

		return *this;
	}

private:
	int get_offset(int x, int y) const;
	bool is_color() const;
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