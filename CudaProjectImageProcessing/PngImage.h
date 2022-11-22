#pragma once

#include <png.h>
#include <string>
#include <format>
#include <stdint.h>
#include <exception>

struct Pixel32 {
    uint8_t red;
    uint8_t green;
    uint8_t blue;
    uint8_t alpha;
};

class PngImageException;

class PngImage
{
private:
    static const int DEFAULT_BIT_DEPTH = 8;

    int width;
    int height;
    png_byte colorType;
    png_byte bitDepth;
    int numberOfPasses;
    png_structp pStruct;
    png_infop pInfo;
    png_bytep* ppRowData;

public:
    PngImage() = delete;
	PngImage(std::string path) throw();
    PngImage(int width, int height);
    ~PngImage();
    Pixel32 getPixelAt(int x, int y) noexcept;
    void setPixelAt(int x, int y, Pixel32 pixel) noexcept;
    void write(std::string path) throw();

private:
    void read(std::string path) throw();
    void initMembers(png_structp pStruct, png_infop pInfo);
};

class PngImageException : public std::exception
{
private:
    std::string message;

public:
    PngImageException(std::string msg) : message(msg) {}

    const char* what() const noexcept override
    {
        return message.c_str();
    }
};