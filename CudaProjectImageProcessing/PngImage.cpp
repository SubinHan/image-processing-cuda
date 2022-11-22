#include "PngImage.h"


PngImage::PngImage(std::string path) throw()
{
	read(path);
}

PngImage::PngImage(int width, int height)
{
    png_byte** row_pointers = NULL;

    pStruct = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!pStruct) {
        png_destroy_write_struct(&pStruct, &pInfo);
        throw PngImageException("");
    }

    pInfo = png_create_info_struct(pStruct);
    if (pInfo == NULL) {
        png_destroy_write_struct(&pStruct, &pInfo);
        throw PngImageException("");
    }

    if (setjmp(png_jmpbuf(pStruct))) {
        png_destroy_write_struct(&pStruct, &pInfo);
        throw PngImageException("");
    }


    /* Set image attributes. */

    png_set_IHDR(pStruct,
        pInfo,
        width,
        height,
        DEFAULT_BIT_DEPTH,
        PNG_COLOR_TYPE_RGBA,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT);

    initMembers(pStruct, pInfo);

    /* Initialize rows of PNG. */

    ppRowData = (png_byte**)png_malloc(pStruct, height * sizeof(png_byte*));
    for (int y = 0; y < height; y++) {
        png_byte* row =
            (png_byte*)png_malloc(pStruct, sizeof(uint8_t) * width * 4);
        ppRowData[y] = row;
        for (int x = 0; x < width; x++) {
            *row++ = 255;
            *row++ = 255;
            *row++ = 255;
            *row++ = 255;
        }
    }
}

PngImage::~PngImage()
{
    for (int y = 0; y < height; y++)
        free(ppRowData[y]);
    free(ppRowData);
}

Pixel32 PngImage::getPixelAt(int x, int y) noexcept
{
    Pixel32 pixel = {};

    const int dataSize = colorType == PNG_COLOR_TYPE_RGBA ? 4 : 3;

    png_bytep yRow = ppRowData[y];
    png_bytep point = &yRow[x * dataSize];

    pixel.red = point[0];
    pixel.green = point[1];
    pixel.blue = point[2];
    pixel.alpha = colorType == PNG_COLOR_TYPE_RGBA ? point[3] : 255;

	return pixel;
}

void PngImage::setPixelAt(int x, int y, Pixel32 pixel) noexcept
{
    const int dataSize = colorType == PNG_COLOR_TYPE_RGBA ? 4 : 3;

    png_bytep yRow = ppRowData[y];
    png_bytep point = &yRow[x * dataSize];
    
    point[0] = pixel.red;
    point[1] = pixel.green;
    point[2] = pixel.blue;
    
    if (colorType == PNG_COLOR_TYPE_RGBA)
        point[3] = pixel.alpha;
}

void PngImage::write(std::string path) throw()
{
    FILE* fp;
    png_structp pStructToWrite = NULL;
    png_infop pInfoToWrite = NULL;

    fp = fopen(path.c_str(), "wb");
    if (!fp) {
        throw PngImageException("cannot write a file from the given path");
    }

    pStructToWrite = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (pStructToWrite == NULL) {
        png_destroy_write_struct(&pStructToWrite, &pInfoToWrite);
        fclose(fp);
        throw PngImageException("cannot create struct to write");
    }

    pInfoToWrite = png_create_info_struct(pStructToWrite);
    if (pInfoToWrite == NULL) {
        png_destroy_write_struct(&pStructToWrite, &pInfoToWrite);
        fclose(fp);
        throw PngImageException("cannot create info to write");
    }

    /* Set up error handling. */

    if (setjmp(png_jmpbuf(pStructToWrite))) {
        png_destroy_write_struct(&pStructToWrite, &pInfoToWrite);
        fclose(fp);
        throw PngImageException("cannot init info");
    }

    png_init_io(pStructToWrite, fp);

    /* Set image attributes. */

    png_set_IHDR(pStructToWrite,
        pInfoToWrite,
        width,
        height,
        bitDepth,
        colorType,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT);

    png_write_info(pStructToWrite, pInfoToWrite);
    /* write bytes */
    if (setjmp(png_jmpbuf(pStructToWrite)))
        throw PngImageException("Error during writing bytes");

    png_write_image(pStructToWrite, ppRowData);


    /* end write */
    if (setjmp(png_jmpbuf(pStructToWrite)))
        throw PngImageException("Error during end of write");

    png_write_end(pStructToWrite, NULL);

    fclose(fp);
}

void PngImage::read(std::string path) throw()
{
    unsigned char header[8];    // 8 is the maximum size that can be checked

   /* open file and test for it being a png */
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp)
        throw PngImageException("File" + path + "could not be opened for reading");
    fread(header, 1, 8, fp);
    if (png_sig_cmp(header, 0, 8))
        throw PngImageException("File " + path + " is not recognized as a PNG file");

    pStruct = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!pStruct)
        throw PngImageException("png_create_read_struct failed");

    pInfo = png_create_info_struct(pStruct);

    if (!pInfo)
        throw PngImageException("png_create_info_struct failed");

    if (setjmp(png_jmpbuf(pStruct)))
        throw PngImageException("Error during init_io");

    png_init_io(pStruct, fp);
    png_set_sig_bytes(pStruct, 8);

    png_read_info(pStruct, pInfo);

    initMembers(pStruct, pInfo);

    png_read_update_info(pStruct, pInfo);
    
    /* read file */
    if (setjmp(png_jmpbuf(pStruct)))
        throw PngImageException("Error during read_image");

    ppRowData = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++)
        ppRowData[y] = (png_byte*)malloc(png_get_rowbytes(pStruct, pInfo));
     
    png_read_image(pStruct, ppRowData);

    fclose(fp);
}

void PngImage::initMembers(png_structp pStruct, png_infop pInfo)
{
    width = png_get_image_width(pStruct, pInfo);
    height = png_get_image_height(pStruct, pInfo);
    colorType = png_get_color_type(pStruct, pInfo);
    bitDepth = png_get_bit_depth(pStruct, pInfo);
    numberOfPasses = png_set_interlace_handling(pStruct);
}

