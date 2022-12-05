#pragma once

#include "pch.h"
#include "CppUnitTest.h"

#include "../CudaProjectImageProcessing/Image.cpp"
#include <string>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace TestProject
{
	TEST_CLASS(TestImage)
	{
	public:
		
		TEST_METHOD(TestImageGetPixel)
		{
            Image image("test/tiny_color.png");
            Pixel pixel = image.get_pixel_at(0, 0);
            Assert::AreEqual(static_cast<int>(pixel.red), 255);
            Assert::AreEqual(static_cast<int>(pixel.green), 0);
            Assert::AreEqual(static_cast<int>(pixel.blue), 8);
            Assert::AreEqual(static_cast<int>(pixel.intensity), 54);
		}

        TEST_METHOD(TestImageSetPixel)
        {
            Image image("test/tiny_color.png");
            Pixel pixel = image.get_pixel_at(0, 0);

            pixel.red = 0;
            pixel.green = 0;
            pixel.blue = 0;
            pixel.intensity = 0;

            Pixel pixel_must_not_changed = image.get_pixel_at(0, 0);
            Assert::AreEqual(static_cast<int>(pixel_must_not_changed.red), 255);
            Assert::AreEqual(static_cast<int>(pixel_must_not_changed.green), 0);
            Assert::AreEqual(static_cast<int>(pixel_must_not_changed.blue), 8);
            Assert::AreEqual(static_cast<int>(pixel_must_not_changed.intensity), 54);

            image.set_pixel_at(0, 0, pixel);
            Pixel pixel_changed = image.get_pixel_at(0, 0);
            Assert::AreEqual(static_cast<int>(pixel_changed.red), 0);
            Assert::AreEqual(static_cast<int>(pixel_changed.green), 0);
            Assert::AreEqual(static_cast<int>(pixel_changed.blue), 0);
            Assert::AreEqual(static_cast<int>(pixel_changed.intensity), 0);
        }

        TEST_METHOD(TestCreateImage)
        {
            constexpr int WIDTH = 32;
            constexpr int HEIGHT = 32;

            Image image(WIDTH, HEIGHT);

            for (int i = 0; i < WIDTH; i++)
            {
                for (int j = 0; j < HEIGHT; j++)
                {
                    Pixel pixel = image.get_pixel_at(i, j);
                    Assert::AreEqual(static_cast<int>(pixel.red), Image::DEFAULT_INTENSITY);
                    Assert::AreEqual(static_cast<int>(pixel.green), Image::DEFAULT_INTENSITY);
                    Assert::AreEqual(static_cast<int>(pixel.blue), Image::DEFAULT_INTENSITY);
                    Assert::AreEqual(static_cast<int>(pixel.intensity), Image::DEFAULT_INTENSITY);
                }
            }
        }
	};

    TEST_CLASS(TestLearning)
    {
    public:

        //TEST_METHOD(TestImageGetPixel)
        //{
        //    Image image("test/tiny_color.png");
        //    Pixel pixel = image.get_pixel_at(0, 0);
        //    Assert::AreEqual(static_cast<int>(pixel.red), 255);
        //    Assert::AreEqual(static_cast<int>(pixel.green), 0);
        //    Assert::AreEqual(static_cast<int>(pixel.blue), 8);
        //    Assert::AreEqual(static_cast<int>(pixel.intensity), 54);
        //}
    };
}
