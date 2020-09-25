#include <getopt.h>
#include <iostream>
#include <cstdarg>
#include <cstdio>
#include <fstream>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/tidl/tidl_provider_factory.h>
#include <onnxruntime/core/providers/dnnl/dnnl_provider_factory.h>

#include "validator.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

bool SetupInput(std::string input_path, cv::Mat& input_image)
{
    // Read image input
    input_image = cv::imread(input_path, CV_LOAD_IMAGE_COLOR);
    if(! input_image.data )                              // Check for invalid input
    {
        std::cout <<  "Could not open or find the image" << std::endl;
        return false;
    }
    std::cout << "Image input: " << input_path.c_str() << std::endl;
    return true;
}

/*
 * Retrieve frame, resize, and record in NCHW format
 */
void CollectFrames(std::vector<uint8_t> &output,
                   cv::Mat &in_image,
                   int width, int height, int channels)
{
    cv::Mat image;
    cv::resize(in_image, image, cv::Size(width, height));
    cv::Mat *spl = new cv::Mat[channels];
    split(image,spl);
    
    // Read the frame in NCHW format
    output.resize(height * width * channels);
    int idx = 0;
    for(int c = 0; c < channels; c++)
    {
        const unsigned char* data = image.ptr();
        for(int x = 0; x < width; x++)
        {
            for(int y = 0; y < height; y++)
            {
                output[idx++] =
                    (uint8_t)data[(channels) * (y + x*width) + (channels - 1) - c];
            }
        }
    }
}

int main(int argc, char* argv[])
{
    OrtStatus *status;
    
    // Initialize  enviroment, maintains thread pools and state info
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    
    // Initialize session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    status = OrtSessionOptionsAppendExecutionProvider_Dnnl(session_options, 1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Model Path
    const char* model_path = "../../../csharp/testdata/squeezenet.onnx";

    // Create Validator
    std::string input_path = "mushroom.png";
    cv::Mat input_image;
    std::vector<uint8_t> image_data;
    
    SetupInput(input_path, input_image);
    CollectFrames(image_data, input_image, 224, 224, 3);
    
    Validator validator(env, model_path, session_options, image_data);

    printf("Done!\n");
    return 0;
}

