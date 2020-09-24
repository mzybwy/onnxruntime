#include <getopt.h>
#include <iostream>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <functional>
#include <vector>
#include <limits>
#include <stdexcept>
#include <libgen.h>
#include <utility>
#include <sys/time.h>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/tidl/tidl_provider_factory.h>
#include <onnxruntime/core/providers/dnnl/dnnl_provider_factory.h>

std::vector<std::string> readFileToVec(std::string fileName);

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

    const char* model_path = "../../../csharp/testdata/squeezenet.onnx";
    Ort::Session session(env, model_path, session_options);

    try
    {
        readFileToVec("poop");
    }
    catch (const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
    }

    printf("Done!\n");
    return 0;
}

std::vector<std::string> readFileToVec(std::string fname)
{
    // Open the File
    std::ifstream file(fname.c_str());
  
    // Check if object is valid
    if(!file)
    {
        throw std::runtime_error("Cannot open file: " + fname);
    }

    // Read the next line from File untill it reaches the end.
    std::string line;
    std::vector<std::string> labels;
    while (std::getline(file, line))
    {
        // Line contains string of length > 0 then save it in vector
        if(!line.empty())
        {
            labels.push_back(line);
        }
    }
  
    // Close The File
    file.close();
    return labels;
}

