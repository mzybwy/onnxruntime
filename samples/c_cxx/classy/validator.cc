#include <assert.h>
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

#include "validator.h"

Validator::Validator(Ort::Env& env,
                     const char* model_path,
                     Ort::SessionOptions& session_options,
                     std::vector<uint8_t>& image_data)
    : _session(env, model_path, session_options),
      _num_input_nodes{_session.GetInputCount()},
      _input_node_names(_num_input_nodes),
      _image_data(image_data)
{
    Validate();
}

int Validator::GetImageSize() const
{
    return _image_size;
}

void Validator::PrepareInputs()
{
    Ort::AllocatorWithDefaultOptions allocator;
    
    printf("Number of inputs = %zu\n", _num_input_nodes);
    
    // iterate over all input nodes
    for (int i = 0; i < _num_input_nodes; i++) {
        // print input node names
        char* input_name = _session.GetInputName(i, allocator);
        printf("Input %d : name=%s\n", i, input_name);
        _input_node_names[i] = input_name;
        
        // print input node types
        Ort::TypeInfo type_info = _session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Input %d : type=%d\n", i, type);
        
        // print input shapes/dims
        _input_node_dims = tensor_info.GetShape();
        printf("Input %d : num_dims=%zu\n", i, _input_node_dims.size());
        for (int j = 0; j < _input_node_dims.size(); j++)
        {
            printf("Input %d : dim %d=%jd\n", i, j, _input_node_dims[j]);
        }
    }
}

void Validator::ScoreModel()
{
    //*************************************************************************
    // Score the model using sample data, and inspect values

    size_t input_tensor_size = 224 * 224 * 3;  // simplify ... using known dim values to calculate size
    // use OrtGetTensorShapeElementCount() to get official size!

    // std::vector<float> input_tensor_values(input_tensor_size);
    std::vector<const char*> output_node_names = {"softmaxout_1"};

    // create input tensor object from data values
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<float> floatvec(_image_data.begin(), _image_data.end());
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, floatvec.data(), input_tensor_size, _input_node_dims.data(), 4);
    assert(input_tensor.IsTensor());

    // score model & input tensor, get back output tensor
    auto run_options = Ort::RunOptions();
    run_options.SetRunLogVerbosityLevel(2);
    
    auto output_tensors = _session.Run(run_options, _input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
    assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

    // Get pointer to output tensor float values
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();

    // Determine most common index
    float max_val = 0.0;
    int max_index = 0;
    for (int i = 0; i < 1000; i++)
    {
        if (floatarr[i] > max_val)
        {
            max_val = floatarr[i];
            max_index = i;
        }
    }
    std::cout << "MAX: class [" << max_index << "] = " << max_val << std::endl;

    std::vector<std::string> labels = ReadFileToVec("labels.txt");
    std::cout << labels[max_index] << std::endl;
}

void Validator::Validate()
{
    PrepareInputs();
    ScoreModel();
}

std::vector<std::string> Validator::ReadFileToVec(std::string fname)
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
