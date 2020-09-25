#ifndef _VALIDATOR_H
#define _VALIDATOR_H

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

class Validator {
private:
    // ORT Session
    Ort::Session _session;

    // Input information
    size_t _num_input_nodes;
    std::vector<const char*> _input_node_names;
    std::vector<int64_t> _input_node_dims;
    std::vector<uint8_t> _image_data;

    int _image_size;

    void PrepareInputs();
    void ScoreModel();
    void Validate();
    
    std::vector<std::string> ReadFileToVec(std::string fname);

public:
    int GetImageSize() const;
    Validator(Ort::Env& env, const char* model_path, Ort::SessionOptions& session_options,
              std::vector<uint8_t>& image_data);
};

#endif // _VALIDATOR_H
