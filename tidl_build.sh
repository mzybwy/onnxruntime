pushd ~/work/ecplr/onnxruntime
./build.sh --config RelWithDebInfo --build_shared_lib --parallel --skip_tests --use_tidl --use_dnnl
popd
