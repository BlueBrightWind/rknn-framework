#ifndef BASE_MODEL
#define BASE_MODEL

#include <rknn_api.h>
#include <string>
#include <vector>

using namespace std;

/*
 *! @brief Encapsulation of the RKNN basic context. All models need to inherit from this basic class.
 *  @param initModelInputs() Allocate space for the data input to the model, which mainly includes type, fmt, and size. Save the pointers in input_buffers. 
 *  The default implementation is to input as uint8 and use the NHWC format.
 *  @param initModelOutputs() Allocate space for the data output of the model, which mainly includes want_float, is_prealloc, size. Save the pointers 
 *  in output_buffers. The default implementation has is_prealloc set to false and want_float set to false, meaning that the output will be generated 
 *  according to the model's output type.
 *  @param releaseModelInputs() Release the space allocated for the model input. Release the space pointed to by the pointers in input_buffers.
 *  @param releaseModelOutputs() Release the space allocated for the model output. The release method varies depending on the allocation method used. 
 *  If is_prealloc is true, you need to manually delete the space pointed to by the pointer; if it is false, you need to call rknn_outputs_release() 
 *  to release the space.
 */ 
class BaseModel {
   protected:
    char* model;
    rknn_context ctx;
    rknn_input_output_num io_num;
    vector<rknn_tensor_attr> input_attrs;
    vector<rknn_tensor_attr> output_attrs;
    vector<rknn_input> inputs;
    vector<rknn_output> outputs;

   protected:
    vector<void*> input_buffers;
    vector<void*> output_buffers;

    virtual bool initModelInputs();
    virtual bool initModelOutputs();
    virtual bool releaseModelInputs();
    virtual bool releaseModelOutputs();

   public:
    BaseModel();
    ~BaseModel();
    bool init(string path, rknn_core_mask core_mask);
    virtual bool destroy();
    bool run();
};
#endif  // BASE_MODEL