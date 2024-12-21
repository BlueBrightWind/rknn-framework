#ifndef BASE_MATMULOR
#define BASE_MATMULOR

#include <rknn_matmul_api.h>
#include <vector>

using namespace std;

class MatMulor {
   private:
    rknn_matmul_ctx ctx;
    rknn_matmul_info info;
    rknn_matmul_io_attr io_attr;
    vector<rknn_tensor_mem*> mats;

   public:
    MatMulor();
    ~MatMulor();
    bool init(int row_A, int col_A, int col_B, rknn_core_mask core_mask, rknn_matmul_type matmul_type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32);
    bool destroy();
    bool input(void* A, void* B);
    void* output();
    bool run();
};

#endif