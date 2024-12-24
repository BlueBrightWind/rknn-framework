#include <arm_neon.h>
#include <base/matmul.h>
#include <string.h>
#include <chrono>
#include <iostream>

#define ROW_A 13
#define COL_A 32
#define COL_B (160 * 160)

using namespace std;
using namespace chrono;

int main() {
    float16_t A[ROW_A * COL_A];
    float16_t B[COL_A * COL_B];
    float C[ROW_A * COL_B];

    for (int i = 0; i < ROW_A; i++) {
        for (int j = 0; j < COL_A; j++) {
            A[i * COL_A + j] = 1.0;
        }
    }

    for (int i = 0; i < COL_A; i++) {
        for (int j = 0; j < COL_B; j++) {
            B[i * COL_B + j] = 1.0;
        }
    }

    MatMulor matmulor;
    matmulor.init(ROW_A, COL_A, COL_B, RKNN_NPU_CORE_0);

    auto t1 = system_clock::now();
    for (int i = 0; i < 100; i++) {
        matmulor.input(A, B);
        matmulor.run();
        memcpy(C, matmulor.output(), ROW_A * COL_B * sizeof(float));
    }
    auto t2 = system_clock::now();

    printf("time cost: %.2f ms\n", duration_cast<microseconds>(t2 - t1).count() / 1000.0 / 100);

    return 0;
}