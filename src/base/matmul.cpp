#include <base/matmul.h>
#include <string.h>

using namespace std;

MatMulor::MatMulor() {
}

MatMulor::~MatMulor() {
    this->destroy();
}

bool MatMulor::init(int row_A, int col_A, int col_B, rknn_core_mask core_mask, rknn_matmul_type matmul_type) {
    memset(&info, 0, sizeof(rknn_matmul_info));
    info.M = row_A;
    info.K = col_A;
    info.N = col_B;
    info.type = matmul_type;
    info.B_layout = 0;
    info.AC_layout = 0;

    memset(&io_attr, 0, sizeof(rknn_matmul_io_attr));
    if (rknn_matmul_create(&ctx, &info, &io_attr))
        return false;

    if (rknn_matmul_set_core_mask(ctx, core_mask))
        return false;

    mats.resize(3);
    mats[0] = rknn_create_mem(ctx, io_attr.A.size);
    mats[1] = rknn_create_mem(ctx, io_attr.B.size);
    mats[2] = rknn_create_mem(ctx, io_attr.C.size);

    if (rknn_matmul_set_io_mem(ctx, mats[2], &io_attr.C))
        return false;

    return true;
}

bool MatMulor::destroy() {
    for (int i = 0; i < mats.size(); i++) {
        if (rknn_destroy_mem(ctx, mats[i]))
            return false;
    }

    if (rknn_matmul_destroy(ctx))
        return false;

    return true;
}

bool MatMulor::input(void* A, void* B) {
    memcpy(mats[0]->virt_addr, A, io_attr.A.size);
    memcpy(mats[1]->virt_addr, B, io_attr.B.size);
    if (rknn_matmul_set_io_mem(ctx, mats[0], &io_attr.A))
        return false;
    if (rknn_matmul_set_io_mem(ctx, mats[1], &io_attr.B))
        return false;

    return true;
}

void* MatMulor::output() {
    return mats[2]->virt_addr;
}

bool MatMulor::run() {
    if (rknn_matmul_run(ctx))
        return false;
    return true;
}
