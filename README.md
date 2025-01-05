# RKNN Framework for rk3588/rk3588s
## Description
The project is a framework which is able to quickly implement rknn AI inference. It is based on the [`rknpu2`](https://github.com/airockchip/rknn-toolkit2) and provides a simple and easy-to-use interface. 

## Highlights
 - `Inference`, `Matrix mul`, and `Asynchronous` framework are supported.
 - Packaged commonly used `object detection`, `image segmentation`, and `pose recognition` models for out-of-the-box use.
 - All the packaging is compatible with both `quantized` and `non-quantized` versions of the models.

## How to use
You can refer to [`this article`](https://bluebrightwind.github.io/2024/12/30/Orangepi-Dev-Env) to build your development environment

```bash
git clone https://github.com/BlueBrightWind/rknn-framework.git --depth=1 rknn-framework
cd rknn-framework
mkdir build
cd build
cmake ..
make
```

Then you can run the `xxxx_demo` in the build dir to test the speed and the result.

## Train your own model
The framework supports the modified model provided by Rockchip. You can refer to [`this repository`](https://github.com/airockchip/rknn_model_zoo) to view the code related to model training

## TODO
 - Adapt and package more models.
