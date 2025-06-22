rm -rf build
mkdir build
cd build
cmake ../ -DUSE_CUDA=ON
make
