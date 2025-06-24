rm -rf build
mkdir build
cd build
cmake ../ -DUSE_CUDA=ON -DCMAKE_VERBOSE_MAKEFILE=ON
make
