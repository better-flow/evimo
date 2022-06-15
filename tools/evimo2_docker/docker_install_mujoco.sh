python3 -m venv mujoco_venv
source mujoco_venv/bin/activate
pip install wheel glfw opencv-python
git clone https://github.com/aftersomemath/mujoco.git
cd mujoco
git checkout simulate-python4
mkdir build; cd build
cmake ../. -DCMAKE_INSTALL_PREFIX:STRING=/tmp/mujoco_install
cmake --build . -j8
cmake --install .
cd ../python
export MUJOCO_PATH=/tmp/mujoco_install
./make_sdist.sh
pip wheel dist/mujoco-2.2.0.tar.gz
pip install mujoco-2.2.0-cp38-cp38-linux_x86_64.whl
