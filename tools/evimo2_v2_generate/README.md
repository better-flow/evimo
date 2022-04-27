# Generate the EVIMO2v2 dataset
Refer to the [documentation](https://better-flow.github.io/evimo/docs/evimo2v2-generation.html) for detailed instructions.

## Quick Reference

### Generate a particular sequence
Generating a single sequence can take a few minutes to an hour depending on the seqeuence. More cores will make it faster.

```
cd evimo/tools/evimo2_docker
./docker_build.sh
./docker_run.sh /media/$USER/EVIMO
cd ~/catkin_ws; catkin_make; pip3 install -e ~/pydvs/lib; cd
source ~/catkin_ws/devel/setup.bash
cd ~/tools/evimo2_v2_generate; ./generate.sh ~/EVIMO/raw/imo/eval/scene13_dyn_test_00
```

### Generate everything
Generating the entire dataset can take days.

```
cd evimo/tools/evimo2_docker
./docker_build.sh
./docker_run.sh /media/$USER/EVIMO
cd ~/catkin_ws; catkin_make; pip3 install -e ~/pydvs/lib; cd
source ~/catkin_ws/devel/setup.bash
cd ~/tools/evimo2_generate; ./generate_all.sh ~/EVIMO/raw
```
