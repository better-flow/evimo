# EVIMO Docker Container
Refer to the [documentation](https://better-flow.github.io/evimo/docs/docker-environment.html) for detailed instructions.

# Quickstart
Download `evimo_data_config` and extract the raw recordings.

Build docker container
`./docker_build.sh`

Run docker container
`./docker_run.sh /media/$(USER)/EVIMO`

Once inside compile the evimo C++ and python tools by running:
```bash
cd ~/catkin_ws; catkin_make; pip3 install -e ~/pydvs/lib; cd
source ~/catkin_ws/devel/setup.bash
```
