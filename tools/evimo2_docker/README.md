# Quickstart
Refer to ../evimo2_generation/README.md to use this container for generating evimo2 data from raw recordings

# Basics
Download `evimo_data_config` and extract the raw recordings as described HERE.

Build docker container
`./docker_build.sh`

Run docker container
`./docker_run.sh /media/$(USER)/EVIMO`

Once inside compile the evimo C++ and python tools by running:
```bash
cd ~/catkin_ws; catkin_make; pip3 install -e ~/pydvs/lib; cd
source ~/catkin_ws/devel/setup.bash
```

`catkin_make` only needs to be run once because `catkin_ws` is stored on the host and mounted into the container.

# Notes
## docker_build.sh does the following things:
* Creates a `docker_home` folder on the host, it is persistent between containers
* Adds a command to source required ROS environment variables to the containers `.bashrc`
* Creates a `catkin_ws` with ROS dependencies in `docker_home`
* Clones the `pydvs` python package to `evimo2_docker`. It is not cloned to `docker_home` in order to avoid accidental deletion.

Because all source code is in the `docker_home` folder, which is mounted into the running container, code changes and ROS compilation is persistent across containers. This allows easy code editing from the host.

## docker_run.sh does the following things:
* Mounts the first argument to `/home/$USER/EVIMO`. This will likely be a 4TB or larger hard drive that will be used storing generated artifacts.
* Mounts the docker_home folder to `/home/$USER` (allows ROS to have persistence across container instances)
* Mounts the catkin workspace in that fake home directory
* Mounts the evimo ROS package into the catkin workspace
* Mounts the evimo tools into the home folder (for generating the dataset)
* Mounts pydvs into the home folder (for generating the dataset)
* Runs as the host user (so permissions and ownership are correct on host machine)
* Allows using sudo, so you can still modify the container's files
* Gives access to hosts X server (GUI apps), as a result it is insecure, only run trusted software in here

## `catkin_make`
Running `catkin_make` is required only when the C++ codes are changed. It does not need to be re-run if the container is restarted.

## `pydvs`
Pydvs is installed using the "editable" mode. This means python codes can be edited and run without running `pip3` again. However, updating pydvs's C components will require running `pip3` again.
