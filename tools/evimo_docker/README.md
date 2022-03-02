# Setup
Download `evimo_data_config` and extract the raw recordings. (TODO autodownloader)
The drive containing `evimo_data_config` should have XX (4?)TB of free space.

Build docker container
`./docker_build.sh`

Run docker container
`./docker_run.sh /path/to/evimo_data_config`

Once inside compile the evimo C++ tools by running:
```bash
cd catkin_ws
catkin_make
```

`catkin_make` only needs to be run once because `catkin_ws` is stored on the host and mounted into the container.

# Notes
docker_run.sh does the following things:
* Uses fake home directory with same name as host user for roscore to store things
* Mounts the catkin workspace in that fake home directory
* Runs as the host user (so permissions and ownership is correct on host machine file system)
* Allows using sudo, so you can still modify the container's files
* Gives access to hosts X server (GUI apps), as a result it is insecure, only run trusted software in here

