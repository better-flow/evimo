evimo
=============

This repository contains a source code for EVIMO dataset - an indoor motion segmentation dataset for event-based sensors. The dataset webpage can be found [here](https://better-flow.github.io/evimo/).

![EVIMO dataset](docs/img/dataset.png)


## QUICK START:
### Run the data generation:
```
rosrun evimo datagen_offline _folder:=EV-IMO/eval/tabletop/raw/seq_03 _with_images:=true _no_bg:=true _generate:=true _show:=-2
```
