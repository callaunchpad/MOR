# MOR
Multi-Objective Robotics

<img src="http://robomo.club/wp-content/uploads/2016/08/Baxter_1_Large.jpg" alt-text="Baxter Robot" width="350" align="right">

## Description
Many real-world problems have conflicting objectives, however, it is difficult to design a single reward function that optimally combines all objectives. To address this, we will simultaneously and independently optimize all solutions on the Pareto front. This project will investigate how to implement and improve the existing MO-CMA-ES algorithm to operate a Baxter robot quickly and adaptively in production for a multi-objective problem such as collision avoidance.

## Setup Docker
1. Install [Docker](https://docs.docker.com/engine/installation/) and `docker login`
2. Source `docker.sh` file (Documentation at bottom)

## Run Default NES Algorithm
*This will run the default NES Algorithm with the parameters in Config.yaml*
1. Clone this directory
2. cd `MOR/`
3. Run `docker_build mor`
4. Run `docker_run_link_gazebo mor1 main mor`
6. Run `python session.py` to run the algorithm in the foreground (append an `&` at the end to run in the background)

## Run Gazebo
*In the docker container*
1. Run `gzserver --verbose &`
2. Copy the `IP` from the display messageL `[Msg] Publicized address: <IP>`

*In the local terminal*

3. Run `gzconnect <NAME> <IP>` where `<NAME>` is the name of your container and `<IP>` is from step 2.

## Results
- Check the `ext/` directory for your output data
  - The `.log` file contains the problem state, the reward function, and the results of each individual of each population.
  - The `.yaml` file contains the parameterws used during training
  - The `.png` files are graphs of the rewards/success per population.

### Bash command usage
  - `docker_build <TAG>`: Builds a new container with the given tag name
  - `docker_run <NAME> <TAG>`: Runs the container with the given tag and labels it with the given name
  - `docker_run_link <NAME> main <TAG>`: Same as run above, but links files between host's working directory and the working directory (`main`) in the container using a docker volume
  - `docker_run_link_gazebo <NAME> main <TAG>`: Same as above, but with Gazebo enabled between remote and local hosts.
  - `docker_inspect <NAME>`: Describes the volume link created by the above command
  - `docker_stop <NAME>`: Pauses the specified docker container
  - `docker_exec <NAME>`: SSH's into the container
  - `docker_rm <NAME>`: Removes a specified container
  - `docker_ls`: Lists all containers
  - `docker_rm_all`: Removes all containers
