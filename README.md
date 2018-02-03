# MOR
Multi-Objective Robotics

<img src="http://robomo.club/wp-content/uploads/2016/08/Baxter_1_Large.jpg" alt-text="Baxter Robot" width="350" align="right">

## Description
Many real-world problems have conflicting objectives, however, it is difficult to design a single reward function that optimally combines all objectives. To address this, we will simultaneously and independently optimize all solutions on the Pareto front. This project will investigate how to implement and improve the existing MO-CMA-ES algorithm to operate a Baxter robot quickly and adaptively in production for a multi-objective problem such as collision avoidance.

## Setup Docker
1. Install [Docker](https://docs.docker.com/engine/installation/) and `docker login`
2. Source `docker.sh` file (Documentation at bottom)

## Run NES Algorithm
*This will run the default NES Algorithm with the parameters in Config.yaml*

#### Local Machine
1. Clone this directory
2. cd `MOR/`
3. Run `./train.py <CONFIG_FILENAME>.yaml` to run the algorithm in the foreground (append an `&` at the end to run in the background)
    - Use `<CONFIG_FILENAME> = "Config"` to run the default Maze example
    - Check other config files in `cfg/` for other options, or write your own `.yaml` config file and add it to `cfg/`.
    - Resolve any dependecy issues that may arise
      - Linux/Mac OSX: `sudo -H pip install numpy tensorflow matplotlib pyyaml`

#### Docker Container (Recommended)
1. Clone this directory
2. cd `MOR/`
3. Run `docker_build mor`
4. Run `docker_run_link_gazebo mor1 main mor`
5. Run `python train <CONFIG_FILENAME>.yaml` to run the algorithm in the foreground (append an `&` at the end to run in the background)
    - Use `<CONFIG_FILENAME> = "Config"` to run the default Maze example
    - Check other config files in `cfg/` for other options, or write your own `.yaml` config file and add it to `cfg/`.

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
