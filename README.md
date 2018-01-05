# MOR
Multi-Objective Robotics

## Setup Docker
1. Install [Docker](https://docs.docker.com/engine/installation/) and `docker login`
2. Add the following to your `~/.zshrc` or `~/.bashrc` file (Documentation at bottom)
```
docker_build() { docker build -t $1 .; }
docker_run() { docker run -it --name $1 -d -p 4000 $2 /bin/bash; }
docker_run_link() { docker run -it --name $1 -v $(pwd):/$2 -d -p 4000 $3 /bin/bash; }
docker_inspect() { docker inspect -f "{{json .Mounts}}" $1; }
docker_stop() { docker container stop $1; }
docker_exec() { docker exec -t -i $1 /bin/bash; }
docker_rm() { docker rm $1; }
docker_ls() { docker container ls -a; }
docker_rm_all() { docker rm $(docker ps -a -q); }
```

## Run Default NES Algorithm
*This will run the default NEW Algorithm with the parameters in Config.yaml*
1. Clone this directory
2. cd `MOR/`
3. Run `docker_build mor`
4. Run `docker_run_link mor1 main mor`
5. Run `docker_exec mor1`
  - Should now be in environment `root@<CONTAINER_ID>:/main#`
6. Run `python session.py` to run the algorithm in the foreground (append an `&` at the end to run in the background)

## Results
- Check the `ext/` directory for your output data
- The `.log` file contains the problem state, the reward function, and the results of each individual of each population.
- The `.yaml` file contains the parameterws used during training
- The `.png` files are graphs of the rewards/success per population.

### Bash command usage
  - `docker_build <TAG>`: Builds a new container with the given tag name
  - `docker_run <NAME> <TAG>`: Runs the container with the given tag and labels it with the given name
  - `docker_run_link <NAME> main <TAG>`: Same as run above, but links files between host's working directory and the working directory (`main`) in the container using a docker volume
  - `docker_inspect <NAME>`: Describes the volume link created by the above command
  - `docker_stop <NAME>`: Pauses the specified docker container
  - `docker_exec <NAME>`: SSH's into the container
  - `docker_rm <NAME>`: Removes a specified container
  - `docker_ls`: Lists all containers
  - `docker_rm_all`: Removes all containers
