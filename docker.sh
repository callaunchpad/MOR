docker_build() { docker build -t $1 .; }
docker_rm_all() { docker rm $(docker ps -a -q); }
docker_rm() { docker rm $1; }
docker_run() { docker run --name $1 -d -p 4000 $2; }
docker_run_link() { docker run -it --name $1 -v $(pwd):/$2 -d -p 4000 $3 /bin/bash; }
docker_run_link_gazebo() {
    mkdir -p $(pwd)/ext/.gazebo/;
    docker run -it \
    --name=$1 \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1"  \
    --volume=$(pwd):/$2 \
    --volume=$(pwd)/ext/.gazebo/:/root/.gazebo/ \
    --volume="/etc/group:/etc/group:ro" \
    --volume="/etc/passwd:/etc/passwd:ro" \
    --volume="/etc/shadow:/etc/shadow:ro" \
    --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    $3 \
    /bin/bash;
    # export containerId=$(docker ps -l -q);
}
docker_stop() { docker container stop $1; }
docker_exec() { docker exec -t -i $1 /bin/bash; }
docker_ls() { docker container ls -a; }
docker_inspect() { docker inspect -f "{{json .Mounts}}" $1; }
