# fsqc Docker

Here we provide a [`Dockerfile`](Dockerfile) that can be used to create a Docker image and subsequently run the fsqc scripts within a Docker container. The main advantage of using virtualization technologies like Docker or Singularity is to create a controlled environment, with fixed versions of installable packages, which greatly helps with the stability and reproducibility of the analyses.

The docker image is based on the official Python 3.11 image (Debian-based) and contains the fsqc scripts, the lapy and brainprint libraries, and any additionally required system packages.

## Build fsqc Docker image

To build the docker image, first download the software from its GitHub repository at `https://github.com/Deep-MI/fsqc`, or clone it directly via `git clone https://github.com/Deep-MI/fsqc`.

Next, execute the following **docker** command from the project's main directory, i.e. `fsqc` (not `fsqc/fsqc`):

```bash
docker build --rm -t deepmi/fsqcdocker -f docker/Dockerfile .
```
The name of the image will be `deepmi/fsqcdocker`, and it will be built from the `Dockerfile` configuration file from the *docker* directory. You have some flexibility in choosing the name, so this is just an example. Note that the `fsqc` directory will be the *build context* for creating the docker image; if you do not want to include any particular files or directories into the image, add them to the `.dockerignore` file.

The `--rm` flag will remove intermediate containers after a successful build; `-t` specifies the name of the image, and `-f` indicates the configuration file from which to build.

Take a look at the contents of the [`Dockerfile`](Dockerfile) to see what is done during the build process: essentially, it is getting the Python3.11 image, installing additional packages, copying and installing the fsqc toolbox, and setting the necessary environment variables. Unless the `Dockerfile` changes, the build process has to be done only once.

## Download the Docker image

As an alternative to building the Docker image yourself, you can also download our pre-built images from [Dockerhub](https://hub.docker.com/r/deepmi/fsqcdocker):

```bash
docker pull deepmi/fsqcdocker
```

## Run fsqc from a Docker image

After building the fsqcdocker image, run it with the following command to see the help message of the fsqc main script:

```bash
docker run --rm --init --user XXXX:YYYY deepmi/fsqcdocker
```

* This corresponds to calling `run_fsqc` from the command line for a non-dockerized version of the program.
* The `--rm` flag takes care of removing the container once the analysis finished.
* The `--init` flag adds a helper program that allows for better process control (e.g. termation via Ctrl+C) and clean-up of finished processes. It is required from version 2.1.4 and later.
* The `--user XXXX:YYYY` part should be changed to the appropriate user id (XXXX, a number) and group id (YYYY, also a number); both can be checked with the commands `id -u` and `id -g` on linux-like systems). All generated files will then belong to the specified user and group. Using this flag is **strongly** recommended. Without the flag, the docker container will be run as root with all corresponding privileges, which is strongly discouraged.
* You can run different versions of the image using `fsqc:<tag>` instead of `fsqc` and replacing `<tag>` with any particular version identifier.

An analysis can be performed by adding several options to the above command (and after adjusting the user-specific settings and file- and pathnames):

```bash
docker run \
    --rm \
    --init \
    --user XXXX:YYYY \
    -v /path/to/my/subjects/directory:/path_to_subjects_directory_inside_docker \
    -v /path/to/my/output/directory:/path_to_output_directory_inside_docker \
    deepmi/fsqcdocker \
    --subjects_dir /path_to_subjects_directory_inside_docker \
    --output_dir /path_to_output_directory_inside_docker
```

* The `--rm` flag is used for cleaning-up after individual runs; it automatically removes the container (not the image) after it exists. It is optional, but recommended.
* The `--init` flag adds a helper program that allows for better process control (e.g. termation via Ctrl+C) and clean-up of finished processes. It is required from version 2.1.4 and later.
* The two `-v` arguments mount your data directory and output directories into the docker image. Inside the image, they are visible under the name following the colon (in this case `/data` and `/output`, but these can be different). From within the docker image / container, there will be read and write access to the directories that are mounted into the image (unless specified otherwise).
* The next part of the docker command is the name of the Docker image, which is `deepmi/fsqcdocker`.
* After that, all other flags are identical to the ones that are used for the `run_fsqc` program (which are explained on the main page and the help message of the program). In addition to the `--subjects_dir` and `--output_dir` arguments, which are mandatory, the `--subjects`, `-screenshots`, `--fornix` arguments, for example, could be specified - in the same way as for non-dockerized version of the program. Note that file- and pathnames need to correspond to the targets of the file / directory mappings within the Docker image, not to the local system.
* Also note that if you supply additional filenames to the fsqc script (using e.g. the `--subjects-file` argument), their locations must be mounted using another `-v` option (unless they are present in one of the already mounted directories), and the filenames given to the script need to refer to the mounted location inside the Docker.
* The `--shape` option is currently not supported for running in conjunction with Docker or Singularity, since it relies on the FreeSurfer software package, which is not included in our Docker images.
