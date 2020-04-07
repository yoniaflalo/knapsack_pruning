import glob
import logging
import os
import time

logger = logging.getLogger(__name__)


def untar_dataset_files(copy_folder, dataset_name, tarfiles_path, download=True):
    """
    Convention: tar files to untar (and only those tars) should be in args.data_path folder
    Make sure there are no extra tar files there.
    :param experiment_name:
    :param dataset_name:
    :param tarfiles_path:
    :param args:
    :return:
    """
    if not os.path.isdir(tarfiles_path):
        tarfiles_path = os.path.dirname(tarfiles_path)

    start_time = time.time()
    destination_dataset_folder = "/{}/{}/".format(copy_folder, dataset_name)
    if not download:
        return destination_dataset_folder

    logger.info("Found .tar.gz, untarring to local drive...")
    create_destination_folder(destination_dataset_folder)
    extract_tar_files_in_folder(tarfiles_path, destination_dataset_folder)
    logger.info('Data copy time (min)=' + str(time.time() - start_time))
    return destination_dataset_folder


def extract_tar_files_in_folder(tarfiles_path, destination_dataset_folder):
    # *.tar
    if len(glob.glob(tarfiles_path + '/*.tar')) > 0:
        tar_files = glob.glob(tarfiles_path + "/*.tar")
        logger.info(f"Extracting files {tar_files}")
        cmd = 'for file in {tarfiles_path}/*.tar; do ' \
              'tar -C {destination_dataset_folder} -xf "$file"; done '.format(
            destination_dataset_folder=destination_dataset_folder, tarfiles_path=tarfiles_path)
        run_cmd(cmd)

    # *.tar.gz
    if len(glob.glob(tarfiles_path + '/*.tar.gz')) > 0:
        targz_files = glob.glob(tarfiles_path + "/*.tar.gz")
        logger.info(f"Extracting files {targz_files} into {destination_dataset_folder}")
        cmd = 'for file in {tarfiles_path}/*.tar.gz; do ' \
              'tar -C {destination_dataset_folder} -xzf "$file"; done '.format(
            destination_dataset_folder=destination_dataset_folder, tarfiles_path=tarfiles_path)
        run_cmd(cmd)


def folder_contains_tar(folder):
    if not os.path.isdir(folder):
        folder = os.path.dirname(folder)
    return len(glob.glob(folder + '/*.tar')) > 0 or len(glob.glob(folder + '/*.tar.gz')) > 0


def has_tar_extension(oss_asset_key):
    return oss_asset_key.lower().endswith(('.tar', '.tar.gz'))


def create_destination_folder(destination_dataset_folder):
    cmd = "mkdir -p {destination_dataset_folder}".format(destination_dataset_folder=destination_dataset_folder)
    run_cmd(cmd)


def run_cmd(cmd):
    import subprocess
    subprocess.check_output([cmd], shell=True)


def endpoint_optimiser(endpoint):
    if "HYPERML_ENV_REGION" not in os.environ:
        return endpoint

    if "internal" not in endpoint \
        and "aliyun-inc" not in endpoint \
        and os.environ["HYPERML_ENV_REGION"] in endpoint:
        new_endpoint = endpoint.replace(os.environ["HYPERML_ENV_REGION"],
                                        os.environ["HYPERML_ENV_REGION"] + "-internal")
        logger.info("Changed endpoint from {} to {}".format(endpoint, new_endpoint))
        return new_endpoint

    if "internal" in endpoint \
        and os.environ["HYPERML_ENV_REGION"] not in endpoint:
        new_endpoint = endpoint.replace("-internal", "")
        logger.info("Changed endpoint from {} to {}".format(endpoint, new_endpoint))
        return new_endpoint

    return endpoint
