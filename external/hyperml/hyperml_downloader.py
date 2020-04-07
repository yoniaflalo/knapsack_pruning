import hyperml
import os


def _has_tar_pth_extension(oss_asset_key):
    return oss_asset_key.lower().endswith(('.tar', '.tar.gz', '.pth'))


def _ensure_folder_ends_with_backslash(oss_dataset_path):
    if not _has_tar_pth_extension(oss_dataset_path) and oss_dataset_path[-1] != "/":
        oss_dataset_path += "/"
    return oss_dataset_path


class HypermlDownloader:

    def __init__(self, oss_path, endpoint, oss_access_key_id, oss_access_key_secret, oss_cache):
        self.oss_path = oss_path
        self.oss_endpoint = endpoint
        self.oss_bucket, self.oss_dataset_path = oss_path.split("oss://")[1].split("/", maxsplit=1)
        self.oss_dataset_path = _ensure_folder_ends_with_backslash(self.oss_dataset_path)
        self.oss_access_key_id = oss_access_key_id
        self.oss_access_key_secret = oss_access_key_secret
        self.cache = oss_cache

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.hyperml_yaml_path = current_dir + "/hyperml_downloader.yaml"

    def download(self, output_path):
        temp_dataset_name = "temp_dataset_name"
        dataset_asset = {
            'local_dir': output_path,
            'endpoint': self.oss_endpoint,
            'bucket': self.oss_bucket,
            'cache': self.cache,
            'keys': [{'name': temp_dataset_name,
                      'key': f"/{self.oss_dataset_path}"}],
            "credentials": {'access_key_id': self.oss_access_key_id,
                            'access_key_secret': self.oss_access_key_secret}
        }

        hyperml.init(self.hyperml_yaml_path, datasets=dataset_asset)

        # Return the absolute destination path of the downloaded data
        return output_path + "/" + self.oss_dataset_path

    def get_output_path(self, output_path):
        return output_path + "/" + self.oss_dataset_path
