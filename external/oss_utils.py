import asyncio
import io
import logging
import os
import subprocess
import pickle
import random
import sys
import tempfile
import zlib
from urllib.parse import urlparse
from pathlib import Path
from typing import Collection, Optional, Union
import aiohttp
import oss2
import torch
import torchvision.datasets as datasets
from PIL import Image
from torch.utils import data as torch_data

before_refactor = hasattr(torch_data.dataloader, '_worker_loop')

if before_refactor:
    from torch.utils.data.dataloader import MP_STATUS_CHECK_INTERVAL, ManagerWatchdog, ExceptionWrapper
else:
    from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL, ExceptionWrapper
    from torch.utils.data._utils.worker import ManagerWatchdog

if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue

from torch._C import _set_worker_signal_handlers

logger = logging.getLogger(__name__)

PathOrStr = Union[Path, str]
FilePathList = Collection[Path]

logger = logging.getLogger(__name__)

from external.distributed_manager import DistributedManager as dist


def _worker_loop_async(dataset, index_queue, data_queue, done_event, collate_fn, seed, init_fn, worker_id):
    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.

    try:
        global _use_shared_memory
        _use_shared_memory = True

        # Intialize C side signal handlers for SIGBUS and SIGSEGV. Python signal
        # module's handlers are executed after Python returns from C low-level
        # handlers, likely when the same fatal signal happened again already.
        # https://docs.python.org/3/library/signal.html Sec. 18.8.1.1
        _set_worker_signal_handlers()

        torch.set_num_threads(1)
        random.seed(seed)
        torch.manual_seed(seed)

        data_queue.cancel_join_thread()

        if init_fn is not None:
            init_fn(worker_id)

        watchdog = ManagerWatchdog()

        while watchdog.is_alive():
            try:
                r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue
            if r is None:
                # Received the final signal
                assert done_event.is_set()
                return
            elif done_event.is_set():
                # Done event is set. But I haven't received the final signal
                # (None) yet. I will keep continuing until get it, and skip the
                # processing steps.
                continue
            idx, batch_indices = r
            try:
                if hasattr(dataset, 'get_items'):
                    samples = collate_fn(dataset.get_items(list(batch_indices)))
                else:
                    samples = collate_fn([dataset[i] for i in batch_indices])
            except Exception:
                # It is important that we don't store exc_info in a variable,
                # see NOTE [ Python Traceback Reference Cycle Problem ]
                data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
            else:
                data_queue.put((idx, samples))
                del samples
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass


def set_dataloader_batch_fetch(enable=True):
    if before_refactor:
        import torch.utils.data.dataloader as module_to_patch
    else:
        import torch.utils.data._utils.worker as module_to_patch

    set_dataloader_batch_fetch._worker_loop_prev = \
        getattr(set_dataloader_batch_fetch, '_worker_loop_prev',
                module_to_patch._worker_loop)

    if enable:
        module_to_patch._worker_loop = _worker_loop_async
    else:
        module_to_patch._worker_loop = set_dataloader_batch_fetch._worker_loop_prev


class ImageFolderOss(object):
    __version = 'v0'

    def __init__(self, url_oss, endpoint, access_key, secret, from_index=True, make_index=False, transform=None,
                 target_transform=None, extensions=None, check_corrupted_files=False):
        """
        :param url_oss: oss://bucket-name/path1/path2/
        :param endpoint: e.g oss-cn-hangzhou-internal.aliyuncs.com
        :param access_key: You OSS access key
        :param secret: Your OSS access key secret
        :param from_index: Disable loading samples list from index
        :param make_index: Enable creating index file if not used
        :param transform: Sample transform
        :param target_transform: Target transform
        :param extensions: Supported sample file extensions
        """
        self.url = url_oss
        self.extensions = extensions or datasets.folder.IMG_EXTENSIONS
        self.transform = transform
        self.target_transform = target_transform
        self.endpoint = endpoint
        self.check_corrupted_files = check_corrupted_files

        url_parsed = urlparse(self.url)
        self.prefix = url_parsed.path.lstrip('/')
        if not self.prefix[-1] == '/':
            self.prefix = self.prefix + '/'
        self.name_bucket = url_parsed.netloc
        self.auth = oss2.Auth(access_key, secret)

        # Used for multi process dataset access
        self._worker_bucket = None
        self._aio_session = None

        self._get_samples(from_index, make_index)
        logger.info(f'\n'
                    f'Number of images: {len(self.keys_relative)}\n'
                    f'Prefix: {self.prefix}\n'
                    f'Bucket: {url_parsed.netloc}')

        if len(self.keys_relative) == 0:
            raise (RuntimeError("Found 0 files in sub folders of: " + self.url))

    @staticmethod
    def loader(data):
        return Image.open(data).convert('RGB')

    def _path_index(self):
        return os.path.join(self.prefix, f'index_{self.__version}.pkl')

    def make_index(self):
        index_dump = zlib.compress(pickle.dumps((self.keys_relative, self.classes, self.class_to_idx)))
        bucket = oss2.Bucket(self.auth, self.endpoint, self.name_bucket)
        upload_path = self._path_index()
        logger.info(f"Uploading index to {upload_path}")
        bucket.put_object(upload_path, index_dump)

    def has_valid_extension(self, key):
        _, ext = os.path.splitext(key)
        return ext.lower() in self.extensions

    def _filter_samples_by_extension(self):
        self.keys_relative = list(filter(self.has_valid_extension, self.keys_relative))

    def _get_samples_from_index(self, bucket):
        logger.info(f"Index path {self._path_index()}")

        with tempfile.NamedTemporaryFile() as file_tmp:
            # Used intrad of put_object to avoid timeouts
            oss2.resumable_download(bucket, self._path_index(), file_tmp.name, num_threads=5)
            with open(file_tmp.name, 'rb') as fp:
                data = fp.read()
                logger.info(f"Index size: {len(data)}")

            self.keys_relative, self.classes, self.class_to_idx = pickle.loads(
                zlib.decompress(data))

            # Patch for indices including ''
            if '' in self.classes:
                self.classes, self.class_to_idx = self._find_classes()

    def _get_samples_from_tree(self, bucket):
        self.keys_relative = []

        # max_keys=1000 is the current max value OSS supports
        iter_prefix = oss2.ObjectIterator(bucket, prefix=self.prefix, max_keys=1000)
        for idx, object_info in enumerate(iter_prefix):
            if idx % 1000 == 0:
                logger.info(f'Discovered samples: {idx}')

            key = object_info.key[len(self.prefix):]
            self.keys_relative.append(key)

        self.classes, self.class_to_idx = self._find_classes()

    def _index_exists(self, bucket):
        try:
            bucket.get_object_meta(self._path_index())
        except oss2.exceptions.NoSuchKey:
            return False

        return True

    def _get_samples(self, from_index, make_index):
        bucket = oss2.Bucket(self.auth, self.endpoint, self.name_bucket)

        if from_index and self._index_exists(bucket):
            self._get_samples_from_index(bucket)
        else:
            self._get_samples_from_tree(bucket)
            if make_index and len(self.keys_relative) > 0:
                self.make_index()

        self._filter_samples_by_extension()

    @staticmethod
    def _key_to_class(key):
        return os.path.basename(os.path.dirname(key))

    def _find_classes(self):
        """
        Finds the class folders in a dataset.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        """

        classes = {self._key_to_class(key) for key in self.keys_relative}
        classes.discard('')
        classes = sorted(classes)
        class_to_idx = {class_: target for target, class_ in enumerate(classes)}
        return classes, class_to_idx

    @staticmethod
    async def fetch(url, session, n_retry=10):
        for i_retry in range(n_retry):
            try:
                async with session.get(url) as response:
                    return await response.read()
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if i_retry + 1 == n_retry:
                    raise

    async def _get_async(self, session, index):
        key = os.path.join(self.prefix, self.keys_relative[index])
        url = self._worker_bucket.sign_url('GET', key, 5 * 60)
        target = self.class_to_idx[self._key_to_class(key)]
        try:
            data = await self.fetch(url, session)
            sample = self.loader(io.BytesIO(data))
        except:
            sample = Image.new('RGB', (128, 128))
            if self.check_corrupted_files:
                target = len(self.classes)
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)

        return sample, target, url

    def get_items(self, indices):
        # Required since OSS SDK doesn't support multiprocessing gracefully
        self._worker_bucket = self._worker_bucket if self._worker_bucket is not None else \
            oss2.Bucket(self.auth, self.endpoint, self.name_bucket)
        self._aio_session = self._aio_session if self._aio_session is not None \
            else aiohttp.ClientSession(raise_for_status=True, headers={"Connection": "close"})

        loop = asyncio.get_event_loop()
        tasks = [self._get_async(self._aio_session, index) for index in indices]
        res = loop.run_until_complete(asyncio.gather(*tasks))

        return res

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        # Required since OSS SDK doesn't support multiprocessing gracefully
        if self._worker_bucket is None:
            self._worker_bucket = oss2.Bucket(self.auth, self.endpoint, self.name_bucket)

        key = os.path.join(self.prefix, self.keys_relative[index])
        data = self._worker_bucket.get_object(key)
        try:
            sample = self.loader(data)
        except:
            return sample, len(self.classes), key

        if self.transform:
            sample = self.transform(sample)

        target = self.class_to_idx[self._key_to_class(key)]
        if self.target_transform:
            target = self.target_transform(target)

        return sample, target, key

    def __len__(self):
        return len(self.keys_relative)


def get_files(path: PathOrStr, extensions: Collection[str] = None, recurse: bool = False,
              include: Optional[Collection[str]] = None, presort: bool = False,
              followlinks: bool = False) -> FilePathList:
    "Return list of files in `path` that have a suffix in `extensions`; optionally `recurse`."
    if recurse:
        res = []
        for i, (p, d, f) in enumerate(os.walk(path, followlinks=followlinks)):
            # skip hidden dirs
            if include is not None and i == 0:
                d[:] = [o for o in d if o in include]
            else:
                d[:] = [o for o in d if not o.startswith('.')]
            res += _get_files(path, p, f, extensions)
        if presort: res = sorted(res, key=lambda p: _path_to_same_str(p), reverse=False)
        return res
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, path, f, extensions)
        if presort: res = sorted(res, key=lambda p: _path_to_same_str(p), reverse=False)
        return res


def _path_to_same_str(p_fn):
    "path -> str, but same on nt+posix, for alpha-sort only"
    s_fn = str(p_fn)
    s_fn = s_fn.replace('\\', '.')
    s_fn = s_fn.replace('/', '.')
    return s_fn


def _get_files(parent, p, f, extensions):
    p = Path(p).relative_to(parent)
    if isinstance(extensions, str): extensions = [extensions]
    low_extensions = [e.lower() for e in extensions] if extensions is not None else None
    res = [p / o for o in f if not o.startswith('.')
           and (extensions is None or f'.{o.split(".")[-1].lower()}' in low_extensions)]
    return res


def build_oss_asset_info(args):
    oss_asset_info = {
        'oss_access_key_id': args.oss_access_key_id,
        'oss_access_key_secret': args.oss_access_key_secret,
        'oss_dataset_path': args.oss_dataset_path,
        'oss_endpoint': args.oss_dataset_endpoint
    }
    return oss_asset_info


def upload_dir_to_oss(oss_asset_info, dir_in: str):
    path = Path(dir_in)
    files = get_files(path, recurse=True)
    dir_out = path.parts[-1]

    for file in files:
        fname_in = str(path.joinpath(file))
        fname_out = str(Path(dir_out).joinpath(file))
        upload_file_to_oss(oss_asset_info=oss_asset_info, fname_in=fname_in, fname_out=fname_out)

    return


def upload_file_to_oss(oss_asset_info, fname_in: str, fname_out: str):
    url_parsed = urlparse(oss_asset_info.get('oss_dataset_path'))
    name_bucket = url_parsed.netloc
    url_dataset_path = url_parsed.path[1:]

    auth = oss2.Auth(oss_asset_info.get('oss_access_key_id'), oss_asset_info.get('oss_access_key_secret'))
    bucket = oss2.Bucket(auth, oss_asset_info.get('oss_endpoint'), name_bucket)

    bucket.put_object_from_file(os.path.join(url_dataset_path, fname_out), fname_in)

    logger.info("Saving file: [{}] to OSS path: [{}].".format(fname_in, url_dataset_path + fname_out))

    return


def copy_model_from_http(checkpoint_path):
    base_folder = os.path.join(os.path.abspath(os.sep), '/data/pytorch_models')
    target_file = os.path.join(base_folder, 'prev_checkpoint.tar')
    if dist.is_master():
        try:
            os.makedirs(base_folder)
        except:
            pass

        str_cpy = 'curl ' + checkpoint_path + ' --output ' + target_file
        logger.info(f"Downloading model from HTTP: {checkpoint_path}")
        logger.info("RUNNING: " + str_cpy)

        subprocess.check_output(str_cpy, shell=True)
    dist.set_barrier()

    return target_file


def get_oss_dataset(oss_dataset_path, oss_endpoint, oss_access_key, oss_secret, oss_make_index, oss_val_dir,
                    train_transform,
                    valid_transform, check_corrupted_files=False):
    """
    Retrieves the Dataset (torchvision) data structure using utils.get_data utility function
    :param dset_name:
    :param data:
    :param is_ImageFolder:
    :param train_transform:
    :param valid_transform:
    :return:
    """
    set_dataloader_batch_fetch()
    train_data = ImageFolderOss(oss_dataset_path + 'train/', oss_endpoint, access_key=oss_access_key,
                                secret=oss_secret, transform=train_transform,
                                from_index=not oss_make_index, make_index=oss_make_index,
                                check_corrupted_files=check_corrupted_files)
    valid_data = ImageFolderOss(oss_dataset_path + oss_val_dir, oss_endpoint, access_key=oss_access_key,
                                secret=oss_secret, transform=valid_transform,
                                from_index=not oss_make_index, make_index=oss_make_index,
                                check_corrupted_files=check_corrupted_files)
    return train_data, valid_data
