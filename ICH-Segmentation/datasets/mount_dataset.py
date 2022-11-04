import os
from dotenv import load_dotenv

load_dotenv()


def umount(mount_dir):
    if os.path.ismount(mount_dir):
        os.system(f"umount {mount_dir}")


def _mount(host, remote_dir, local_dir):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
    os.system(f"mount -t nfs {host}:{remote_dir} {local_dir}")


def mount_ich_420_dataset(mount_dir):
    if not os.path.ismount(mount_dir):
        _mount(os.getenv("NFS_HOST"), os.getenv("NFS_ICH420_DIR"), mount_dir)
