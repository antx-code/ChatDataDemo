import os
import ipaddress
import hashlib
import json
from loguru import logger


@logger.catch
def is_file_or_directory(path):
    """
    判断给定的path是文件还是目录

    :param path: 给定的path
    :return: 如果是文件返回True，否则返回False

    """
    if os.path.isfile(path):
        return 'file'
    elif os.path.isdir(path):
        return 'directory'
    else:
        return None


@logger.catch
def is_private_ip(ip: str) -> bool:
    """
    Check if an IP address is a private IP address.
    """
    try:
        addr = ipaddress.ip_address(ip)
        return addr.is_private
    except ValueError:
        return False


@logger.catch(level='ERROR')
def md5H(_str):
    hl = hashlib.md5()
    hl.update(_str.encode(encoding="utf-8"))
    return hl.hexdigest()


@logger.catch(level='ERROR')
def json_md5H(result):
    return md5H(json.dumps(result, ensure_ascii=False))
