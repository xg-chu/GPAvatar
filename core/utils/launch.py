# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import logging
from datetime import timedelta
import torch
import torch.multiprocessing as mp

import core.utils.distributed as dist

__all__ = ["DEFAULT_TIMEOUT", "launch"]

DEFAULT_TIMEOUT = timedelta(minutes=30)


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def launch_local(main_func, devices, timeout=DEFAULT_TIMEOUT, **kwargs):
    world_size = len(devices)
    if world_size > 1:
        port = _find_free_port()
        dist_url = f"tcp://127.0.0.1:{port}"
        torch.multiprocessing.spawn(
            _distributed_worker,
            nprocs=world_size,
            args=(main_func, world_size, dist_url, timeout, kwargs),
            daemon=False,
        )
    else:
        main_func(**kwargs)


def _distributed_worker(local_rank, main_func, world_size, dist_url, timeout=DEFAULT_TIMEOUT, kwargs=None):
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    try:
        torch.distributed.init_process_group(
            backend="NCCL",
            init_method=dist_url,
            world_size=world_size,
            rank=local_rank,
            timeout=timeout,
        )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Process group URL: {}".format(dist_url))
        raise e
    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    torch.cuda.set_device(local_rank)
    # Setup the local process group (which contains ranks within the same machine)
    assert dist._LOCAL_PROCESS_GROUP is None
    ranks_on_this = list(range(world_size))
    pg = torch.distributed.new_group(ranks_on_this)
    dist._LOCAL_PROCESS_GROUP = pg

    dist.synchronize()
    main_func(**kwargs)
