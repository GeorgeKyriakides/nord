import torch


class DistributedConfig:

    def __init__(self, master_address: str, master_port: str, world_size: int, world_rank: int, debug_level: str = 'DETAIL', backend: str = 'nccl') -> None:
        self.master_address = master_address
        self.master_port = master_port
        self.world_size = world_size
        self.world_rank = world_rank
        self.debug_level = debug_level
        self.local_rank = world_rank % torch.cuda.device_count()
        self.backend = backend

    def __str__(self) -> str:
        return ('M. Address : '+self.master_address +
                ', M. Port: '+self.master_port +
                ', World Size: '+str(self.world_size) +
                ', World Rank: '+str(self.world_rank) +
                ', Local Rank: '+str(self.local_rank) +
                ', Backend: '+self.backend)
