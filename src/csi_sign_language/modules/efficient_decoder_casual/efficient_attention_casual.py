import torch
from torch import nn
from einops import rearrange
from torch._prims_common import Tensor


class BucketRandomAttentionCausal(nn.Module):
    """
    This Module divdes the temporal dimension into buckets and only samples one element from each bucket.
    Implemented by torch.nn.Multihead, which is native self-attention, was faster
    """

    def __init__(
        self, d_model, num_heads, bucket_size, future=0, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads,
        )
        self.bucket_size = bucket_size
        self.future = future

    @staticmethod
    def split_list_into_groups(lst, group_size):
        return [lst[i : i + group_size] for i in range(0, len(lst), group_size)]

    @staticmethod
    def modify_key_length(key_length, sampled_index):
        """
        @param key_length: [b]
        @param sampled_index: [L]
        """
        sampled_index = rearrange(sampled_index, "l -> 1 l")
        key_length = rearrange(key_length, "b -> b 1")

        ret = sampled_index < key_length
        ret = ret.int()
        ret = torch.sum(ret, dim=1, keepdim=False)
        return ret

    # NOTE: old version is not efficient
    #
    # def sample_index(self, Lk, device):
    #     groups = self.split_list_into_groups(list(range(Lk)), self.bucket_size)
    #     sampled_index = [random.choice(group) for group in groups]
    #     sampled_index = sorted(sampled_index)
    #     sampled_index = torch.tensor(sampled_index, dtype=torch.int64, device=device)
    #     return sampled_index
    #
    @torch.no_grad()
    def sample_index(self, Lk, device):
        array = torch.arange(Lk, dtype=torch.int64, device=device)

        # split buketes
        # Number of full buckets
        num_full_buckets = array.size(0) // self.bucket_size
        # Size of the remaining part
        remaining_elements = array.size(0) % self.bucket_size

        # Process full buckets
        buckets = array[: num_full_buckets * self.bucket_size].view(
            -1, self.bucket_size
        )
        indices = torch.randint(0, self.bucket_size, (buckets.size(0),))
        sampled_elements = buckets[torch.arange(buckets.size(0)), indices]

        # Process incomplete buckets (if there are remaining elements)
        if remaining_elements > 0:
            remaining_bucket = array[num_full_buckets * self.bucket_size :]
            sampled_remaining = remaining_bucket[
                torch.randint(0, remaining_elements, (1,))
            ]
            sampled_elements = torch.cat((sampled_elements, sampled_remaining))
        return sampled_elements

    @staticmethod
    def _make_key_padding_mask(t_length: torch.Tensor, temporal_dim):
        """
        @param t_length: [B]
        @param temporal_dim: int, the max length of the temporal dimension
        """
        B = t_length.size(dim=0)
        mask = torch.range(0, temporal_dim - 1, device=t_length.device)
        mask = rearrange(mask, "t -> 1 t")
        t_length = rearrange(t_length, "b -> b 1")

        mask = mask >= t_length
        return mask

    # NOTE: only current indexes lower than lengths + future is avialiable, note that True means not attend
    def _make_casual_mask_sampled(
        self, Lq: int, Lk: int, sampled_index: Tensor, future=0
    ):
        assert Lq == Lk, "casual mask only works when using self attention"
        device = sampled_index.device
        origin_index = torch.arange(Lq, device=device)
        origin_index = rearrange(origin_index, "l -> l 1")
        sampled_index = rearrange(sampled_index, "s -> 1 s")

        mask = origin_index + future < sampled_index
        return mask

    def forward(self, q, k, v, key_length=None):
        # [t n c]

        Lk = k.shape[0]
        Lq = q.shape[0]

        if self.bucket_size > 1:
            sampled_index = self.sample_index(Lk, q.device)
            modified_key_length = self.modify_key_length(key_length, sampled_index)

            k = k[sampled_index, :, :]
            v = v[sampled_index, :, :]

        elif self.bucket_size == 1:
            modified_key_length = key_length

        else:
            raise ValueError("Bucket size must be greater than 0")

        if modified_key_length is not None:
            mask = self._make_key_padding_mask(modified_key_length, k.size(dim=0))
        else:
            mask = None

        if self.bucket_size == 1:
            # just as original causal implementation in transformer
            return self.attn(q, k, v, is_causal=True, key_padding_mask=mask)
        if self.bucket_size > 1:
            # her because we modified the keys, thus need new causal mask
            attn_mask = self._make_casual_mask_sampled(
                Lq, Lk, sampled_index, future=self.future
            )
            return self.attn(q, k, v, attn_mask=attn_mask, key_padding_mask=mask)


if __name__ == "__main__":
    attn = BucketRandomAttentionCausal(
        d_model=256,
        num_heads=8,
        bucket_size=2,
        future=1,
    )
    qkv = torch.rand(23, 2, 256)
    result, _ = attn(qkv, qkv, qkv, key_length=torch.tensor([10, 20]))
    print(result.shape)
