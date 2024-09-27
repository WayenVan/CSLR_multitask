import torch


bucket_size = 4
Lk = 50
device = "cuda:0"

array = torch.arange(Lk, device=device)
print(array)

# split buketes
# Number of full buckets
num_full_buckets = array.size(0) // bucket_size
# Size of the remaining part
remaining_elements = array.size(0) % bucket_size

# Process full buckets
buckets = array[: num_full_buckets * bucket_size].view(-1, bucket_size)
indices = torch.randint(0, bucket_size, (buckets.size(0),))
sampled_elements = buckets[torch.arange(buckets.size(0)), indices]

# Process incomplete buckets (if there are remaining elements)
if remaining_elements > 0:
    remaining_bucket = array[num_full_buckets * bucket_size :]
    sampled_remaining = remaining_bucket[torch.randint(0, remaining_elements, (1,))]
    sampled_elements = torch.cat((sampled_elements, sampled_remaining))

print(sampled_elements)
