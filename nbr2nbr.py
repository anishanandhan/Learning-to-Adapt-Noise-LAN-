import torch

operation_seed_counter = 0

def get_generator(device):
    """
    Proper generator creation that works for CUDA, MPS, or CPU devices.
    """
    global operation_seed_counter
    operation_seed_counter += 1

    # Normalize device type
    dev_type = device.type if isinstance(device, torch.device) else str(device)
    if dev_type not in ["cuda", "mps", "cpu"]:
        dev_type = "cpu"

    g_cuda_generator = torch.Generator(device=dev_type)
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size, w // block_size)


def generate_mask_pair(img, device):
    """
    Generate two complementary random masks directly on the *image's device*.
    """
    n, c, h, w = img.shape
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4,), dtype=torch.bool, device=img.device)
    mask2 = torch.zeros_like(mask1, dtype=torch.bool, device=img.device)

    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64, device=img.device
    )

    rd_idx = torch.zeros(size=(n * h // 2 * w // 2,), dtype=torch.int64, device=img.device)

    # 🧩 FIX: generator now uses img.device, not external device
    torch.randint(
        low=0, high=8,
        size=(n * h // 2 * w // 2,),
        generator=get_generator(img.device),
        out=rd_idx
    )

    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(
        start=0, end=n * h // 2 * w // 2 * 4, step=4,
        dtype=torch.int64, device=img.device
    ).reshape(-1, 1)

    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2


def generate_subimages(img, mask):
    n, c, h, w = img.shape
    subimage = torch.zeros(
        n, c, h // 2, w // 2,
        dtype=img.dtype, layout=img.layout, device=img.device
    )

    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n, h // 2, w // 2, 1
        ).permute(0, 3, 1, 2)

    return subimage


def loss_func(noisy, network, tmp_iter, max_iter, device):
    """
    Neighbor2Neighbor self-supervised loss with RGB<->BGR conversion handling.
    """
    mask1, mask2 = generate_mask_pair(noisy, noisy.device)

    noisy_sub1 = generate_subimages(noisy, mask1)
    noisy_sub2 = generate_subimages(noisy, mask2)

    with torch.no_grad():
        noisy_denoised = network(noisy).clip(0, 1)

    noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1)
    noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2)

    noisy_output = network(noisy_sub1).clip(0, 1)
    noisy_target = noisy_sub2

    Lambda = tmp_iter / max_iter * 0.1
    diff = noisy_output - noisy_target
    exp_diff = noisy_sub1_denoised - noisy_sub2_denoised

    loss1 = torch.mean(diff**2)
    loss2 = Lambda * torch.mean((diff - exp_diff)**2)
    return loss1 + loss2
  