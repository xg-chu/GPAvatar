# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import torchmetrics

calc_psnr = torchmetrics.functional.image.peak_signal_noise_ratio
calc_ssim = torchmetrics.functional.image.structural_similarity_index_measure
# lpips_model = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity
