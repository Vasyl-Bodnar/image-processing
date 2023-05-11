"""
General algorithms that I used throughout the semester
Many are somewhat outdated in their design considiring my experience with them, were made slopily, and may not even work
Requires 3.10 python as I use a couple of match statements
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage as si
from scipy.stats import skew, kurtosis, entropy
from multiprocess import pool

def get_img(path):
    return cv2.imread(path, 1)

def get_gray_img(path):
    return cv2.imread(path, 0)

def gauss_kernel(ksize, s):
    return np.outer(cv2.getGaussianKernel(ksize, s), cv2.getGaussianKernel(ksize, s))

def gauss_filter(img, ksize, s):
    return cv2.filter2D(img, -1, gauss_kernel(ksize, s))

def switch_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def calculate_lbp_feature(img, kernel_size=3, n_points=8):
    lbp = np.zeros_like(img)
    center_points = [
        (i, j)
        for i in range(kernel_size, img.shape[0] - kernel_size)
        for j in range(kernel_size, img.shape[1] - kernel_size)
    ]

    def calculate_lbp(center):
        i, j = center
        center_value = img[i, j]
        code = 0
        for k in range(n_points):
            x = i + int((kernel_size - 1) * np.cos(2 * np.pi * k / n_points))
            y = j - int((kernel_size - 1) * np.sin(2 * np.pi * k / n_points))
            if img[x, y] > center_value:
                code += 2**k
        return code

    with pool.Pool(12) as p:
        codes = p.map(calculate_lbp, center_points)

    for i, center in enumerate(center_points):
        lbp[center[0], center[1]] = codes[i]

    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(2**n_points + 1))
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-7
    feature_vector = hist.flatten()
    return feature_vector


def intensity_feats(hist):
    mean = np.mean(hist)
    median = np.median(hist)
    var = np.var(hist)
    smoothness = np.sum(np.abs(np.diff(hist, n=2)))
    skewness = skew(hist)
    kurt = kurtosis(hist)
    ent = entropy(hist)
    return [mean, median, var, smoothness, skewness[0], kurt[0], ent[0]]  # type:ignore


def area_feat(img):
    contours = sorted(find_contours(img), key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]
    return (
        cv2.contourArea(largest_contour),
        cv2.drawContours(
            np.zeros_like(img) + 255, [largest_contour], -1, (0, 255, 0), 2
        ),
        cv2.arcLength(largest_contour, True),
    )


def generate_chain_code(img):
    largest_contour = max(find_contours(img), key=cv2.contourArea)
    chain = []
    prev_point = largest_contour[0][0]
    for point in largest_contour[1:]:
        point = point[0]
        diff = point - prev_point
        match diff[0], diff[1]:
            case 1, 0:
                chain.append(0)
            case 1, -1:
                chain.append(1)
            case 0, -1:
                chain.append(2)
            case -1, -1:
                chain.append(3)
            case -1, 0:
                chain.append(4)
            case -1, 1:
                chain.append(5)
            case 0, 1:
                chain.append(6)
            case 1, 1:
                chain.append(7)
        prev_point = point
    return chain


def find_contours(img):
    edges = enh_seg(img)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def enhance(img):
    return lut_stretch(img)


def segment(img):
    return otsu(img)


def enh_seg(img):
    return enhance(img) * segment(img)


def box_counting_dim_single(
    img, grid_sizes=[2, 4, 8, 16, 32, 64, 128, 256], apply=enh_seg
):
    height = 600 # This constant should not exist
    img_resized = cv2.resize(img, (int(height * img.shape[1] / img.shape[0]), height))
    img_bw = apply(img_resized)
    counts = []
    for size in grid_sizes:
        count = 0
        for i in range(0, img_bw.shape[0] - size, size):
            for j in range(0, img_bw.shape[1] - size, size):
                if np.sum(img_bw[i : i + size, j : j + size]) > 0:
                    count += 1
        counts.append(count)
    log_sizes = np.log(grid_sizes)
    log_counts = np.log(np.array(counts) + 1)
    return (log_counts[-1] - log_counts[0]) / (log_sizes[-1] - log_sizes[0])


def box_counting_dim(
    ref_img, test_imgs, grid_sizes=[2, 4, 8, 16, 32, 64, 128, 256], apply=enh_seg
):
    test_imgs.append(ref_img)
    height = 600 # This constant should not exist
    counts_list = []
    for img in test_imgs:
        img_resized = cv2.resize(
            img, (int(height * img.shape[1] / img.shape[0]), height)
        )
        img_bw = apply(img_resized)
        counts = []
        for size in grid_sizes:
            count = 0
            for i in range(0, img_bw.shape[0] - size, size):
                for j in range(0, img_bw.shape[1] - size, size):
                    if np.sum(img_bw[i : i + size, j : j + size]) > 0:
                        count += 1
            counts.append(count)
        counts_list.append(counts)
    log_sizes = np.log(grid_sizes)
    log_counts_ref = np.log(counts_list[-1])
    slope_list = []
    for counts in counts_list[:-1]:
        log_counts = np.log(np.array(counts) + 1e-20)
        slope = (log_counts[-1] - log_counts[0]) / (log_sizes[-1] - log_sizes[0]) - (
            log_counts_ref[-1] - log_counts_ref[0]
        ) / (log_sizes[-1] - log_sizes[0])
        slope_list.append(slope)
    return slope_list


def make_form(k, form):
    match form:
        case "square":
            return cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        case "cross":
            return cv2.getStructuringElement(cv2.MORPH_CROSS, (k, k))
        case "xshape":
            return np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.uint8)  # non-k
        case "circ":
            return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)  # non-k


def erode(img, k=3, form="cross"):
    return cv2.erode(img, make_form(k, form))


def dilate(img, k=3, form="cross"):
    return cv2.dilate(img, make_form(k, form))


def opening(img, k=3, form="cross"):
    return dilate(erode(img, k, form), k, form)


def closing(img, k=3, form="cross"):
    return erode(dilate(img, k, form), k, form)


def morph_grad(img, k1=3, form1="cross", k2=None, form2=None):
    return dilate(img, k1, form1) - erode(
        img, k1 if k2 is None else k2, form1 if form2 is None else form2
    )


def top_hat(img, k=3, form="cross"):
    return (img - opening(img, k, form).astype(np.uint8)).astype(np.uint8)


def bot_hat(img, k=3, form="cross"):
    return (img - closing(img, k, form).astype(np.uint8)).astype(np.uint8)


def any_neighbor_zero(img, i, j):
    for k in range(-1, 2):
        for _ in range(-1, 2):
            if img[i + k, j + k] == 0:
                return True
    return False


def zero_crossing(img):
    img[img > 0] = 1
    img[img < 0] = 0
    out_img = np.zeros(img.shape)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if img[i, j] > 0 and any_neighbor_zero(img, i, j):
                out_img[i, j] = 255
    return out_img


def laplacian(img, s=1, T=30):
    smoothed = si.filters.gaussian(img, s)
    laplacian = cv2.Laplacian(smoothed, cv2.CV_64F)
    x, y = laplacian.shape
    edges = np.zeros((x, y))
    w = edges.shape[1]
    h = edges.shape[0]
    for n in range(0, x):
        for m in range(0, y):
            if laplacian[n, m] > 0:
                edges[n, m] = 255
            elif laplacian[n, m] < 0:
                edges[n, m] = -255
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            patch = laplacian[y : y + 2, x : x + 1]
            p = laplacian[y, x]
            maxP = patch.max()
            minP = patch.min()
            if p > 0:
                zeroCross = True if minP < 0 else False
            else:
                zeroCross = True if minP > 0 else False
            if ((maxP - minP) > T) and zeroCross:
                edges[y, x] = 255
    return smoothed * 255, laplacian * 255, edges  # type:ignore


def gaussian_edge(img, s=2, T=50):
    smoothed = si.filters.gaussian(img, s)
    sigma = s
    x, y = np.meshgrid(
        np.linspace(-3 * sigma, 3 * sigma, 7), np.linspace(-3 * sigma, 3 * sigma, 7)
    )
    g = -((x**2 + y**2 - 2 * sigma**2) / (2 * np.pi * sigma**4)) * np.exp(
        -(x**2 + y**2) / (2 * sigma**2)
    )
    laplacian = cv2.filter2D(smoothed, -1, g)
    x, y = laplacian.shape
    edges = np.zeros((x, y))
    w = edges.shape[1]
    h = edges.shape[0]
    for n in range(0, x):
        for m in range(0, y):
            if laplacian[n, m] > 0:
                edges[n, m] = 255
            elif laplacian[n, m] < 0:
                edges[n, m] = 0
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            patch = laplacian[y : y + 2, x : x + 1]
            p = laplacian[y, x]
            maxP = patch.max()
            minP = patch.min()
            if p > 0:
                zeroCross = True if minP < 0 else False
            else:
                zeroCross = True if minP > 0 else False
            if ((maxP - minP) > T) and zeroCross:
                edges[y, x] = 255
    return smoothed * 255, laplacian * 255, edges  # type: ignore


def sobel(img, T):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx**2 + sobely**2)
    ang = np.arctan2(sobely, sobelx) * 180 / np.pi
    return sobelx, sobely, mag, ang, global_thresh(mag, T)


def canny(img, T1, T2=None):
    return cv2.Canny(img, T1, T1 * 3 if T2 is None else T2)


def robinson(img, T):
    kernels = [
        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),  # N-S
        np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]),  # NE-SW
        np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),  # E-W
        np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]]),  # SE-NW
        np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),  # S-N
        np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]]),  # SW-NE
        np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),  # W-E
        np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]),  # NW-SE
    ]
    mag = np.zeros_like(img)
    ang = np.zeros_like(img)
    for kernel in kernels:
        conv = cv2.filter2D(img, -1, kernel)
        mag = np.maximum(mag, conv)
        ang[conv == mag] = np.arctan2(kernel[1, 0], kernel[0, 0])
    return mag, ang, global_thresh(mag, T)


def hog(img):
    return si.feature.hog(img, visualize=True)[1]


def multilevel_thresh(img, threshold_values):
    img[img < threshold_values[0]] = 0
    img[(img < threshold_values[1]) & (img > 0)] = 125
    img[img > 125] = 255
    return img


def adaptive_thresh(img, split=11):
    return cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, split, 2
    )


def otsu(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def global_thresh(img, T):
    fin = np.zeros(img)
    fin[img >= T] = 255
    return fin


def niblack(img, window_size=30, k=-0.3):
    mean = cv2.blur(img, (window_size, window_size))
    variance = cv2.boxFilter(
        np.square(img), -1, (window_size, window_size)
    ) - np.square(mean)
    threshold = mean + k * np.sqrt(variance)
    binary = img >= threshold
    return binary.astype(np.uint8) * 255


def bernsen(img, window_size=30, c=15):
    min_vals = cv2.erode(img, np.ones((window_size, window_size)))
    max_vals = cv2.dilate(img, np.ones((window_size, window_size)))
    contrast = max_vals - min_vals
    threshold = np.where(contrast <= c, (min_vals + max_vals) // 2, img > max_vals / 2)
    binary = threshold.astype(np.uint8) * 255
    return binary


def sauvola(image, window_size=30, k=0.34, r=128):
    mean = cv2.blur(image, (window_size, window_size))
    variance = cv2.boxFilter(
        np.square(image), -1, (window_size, window_size)
    ) - np.square(mean)
    std_dev = np.sqrt(np.maximum(variance, 0))
    threshold = mean * (1 + k * ((std_dev / r) - 1))
    binary = image >= threshold
    return binary.astype(np.uint8) * 255


def alpha_trim_mean(img, alpha=2, kernels=[5, 7]):
    result = []
    for k in kernels:
        filtered = np.zeros_like(img)
        for i in range(k // 2, img.shape[0] - k // 2):
            for j in range(k // 2, img.shape[1] - k // 2):
                neighborhood = img[
                    i - k // 2 : i + k // 2 + 1, j - k // 2 : j + k // 2 + 1
                ]
                flat_neighborhood = neighborhood.flatten()
                sorted_neighborhood = np.sort(flat_neighborhood)
                trimmed_neighborhood = sorted_neighborhood[alpha:-alpha]
                mean_value = np.mean(trimmed_neighborhood)
                filtered[i, j] = mean_value
        result.append(filtered)
    return result


def weighted_median(
    img,
    kernels=[
        (3, np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])),
        (
            7,
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 2, 2, 2, 2, 2, 1],
                    [1, 2, 3, 3, 3, 2, 1],
                    [1, 2, 3, 4, 3, 2, 1],
                    [1, 2, 3, 3, 3, 2, 1],
                    [1, 2, 2, 2, 2, 2, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                ]
            ),
        ),
    ],
):
    result = []
    for kernel in kernels:
        filtered = np.zeros_like(img)
        window_size, weight_kernel = kernel
        for i in range(window_size // 2, img.shape[0] - window_size // 2):
            for j in range(window_size // 2, img.shape[1] - window_size // 2):
                neighborhood = img[
                    i - window_size // 2 : i + window_size // 2 + 1,
                    j - window_size // 2 : j + window_size // 2 + 1,
                ]
                flat_neighborhood = neighborhood.flatten()
                flat_weights = weight_kernel.flatten()
                sorted_indices = np.argsort(flat_neighborhood)
                sorted_neighborhood = flat_neighborhood[sorted_indices]
                sorted_weights = flat_weights[sorted_indices]
                cumsum_weights = np.cumsum(sorted_weights)
                median_index = np.searchsorted(cumsum_weights, cumsum_weights[-1] // 2)
                median_value = sorted_neighborhood[median_index]
                filtered[i, j] = median_value
        result.append(filtered)
    return result


def midpoint_filter(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.float32)
    min_values = cv2.erode(img, kernel)
    max_values = cv2.dilate(img, kernel)
    return (min_values + max_values) // 2


def geometric_filter(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    k = kernel_size * kernel_size
    img_log = np.log(img.astype(np.float32) + 1)
    img_filtered_log = cv2.filter2D(img_log, -1, kernel)
    img_filtered_log = img_filtered_log / k
    img_filtered = np.exp(img_filtered_log) - 1
    img_filtered = np.clip(img_filtered, 0, 255)
    return img_filtered.astype(np.uint8)


def arithm_filter(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.float32)
    kernel /= kernel_size * kernel_size
    return cv2.filter2D(img, -1, kernel)


def weighted_filter(img, kernel_size=3, weights=[[1, 2, 1], [2, 4, 2], [1, 2, 1]]):
    weights = np.array(weights).flatten()
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] = weights[i * kernel_size + j]
    kernel = kernel / np.sum(kernel)
    return cv2.filter2D(img, -1, kernel)


def wf_3x3(img):
    return weighted_filter(img)


def wf_7x7(img):
    return weighted_filter(
        img,
        7,
        [
            [1, 1, 2, 2, 4, 2, 2, 1, 1],
            [1, 2, 2, 4, 8, 4, 2, 2, 1],
            [2, 2, 4, 8, 16, 8, 4, 2, 2],
            [2, 4, 8, 16, 32, 16, 8, 4, 2],
            [4, 8, 16, 32, 64, 32, 16, 8, 4],
            [2, 4, 8, 16, 32, 16, 8, 4, 2],
            [1, 2, 2, 4, 8, 4, 2, 2, 1],
            [1, 1, 2, 2, 4, 2, 2, 1, 1],
        ],
    )


def wf_9x9(img):
    return weighted_filter(
        img,
        9,
        [
            [1, 1, 1, 2, 2, 4, 2, 1, 1],
            [1, 1, 2, 2, 4, 8, 4, 2, 1],
            [1, 2, 2, 4, 8, 16, 8, 2, 1],
            [2, 2, 4, 8, 16, 32, 16, 4, 2],
            [2, 4, 8, 16, 32, 64, 32, 8, 4],
            [4, 8, 16, 32, 64, 128, 64, 16, 8],
            [2, 4, 8, 16, 32, 64, 32, 8, 4],
            [1, 2, 2, 4, 8, 16, 8, 2, 1],
            [1, 1, 2, 2, 4, 8, 4, 2, 1],
        ],
    )


def wf_15x15(img):
    return weighted_filter(
        img,
        15,
        [
            [2, 2, 2, 2, 2, 2, 2],
            [2, 3, 3, 3, 3, 3, 2],
            [2, 3, 4, 4, 4, 3, 2],
            [2, 3, 4, 5, 4, 3, 2],
            [2, 3, 4, 4, 4, 3, 2],
            [2, 3, 3, 3, 3, 3, 2],
            [2, 2, 2, 2, 2, 2, 2],
        ],
    )


def estimate_noise(img, region):
    if region is not None:
        x, y, w, h = region
        roi = img[y : y + h, x : x + w]
        mean, std = cv2.meanStdDev(roi)
        variance = std**2
        return (mean[0][0], variance[0][0])
    mean, std = cv2.meanStdDev(img)
    variance = std**2
    return (mean[0][0], variance[0][0])


# Combine Gauss and Salt and Pepper noise
def combo_gauss_sp_noise(img, mean=1, var=50, p=0.05):
    height, width = img.shape[:2]
    mask = np.random.choice([0, 1, 2], size=(height, width), p=[1 - p - p, p, p])
    noise = np.zeros_like(img)
    noise[mask == 0] = 120
    noise[mask == 1] = 255
    noise[mask == 2] = 0

    noisy_image = np.copy(img)
    noisy_image[mask == 1] = 255
    noisy_image[mask == 2] = 0
    noisy_image1 = noisy_image

    gaussian = np.random.normal(mean, var, img.shape)
    noisy_image = img + gaussian
    noisy_image2 = np.clip(noisy_image, 0, 255).astype(np.uint8)

    noisy_image = noisy_image1 + gaussian
    return (noisy_image, (noisy_image1, noise), (noisy_image2, gaussian))


# Combine Gauss and Salt and Pepper noises, do not return noises
def combo_gauss_sp(img, mean=1, var=50, p=0.05):
    noisy_image, (noisy_image1, _), (noisy_image2, _) = combo_gauss_sp_noise(
        img, mean, var, p
    )
    return noisy_image, noisy_image1, noisy_image2


def gaussian_noise(img, mean=1, var=50):
    gaussian = np.random.normal(mean, var, img.shape)
    noisy_image = img + gaussian
    noisy_image = np.clip(noisy_image, 0, 255)
    return cv2.convertScaleAbs(noisy_image)


# Add salt and pepper noise
def s_and_p_noise(image, pa=0.05, pb=0.05):
    height, width = image.shape[:2]
    mask = np.random.choice([0, 1, 2], size=(height, width), p=[1 - pa - pb, pa, pb])
    noisy_image = np.copy(image)
    noisy_image[mask == 1] = 255
    noisy_image[mask == 2] = 0
    return noisy_image


def salt_noise(img, prob=0.05):
    noise = np.zeros_like(img)
    mask = np.random.choice([0, 1], size=img.shape[:2], p=[1 - prob, prob])
    noise[mask == 1] = 255
    noise[mask == 0] = 0
    noisy_image = cv2.add(img, noise)
    noisy_image = np.clip(noisy_image, 0, 255)
    return cv2.convertScaleAbs(noisy_image)


# Gray from constants
def gray(img, r, g, b):
    B, G, R = cv2.split(img)
    gray = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            gray[i, j] = r * R[i, j] + g * G[i, j] + b * B[i, j]
    return gray


def draw_hist(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.xlabel("Gray level")
    plt.ylabel("Frequency")
    plt.show()


# Frequently used histogram saving function
def save_hist(img, name, excp=None):
    img = cv2.convertScaleAbs(img)
    hist = cv2.calcHist(
        [img], [0], None, [256], [0, 119, 121, 256] if excp is not None else [0, 256]
    )
    plt.plot(hist)
    plt.xlabel("Gray level")
    plt.ylabel("Frequency")
    plt.savefig("hist/" + name)
    plt.close()


# Apply rgb_to_hsv on image
def rgb_img_to_hsv(img):
    hsv = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hsv[i, j] = rgb_to_hsv(img[i, j])
    return hsv


def rgb_to_hsv(rgb):
    r, g, b = rgb / 255.0
    max_val = np.max([r, g, b])
    min_val = np.min([r, g, b])
    if max_val == min_val:
        h = 0.0
    elif max_val == r:
        h = ((g - b) / (max_val - min_val)) % 6.0
    elif max_val == g:
        h = (b - r) / (max_val - min_val) + 2.0
    else:
        h = (r - g) / (max_val - min_val) + 4.0
    h = h * 60.0
    if h < 0:
        h += 360
    if max_val == 0.0:
        s = 0.0
    else:
        s = (max_val - min_val) / max_val
    v = max_val
    return np.array([h / 2, s, v])


def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    match i:
        case 0:
            return v, t, p
        case 1:
            return q, v, p
        case 2:
            return p, v, t
        case 3:
            return p, q, v
        case 4:
            return t, p, v
        case 5:
            return v, p, q


# Broken minmax stretching
def mm_stretch(img):
    minmax_img = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    min = np.min(img)
    max_min = np.max(img) - min
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            minmax_img[i, j] = (255 * (img[i, j] - min)) / max_min
    return minmax_img


# LUT contrast stretching
def lut_stretch(img):
    xp = [0, 64, 128, 192, 255]
    fp = [0, 16, 128, 240, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype("uint8")
    return cv2.LUT(img, table)


# LUT gamma correction
def gamma_correct(img, gamma):
    invGamma = 1 / gamma
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv2.LUT(img, table)


# Basic histogram making
def make_hist(img):
    histogram = np.zeros(256)

    for pixel in img:
        histogram[pixel] += 1

    return histogram


def cumm_sum(a):
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)


def normalize(cs):
    return (255 * (cs - cs.min())) / (cs.max() - cs.min())


# Basic histogram equalization
def hist_equal(img):
    flat_img = img.flatten()
    equal_img = normalize(cumm_sum(make_hist(flat_img))).astype("uint8")[flat_img]
    return np.reshape(equal_img, img.shape)


def intensity_stretch(img, r1, s1, r2, s2):
    def pix_manpl(pix):
        if 0 <= pix and pix <= r1:
            return (s1 / r1) * pix
        elif r1 < pix and pix <= r2:
            return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
        else:
            return ((255 - s2) / (255 - r2)) * (pix - r2) + s2

    return np.vectorize(pix_manpl)(img)


def dist(hist1, hist2):
    sum = 0
    for i in range(256):
        sum += (hist1[i] - hist2[i]) ** 2
    return np.sqrt(sum)


# TODO: All Retinex code needs to be credited
def singleScaleRetinex(img, variance):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), variance))
    return retinex


def multiScaleRetinex(img, variance_list):
    retinex = np.zeros_like(img)
    for variance in variance_list:
        retinex += singleScaleRetinex(img, variance)
    retinex = retinex / len(variance_list)
    return retinex


def MSR(img, variance_list):
    img = np.float64(img) + 1.0
    img_retinex = multiScaleRetinex(img, variance_list)

    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(
            np.int32(img_retinex[:, :, i] * 100), return_counts=True
        )
        zero_count = 1
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break
        img_retinex[:, :, i] = np.maximum(
            np.minimum(img_retinex[:, :, i], high_val), low_val
        )

        img_retinex[:, :, i] = (
            (img_retinex[:, :, i] - np.min(img_retinex[:, :, i]))
            / (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i]))
            * 255
        )
    img_retinex = np.uint8(img_retinex)
    return img_retinex


def SSR(img, variance):
    img = np.float64(img) + 1.0
    img_retinex = singleScaleRetinex(img, variance)
    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(
            np.int32(img_retinex[:, :, i] * 100), return_counts=True
        )
        zero_count = 0
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break
        img_retinex[:, :, i] = np.maximum(
            np.minimum(img_retinex[:, :, i], high_val), low_val
        )

        img_retinex[:, :, i] = (
            (img_retinex[:, :, i] - np.min(img_retinex[:, :, i]))
            / (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i]))
            * 255
        )
    img_retinex = np.uint8(img_retinex)
    return img_retinex


def interpolate(subBin, LU, RU, LB, RB, subX, subY):
    """
    Image interpolation function, need to be applied after CLAHE
    """
    subImage = np.zeros(subBin.shape)
    num = subX * subY
    for i in range(subX):
        inverseI = subX - i
        for j in range(subY):
            inverseJ = subY - j
            val = subBin[i, j].astype(int)
            subImage[i, j] = np.floor(
                (
                    inverseI * (inverseJ * LU[val] + j * RU[val])
                    + i * (inverseJ * LB[val] + j * RB[val])
                )
                / float(num)
            )
    return subImage


# FIX: Divide into sep fns
# TODO: Credit
def clahe(img, clipLimit, nrBins=128, nrX=0, nrY=0):
    """
    CLAHE algorithm implementation
    :param img: Input image
    :param clipLimit: Normalized clipLimit. Higher value gives more contrast
    :param nrBins: Number of gray level bins for histogram("dynamic range")
    :param nrX: Number of contextual regions in X direction
    :param nrY: Number of contextual regions in Y direction
    """
    h, w = img.shape
    if clipLimit == 1:
        return
    nrBins = max(nrBins, 128)
    if nrX == 0:
        # Taking dimensions of each contextial region to be a square of 32X32
        xsz = 32
        ysz = 32
        nrX = np.ceil(h / xsz)  # 240
        # Excess number of pixels to get an integer value of nrX and nrY
        excX = int(xsz * (nrX - h / xsz))
        nrY = np.ceil(w / ysz)  # 320
        excY = int(ysz * (nrY - w / ysz))
        # Pad that number of pixels to the image
        if excX != 0:
            img = np.append(img, np.zeros((excX, img.shape[1])).astype(int), axis=0)
        if excY != 0:
            img = np.append(img, np.zeros((img.shape[0], excY)).astype(int), axis=1)
    else:
        xsz = np.round(h / nrX)
        ysz = np.round(w / nrY)
        excX = int(xsz * (nrX - h / xsz))
        excY = int(ysz * (nrY - w / ysz))
        if excX != 0:
            img = np.append(img, np.zeros((excX, img.shape[1])).astype(int), axis=0)
        if excY != 0:
            img = np.append(img, np.zeros((img.shape[0], excY)).astype(int), axis=1)

    nrPixels = xsz * ysz
    claheimg = np.zeros(img.shape)

    if clipLimit > 0:
        clipLimit = max(1, clipLimit * xsz * ysz / nrBins)
    else:
        clipLimit = 50

    # Making LUT
    print("...Make the LUT...")
    minVal = 0  # np.min(img)
    maxVal = 255  # np.max(img)

    binSz = np.floor(1 + (maxVal - minVal) / float(nrBins))
    LUT = np.floor((np.arange(minVal, maxVal + 1) - minVal) / float(binSz))

    # Creating bins from LUT with image
    bins = LUT[img]
    print(bins.shape)

    # Making Histogram
    print("...Making the Histogram...")
    nrX = int(nrX)
    nrY = int(nrY)
    xsz = int(xsz)
    ysz = int(ysz)
    hist = np.zeros((nrX, nrY, nrBins))
    print(nrX, nrY, hist.shape)
    for i in range(nrX):
        for j in range(nrY):
            bin_ = bins[
                i * xsz : (i + 1) * xsz, j * ysz : (j + 1) * ysz  # noqa
            ].astype(int)
            for i1 in range(xsz):
                for j1 in range(ysz):
                    hist[i, j, bin_[i1, j1]] += 1

    # Clipping Histogram
    print("...Clipping the Histogram...")
    if clipLimit > 0:
        for i in range(nrX):
            for j in range(nrY):
                nrExcess = 0
                for nr in range(nrBins):
                    excess = hist[i, j, nr] - clipLimit
                    if excess > 0:
                        nrExcess += excess

                binIncr = nrExcess / nrBins
                upper = clipLimit - binIncr
                for nr in range(nrBins):
                    if hist[i, j, nr] > clipLimit:
                        hist[i, j, nr] = clipLimit
                    else:
                        if hist[i, j, nr] > upper:
                            nrExcess += upper - hist[i, j, nr]
                            hist[i, j, nr] = clipLimit
                        else:
                            nrExcess -= binIncr
                            hist[i, j, nr] += binIncr

                if nrExcess > 0:
                    stepSz = max(1, np.floor(1 + nrExcess / nrBins))
                    for nr in range(nrBins):
                        nrExcess -= stepSz
                        hist[i, j, nr] += stepSz
                        if nrExcess < 1:
                            break

    # Mapping Histogram
    print("...Mapping the Histogram...")
    map_ = np.zeros((nrX, nrY, nrBins))
    # print(map_.shape)
    scale = (maxVal - minVal) / float(nrPixels)
    for i in range(nrX):
        for j in range(nrY):
            sum_ = 0
            for nr in range(nrBins):
                sum_ += hist[i, j, nr]
                map_[i, j, nr] = np.floor(min(minVal + sum_ * scale, maxVal))

    # Interpolation
    print("...Interpolation...")
    xI = 0
    for i in range(nrX + 1):
        if i == 0:
            subX = int(xsz / 2)
            xU = 0
            xB = 0
        elif i == nrX:
            subX = int(xsz / 2)
            xU = nrX - 1
            xB = nrX - 1
        else:
            subX = xsz
            xU = i - 1
            xB = i

        yI = 0
        for j in range(nrY + 1):
            if j == 0:
                subY = int(ysz / 2)
                yL = 0
                yR = 0
            elif j == nrY:
                subY = int(ysz / 2)
                yL = nrY - 1
                yR = nrY - 1
            else:
                subY = ysz
                yL = j - 1
                yR = j
            UL = map_[xU, yL, :]
            UR = map_[xU, yR, :]
            BL = map_[xB, yL, :]
            BR = map_[xB, yR, :]

            subBin = bins[xI : xI + subX, yI : yI + subY]  # noqa

            subImage = interpolate(subBin, UL, UR, BL, BR, subX, subY)
            claheimg[xI : xI + subX, yI : yI + subY] = subImage  # noqa
            yI += subY
        xI += subX

    if excX == 0 and excY != 0:
        return claheimg[:, :-excY]
    elif excX != 0 and excY == 0:
        return claheimg[:-excX, :]
    elif excX != 0 and excY != 0:
        return claheimg[:-excX, :-excY]
    else:
        return claheimg


def display_hist_alt(img):
    color = ("r", "g", "b")
    plt.figure(figsize=(8, 6))
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])

    plt.xticks(ticks=range(0, 257, 16), labels=range(0, 257, 16))
    plt.yscale("log")
    plt.show()
