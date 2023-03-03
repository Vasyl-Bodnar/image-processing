import matplotlib.pyplot as plt
import cv2
import numpy as np

import algos as alg
import write as wrt

ivy = cv2.imread("orig/poison-ivy.jpg")
ivy2 = cv2.imread("orig/poison-ivy2.jpg")
oak_atln = cv2.imread("orig/atlantic-poison-oak.jpg")
oak_east = cv2.imread("orig/eastern-poison-oak.jpg")
gray_ivy = cv2.cvtColor(ivy, cv2.COLOR_BGR2GRAY)
gray_ivy2 = cv2.cvtColor(ivy2, cv2.COLOR_BGR2GRAY)
gray_oak_atln = cv2.cvtColor(oak_atln, cv2.COLOR_BGR2GRAY)
gray_oak_east = cv2.cvtColor(oak_east, cv2.COLOR_BGR2GRAY)


def multilevel_threshold(img, threshold_values):
    img[img < threshold_values[0]] = 0
    img[(img < threshold_values[1]) & (img > 0)] = 125
    img[img > 125] = 255
    return img


def global_thresh(img, T):
    return (cv2.threshold(img, T, 255, cv2.THRESH_BINARY))[1]


def local_threshold(gray, T, window_size=11):
    height, width = gray.shape
    half_size = window_size // 2
    padding_size = half_size + 1
    gray = cv2.copyMakeBorder(
        gray, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_REFLECT
    )
    output = np.zeros((height, width), dtype=np.uint8)
    for i in range(padding_size, height + padding_size):
        for j in range(padding_size, width + padding_size):
            window = gray[
                i - half_size:i + half_size + 1, j - half_size:j + half_size + 1
            ]
            _, threshold = cv2.threshold(
                window, T, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            output[i - padding_size, j - padding_size] = threshold[half_size, half_size]
    return output


def niblack(gray, win_size=30, k=-0.3):
    mean = cv2.blur(gray, (win_size, win_size))
    stddev = np.sqrt(cv2.blur(gray * gray, (win_size, win_size)) - mean * mean)
    threshold = mean + k * stddev
    bin_img = np.zeros_like(gray)
    bin_img[gray >= threshold] = 255
    return bin_img


plt.imshow(global_thresh(gray_oak_atln, 160), cmap="gray")
plt.show()

# dst_med(gray_ivy, "ivy")
# # wrt.median_filter(gray_ivy2, "ivy2")
# dst_med(gray_ivy2, "ivy2")
# # wrt.median_filter(gray_oak_east, "oak_east")
# dst_med(gray_oak_east, "oak_east")

# wrt.alpha_trim(gray_ivy2, "ivy2")
# wrt.weighted_median(gray_ivy2, "ivy2")
# wrt.alpha_trim(gray_oak_east, "oak_east")
# wrt.weighted_median(gray_oak_east, "oak_east")

# def sharpen(img, name, blur, bname, alpha=0.2):
#     lines = cv2.subtract(img, blur)
#     res = cv2.add(img, alpha * lines, dtype=8)
#     cv2.imwrite("enhance/lines_{}-0.3_{}.png".format(bname, name), lines)
#     cv2.imwrite("enhance/sharp_{}-0.3_{}.png".format(bname, name), res)

# for i in ["ivy2", "ivy", "oak_east"]:
#     img = cv2.imread("enhance/lines_med15-0.3_{}.png".format(i), cv2.IMREAD_GRAYSCALE)
#     alg.save_hist(
#         img,
#         "lines_med15-0.3_{}.png".format(i),
#     )

# img = cv2.imread("enhance/sharp_med15-0.6_ivy2.png", cv2.IMREAD_GRAYSCALE)
# alg.save_hist(
#     img,
#     "sharp_med15-0.6_ivy2.png",
# )


# def sharpen(img, name, blur, bname, alpha=0.2):
#     lines = cv2.subtract(img, blur)
#     res = cv2.add(img, alpha * lines, dtype=8)
#     cv2.imwrite("enhance/lines_{}-0.2_{}.png".format(bname, name), lines)
#     cv2.imwrite("enhance/sharp_{}-0.2_{}.png".format(bname, name), res)


# sharpen(
#     gray_ivy2,
#     "ivy2",
#     cv2.imread("enhance/filt-15_ivy2.png", cv2.IMREAD_GRAYSCALE),
#     "mean15"
# )

# img = cv2.imread("enhance/lines_med15-0.6_ivy2.png", cv2.IMREAD_GRAYSCALE)
# alg.save_hist(img, "lines_med15-0.6_ivy2.png")
# sharpen(gray_ivy2, "ivy2", cv2.imread("enhance/med15_ivy2.png", cv2.IMREAD_GRAYSCALE), "med15", 0.3)
# sharpen(gray_ivy2, "ivy2", cv2.imread("enhance/med15_ivy2.png", cv2.IMREAD_GRAYSCALE), "med15", 0.6)
# sharpen(gray_ivy2, "ivy2", cv2.imread("enhance/med7_ivy2.png", cv2.IMREAD_GRAYSCALE), "med7")
# sharpen(gray_oak_east, "oak_east", cv2.imread("enhance/med15_oak_east.png", cv2.IMREAD_GRAYSCALE), "med15")
# sharpen(gray_oak_east, "oak_east", cv2.imread("enhance/med7_oak_east.png", cv2.IMREAD_GRAYSCALE), "med7")

# wrt.f_filter(gray_ivy, "ivy", alg.weighted_median, "pmid")
# cv2.imshow("s", alg.midpoint_filter(wrt.combo_get_no_noise("ivy")[0], 3))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# def tmp_gauss(m, v, p):
#     _, gauss, _ = wrt.combo_get_no_noise("ivy", m, v, p)
#     print(alg.dist(alg.make_hist(gray_ivy), alg.make_hist(gauss)))


# def tmp_sp(m, v, p):
#     sp, _, _ = wrt.combo_get_no_noise("ivy", m, v, p)
#     print(alg.dist(alg.make_hist(gray_ivy), alg.make_hist(sp)))


# tmp_sp(1, 50, 0.05)
# tmp_sp(10, 5, 0.01)
# tmp_sp(20, 40, 0.2)

# plt.imshow(alg.geometric_filter(gray_ivy2, 9), cmap="gray")
# plt.imshow(gray_ivy, cmap="gray")
# plt.show()

# wrt.noise_estimation(
#     gray_ivy,
#     "ivy",
#     1,
#     50,
#     0.05,
#     [(100, 100, 200, 200), (1250, 750, 300, 300), (2200, 400, 200, 200), None],
#     3
# )

# wrt.noise_estimation(
#     gray_ivy2,
#     "ivy2",
#     1,
#     50,
#     0.05,
#     [(0, 0, 30, 30), (0, 0, 100, 100), (150, 150, 200, 200), None],
#     1
# )

# write_hist_noise(gray_ivy2, "ivy2", 1, 50, 0.05)
# write_hist_noise(gray_ivy2, "ivy2", 10, 5, 0.01)
# write_hist_noise(gray_ivy2, "ivy2", 20, 40, 0.2)
# wrt.write_hist_noise(gray_ivy, "ivy", 1, 50, 0.05)
# wrt.write_hist_noise(gray_ivy, "ivy", 10, 5, 0.01)
# wrt.write_hist_noise(gray_ivy, "ivy", 20, 40, 0.2)
# write_hist_noise(gray_oak_atln, "oak_atln", 1, 50, 0.05)
# write_hist_noise(gray_oak_atln, "oak_atln", 10, 5, 0.01)
# write_hist_noise(gray_oak_atln, "oak_atln", 20, 40, 0.2)
# write_hist_noise(gray_oak_east, "oak_east", 1, 50, 0.05)
# write_hist_noise(gray_oak_east, "oak_east", 10, 5, 0.01)
# write_hist_noise(gray_oak_east, "oak_east", 20, 40, 0.2)

# pl_histt.imshow(alg.gaussian_noise(alg.s_and_p_noise(gray_ivy2)), cmap='gray')
# plt.show()
# hsv_ivy = cv2.cvtColor(ivy, cv2.COLOR_BGR2HSV)
# gray_hsv_ivy = cv2.cvtColor(hsv_ivy, cv2.COLOR_HSV2BGR)
# hsv_ivy2 = cv2.cvtColor(ivy2, cv2.COLOR_BGR2HSV)
# gray_hsv_ivy2 = cv2.cvtColor(hsv_ivy2, cv2.COLOR_HSV2BGR)
# hsv_oak_atln = cv2.cvtColor(oak_atln, cv2.COLOR_BGR2HSV)
# gray_hsv_oak_atln = cv2.cvtColor(hsv_oak_atln, cv2.COLOR_HSV2BGR)
# hsv_oak_east = cv2.cvtColor(oak_east, cv2.COLOR_BGR2HSV)
# gray_hsv_oak_east = cv2.cvtColor(hsv_oak_east, cv2.COLOR_HSV2BGR)

# draw_hist(gray_hsv_ivy)
# draw_hist(gray_hsv_ivy2)
# draw_hist(gray_hsv_oak_atln)
# draw_hist(gray_hsv_oak_east)

# print(alg.dist(alg.make_hist(gray_ivy), alg.make_hist(gray_hsv_ivy)))
# print(alg.dist(alg.make_hist(gray_ivy2), alg.make_hist(gray_hsv_ivy2)))
# print(alg.dist(alg.make_hist(gray_oak_atln), alg.make_hist(gray_hsv_oak_atln)))
# print(alg.dist(alg.make_hist(gray_oak_east), alg.make_hist(gray_hsv_oak_east)))

# rgb_hsv_ivy = cv2.cvtColor(hsv_ivy, cv2.COLOR_HSV2BGR)
# # Compute the histograms
# rgb_hist = cv2.calcHist(
#     [ivy], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256]
# )
# # hsv_hist = cv2.calcHist(
# #     [hsv_ivy], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]
# # )
# # rgb_hsv_hist = cv2.calcHist(
# #     [rgb_hsv_ivy], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
# # )
# cv2.imwrite(
#     "modf/hsv_oak_east.png",
#     cv2.cvtColor(cv2.cvtColor(oak_east, cv2.COLOR_BGR2HSV), cv2.COLOR_HSV2BGR),
# )
# cv2.imwrite(
#     "modf/hsv_oak_atln.png",
#     cv2.cvtColor(cv2.cvtColor(oak_atln, cv2.COLOR_BGR2HSV), cv2.COLOR_HSV2BGR),
# )
# cv2.imwrite(
#     "modf/hsv_ivy2.png",
#     cv2.cvtColor(cv2.cvtColor(ivy2, cv2.COLOR_BGR2HSV), cv2.COLOR_HSV2BGR),
# )
# cv2.imwrite(
#     "modf/hsv_ivy.png",
#     cv2.cvtColor(cv2.cvtColor(ivy, cv2.COLOR_BGR2HSV), cv2.COLOR_HSV2BGR),
# )

# print(alg.dist(alg.make_hist(gray_ivy), alg.make_hist(alg.hist_equal(gray_ivy))))
# print(dist(make_hist(gray_ivy), make_hist(gray_ivy2)))
# print(dist(make_hist(gray_ivy2), make_hist(hist_equal(gray_ivy2))))
# print(dist(make_hist(gray_ivy), make_hist(gray_oak_atln)))
# print(dist(make_hist(gray_oak_atln), make_hist(gray_oak_east)))
# print(dist(make_hist(gray_oak_east), make_hist(gray_oak_atln)))

# cv2.imwrite("enhance/ssr_ivy.png", cv2.cvtColor(SSR(ivy, 200), cv2.COLOR_BGR2GRAY))
# cv2.imwrite("enhance/ssr_ivy2.png", SSR(ivy2, 200))
# cv2.imwrite(
#     "enhance/ssr_oak_atln.png", cv2.cvtColor(SSR(oak_atln, 200), cv2.COLOR_BGR2GRAY)
# )
# cv2.imwrite(
#     "enhance/ssr_oak_east.png", cv2.cvtColor(SSR(oak_east, 200), cv2.COLOR_BGR2GRAY)
# )

# cv2.imwrite(
#     "enhance/msr_ivy.png", cv2.cvtColor(MSR(ivy, [40, 80, 140]), cv2.COLOR_BGR2GRAY)
# )
# cv2.imwrite("enhance/msr_ivy2.png", MSR(ivy2, [40, 80, 140]))
# cv2.imwrite(
#     "enhance/msr_oak_atln.png",
#     cv2.cvtColor(MSR(oak_atln, [40, 80, 140]), cv2.COLOR_BGR2GRAY),
# )
# cv2.imwrite(
#     "enhance/msr_oak_east.png",
#     cv2.cvtColor(MSR(oak_east, [40, 80, 140]), cv2.COLOR_BGR2GRAY),
# )

# clah8 = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
# cv2.imwrite("enhance/ch_ivy.png", clah8.apply(gray_ivy))
# cv2.imwrite("enhance/ch_ivy2.png", clah8.apply(gray_ivy2))
# cv2.imwrite("enhance/ch_oak_atln.png", clah8.apply(gray_oak_atln))
# cv2.imwrite("enhance/ch_oak_east.png", clah8.apply(gray_oak_east))

# cv2.imwrite("enhance/he_ivy.png", hist_equal(gray_ivy))
# cv2.imwrite("enhance/he_ivy2.png", hist_equal(gray_ivy2))
# cv2.imwrite("enhance/he_oak_atln.png", hist_equal(gray_oak_atln))
# cv2.imwrite("enhance/he_oak_east.png", hist_equal(gray_oak_east))

# cv2.imwrite("enhance/in_ivy.png", intensity_stretch(gray_ivy, 70, 0, 140, 255))
# cv2.imwrite("enhance/in_ivy2.png", intensity_stretch(gray_ivy2, 70, 0, 140, 255))
# cv2.imwrite("enhance/in_atln_oak.png", intensity_stretch(gray_oak_atln, 70, 0, 140, 255))
# cv2.imwrite("enhance/in_east_oak.png", intensity_stretch(gray_oak_east, 70, 0, 140, 255))

# cv2.imwrite("enhance/gamma0.6_ivy.png", gamma_correct(gray_ivy, 0.6))
# cv2.imwrite("enhance/gamma1.5_ivy.png", gamma_correct(gray_ivy, 1.5))
# cv2.imwrite("enhance/gamma0.6_ivy2.png", gamma_correct(gray_ivy2, 0.6))
# cv2.imwrite("enhance/gamma1.5_ivy2.png", gamma_correct(gray_ivy2, 1.5))
# cv2.imwrite("enhance/gamma0.6_oak_atln.png", gamma_correct(gray_oak_atln, 0.6))
# cv2.imwrite("enhance/gamma1.5_oak_atln.png", gamma_correct(gray_oak_atln, 1.5))
# cv2.imwrite("enhance/gamma0.6_oak_east.png", gamma_correct(gray_oak_east, 0.6))
# cv2.imwrite("enhance/gamma1.5_oak_east.png", gamma_correct(gray_oak_east, 1.5))

# cv2.imwrite("arithm/ivy+ivy.png", cv2.add(ivy, ivy))  # or using np (ivy/255 + ivy/255) * 255
# ...
# cv2.imwrite("arithm/ivy(div)ivy.png", cv2.divide(ivy, ivy))
# cv2.imwrite("arithm/ivy(div)oak.png", cv2.divide(small_ivy, small_oak))
# cv2.imwrite("arithm/ivy(div)2.png", cv2.divide(gray_ivy, 2))

# fig.add_subplot(1, 2, 1)
# # plt.imshow(gray_ivy2, cmap="gray")
# plt.hist(gray_ivy2, bins=50)
# # plt.imshow(intense, cmap="gray")
# plt.hist(intense, bins=50)
# axis[0].hist(hist_equal(gray_ivy))
# axis[1].hist(hist_equal(gray_ivy2))
# axis[2].hist(hist_equal(gray_oak_atln))
# axis[3].hist(hist_equal(gray_oak_east))
# plt.show()
# fig, axis = plt.subplots(2, 1)
# axis[0].imshow(cv2.cvtColor(MSR(ivy, [60, 100, 180]), cv2.COLOR_BGR2GRAY), cmap="gray")
# axis[1].imshow(cv2.cvtColor(MSR(ivy, [40, 80, 140]), cv2.COLOR_BGR2GRAY), cmap="gray")
# axis[0].hist(hist_equal(gray_ivy2))
# axis[1].hist(gray_ivy2)
# plt.show()
