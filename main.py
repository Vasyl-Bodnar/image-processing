"""
TODO: Clean up "##OLD"
"""
import matplotlib.pyplot as plt
import cv2
from cv2 import equalizeHist
import numpy as np
from multiprocess import pool

from util import identity
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
apple = cv2.imread("orig/apple.png")
apple = cv2.cvtColor(apple, cv2.COLOR_BGR2GRAY)

test_image_numbers = [
    (str(x) if x >= 100 else ("0" + str(x) if x >= 10 else "00" + str(x))) + ".jpg"
    for x in [y for y in range(30)]
]


def magic(gray):
    # Apply a Gaussian filter to create a blurred version of the image
    blurred = cv2.GaussianBlur(gray, (3, 3), 1)

    # Subtract the blurred version of the image from the original image to create a mask
    mask = cv2.subtract(gray, blurred)

    # Multiply the mask by a factor (0.5 in this example) and add it back to the original image
    sharpened = cv2.addWeighted(gray, 1.5, mask, 0.5, 0)
    return blurred, sharpened


def enhance(img):
    return alg.lut_stretch(img)


def segment(img):
    return alg.otsu(img)


def combine_enh_seg(img):
    return enhance(img) * segment(img)

def combine_g(img, f, g):
    return f(img) * g(img)

def gen_hist(img, feats=False):
    hist = cv2.calcHist([img], [0], None, [255], [0, 256], accumulate=False)
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)  # type: ignore
    if feats:
        hist = np.append(hist, [np.median(hist), np.mean(hist), np.amax(hist), np.amin(hist)])  # type: ignore
    return hist


def chi_compare(img1, img2, feats=False):
    return cv2.compareHist(gen_hist(img1, feats), gen_hist(img2, feats), 1)


def loop(iimg, feats=False, f=combine_enh_seg):
    successes = []
    for lst, name in [
        (test_image_numbers, p)
        for p in [
            "poison_ivy",
            "poison_ivy_west",
            "poison_oak_west",
            "poison_oak_east",
            "poison_sumac",
            "bear_oak",
            "boxelder",
            "fragrant_sumac",
            "jack_in_the_pulpit",
            "virginia_creeper",
        ]
    ]:
        if name.__contains__("poison"):  # type: ignore
            for x in lst:
                img = cv2.cvtColor(
                    cv2.imread(f"toxic_images/{name}/{x}"), cv2.COLOR_BGR2GRAY
                )
                compare = (chi_compare(f(img), f(iimg), feats), name, x)
                if compare[0] > 0.0:
                    successes.append(compare)
        else:
            for x in lst:
                img = cv2.cvtColor(
                    cv2.imread(f"nontoxic_images/{name}/{x}"), cv2.COLOR_BGR2GRAY
                )
                compare = (chi_compare(f(img), f(iimg), feats), name, x)
                if compare[0] > 0.0:
                    successes.append(compare)
    successes.sort()
    return successes


collect = []


def loop_over(ilstname, f=combine_enh_seg):
    lst = []
    if ilstname[1].__contains__("poison"):
        for x in ilstname[0]:
            img = cv2.cvtColor(
                cv2.imread(f"toxic_images/{ilstname[1]}/{x}"), cv2.COLOR_BGR2GRAY
            )
            s = loop(img, False, f)
            # print(f"{name}, {x}: ", s[:1])
            lst.append((f"{ilstname[1]}", s[:1]))
    else:
        for x in ilstname[0]:
            img = cv2.cvtColor(
                cv2.imread(f"nontoxic_images/{ilstname[1]}/{x}"), cv2.COLOR_BGR2GRAY
            )
            s = loop(img, False, f)  # f"{ilstname[2]}, {x}")
            # print(f"{name}, {x}: ", s[:1])
            lst.append((f"{ilstname[1]}", s[:1]))
    return lst


def thread_party(f=enhance, g=segment):
    with pool.Pool(12) as p:
        collect.append(
            p.map(
                lambda x: loop_over(x, lambda img: combine_g(img, f, g)),
                [
                    (test_image_numbers, p)
                    for p in [
                        "poison_ivy",
                        "poison_ivy_west",
                        "poison_oak_east",
                        "poison_oak_west",
                        "poison_sumac",
                        "bear_oak",
                        "boxelder",
                        "fragrant_sumac",
                        "jack_in_the_pulpit",
                        "virginia_creeper",
                    ]
                ],
            )
        )
        return collect


def dflatten(l):
    return [
        item
        for sublist in [item for sublist in l for item in sublist]
        for item in sublist
    ]


def count_successes_fails(collect):
    return (
        sum([1 if c[0] == c[1][0][1] else 0 for c in dflatten(collect)]),
        sum(
            [
                1
                if c[0].__contains__("poison") and c[1][0][1].__contains__("poison")
                else 0
                for c in dflatten(collect)
            ]
        ),
        sum(
            [
                1
                if (not c[0].__contains__("poison")) and (not c[1][0][1].__contains__("poison"))
                else 0
                for c in dflatten(collect)
            ]
        ),
        sum(
            [
                1
                if c[0].__contains__("poison") and (not c[1][0][1].__contains__("poison"))
                else 0
                for c in dflatten(collect)
            ]
        ),
        sum(
            [
                1
                if (not c[0].__contains__("poison")) and c[1][0][1].__contains__("poison")
                else 0
                for c in dflatten(collect)
            ]
        ),
    )
    # return collect

if __name__ == '__main__':
    collect = thread_party(lambda img: equalizeHist(enhance(img)), lambda img: alg.global_thresh(img, 130))
    print(count_successes_fails(collect))
print([x/300 for x in count_successes_fails(collect)])

# if __name__ == '__main__':
# collect = thread_party()
# print(count_sucesses(collect))

# [print("TRUE: ", c) if c[0] == c[1][0][1] else print("FALSE:", c) for c in collect]
# print("avg", np.mean(np.array([s[0] for s in successes])))
# print("median", np.median(np.array([s[0] for s in successes])))

# cv2.imwrite("ivy-f.png", magic(ivy2)[0])
# cv2.imwrite("ivy-s.png", magic(ivy2)[1])
# f = combine_enh_seg
# wrt.multi_save(
# "midterm",
# [
# ("apple", apple),
# ("ivy2", gray_ivy2),
# ("oak_east", gray_oak_east),
# ("oak_atln", gray_oak_atln),
# ("ivy", gray_ivy),
# ],
# [
## ("enhance", enhance),
## ("segment", segment),
## ("combine", combine_enh_seg),
# (
# "compare-oak-atln_",
# lambda img: print(chi_compare(gray_ivy2, f(img), True)),
# )
# ],
# )
# for (name, img) in [("ivy2", gray_ivy2), ("oak_east", gray_oak_east), ("oak_atln", gray_oak_atln), ("ivy", gray_ivy)]:
# alg.save_hist(combine_enh_seg(img), f"{name}/midterm")
# wrt.test(
# gray_ivy,
# [
# enhance,
# segment,
# combine_enh_seg,
# ],
# )

## OLD
# cv2.imwrite("morph/ivy/sc-k7-cross.png", closing(gray_ivy, 7))
# cv2.imwrite("morph/ivy/soc-k7-cross.png", opening(closing(gray_ivy, 7), 7))
# cv2.imwrite("morph/ivy/sooc-k7-cross.png", opening(opening(closing(gray_ivy, 7), 7), 7))
# cv2.imwrite("morph/ivy/scooc-k7-cross.png", closing(opening(opening(closing(gray_ivy, 7), 7), 7), 7))
# cv2.imwrite("morph/ivy/sccooc-k7-cross.png", closing(closing(opening(opening(closing(gray_ivy, 7), 7), 7), 7), 7))
# cv2.imwrite("morph/ivy/soccooc-k7-cross.png", opening(closing(closing(opening(opening(closing(gray_ivy, 7), 7), 7), 7), 7), 7))
# cv2.imwrite("morph/ivy/scoccooc-k7-cross.png", closing(opening(closing(closing(opening(opening(closing(gray_ivy, 7), 7), 7), 7), 7), 7), 7))

# for e in ["cross", "square", "xshape", "circ"]:
# wrt.multi_save(
# "morph",
# [("ivy2", gray_ivy2), ("oak_atln", gray_oak_atln), ("ivy", gray_ivy)],
# [
# ("dilation", dilate),
# ("erosion", erode),
# ("opening", opening),
# ("closing", closing),
# ("gradient", morph_grad),
# ("top_hat", top_hat),
# ("bot_hat", lambda img, *args: (bot_hat(img, *args)) * 255),
# ],
# [(3, "k3"), (e, e)],
##("global-128", lambda img: alg.global_thresh(img, 128)),
# )
# wrt.multi_save(
# "morph",
# [("ivy2", gray_ivy2), ("oak_atln", gray_oak_atln)],
# [
# ("dilation", dilate),
# ("erosion", erode),
# ("opening", opening),
# ("closing", closing),
# ("gradient", morph_grad),
# ("top_hat", top_hat),
# ("bot_hat", lambda img, *args: (bot_hat(img, *args)) * 255),
# ],
# [(7, "k7"), (e, e)],
##("global-128", lambda img: alg.global_thresh(img, 128)),
# )

# cv2.imwrite("t128_ivy.png", alg.global_thresh(gray_ivy, 128))
# wrt.test(
# gray_ivy2,
# [identity, erode, dilate, opening, closing, morph_grad, bot_hat, top_hat],
# 2,
# 3,
# "cross",
# )
# cv2.imwrite("enhance/gaussfilt3-s2_oak_atln.png", cv2.GaussianBlur(gray_oak_atln, (3,3), sigmaX=2))
# alg.save_hist(cv2.GaussianBlur(gray_oak_atln, (3,3), sigmaX=2), "gaussfilt3-s2_oak_atln.png")

# write_hist_hog(gray_oak_atln, "oak_atln", [2.0, 3.0, 5.0], [5, 15, 2])
# write_combo_gauss_sobel(gray_oak_atln, "oak_atln", 3, 2, 20)
# write_combo_gauss_sobel(gray_oak_atln, "oak_atln", 3, 2, 40)
# write_combo_gauss_sobel(gray_oak_atln, "oak_atln", 3, 2, 128)
# write_combo_gauss_sobel(gray_oak_atln, "oak_atln", 3, 2, 60)
# write_combo_gauss_sobel(gray_oak_atln, "oak_atln", 3, 2, 80)
# write_combo_gauss_sobel(gray_oak_atln, "oak_atln", 3, 2, 100)
# write_hist_robinson(gray_oak_atln, "oak_atln", 30)
# write_hist_robinson(gray_oak_atln, "oak_atln", 50)
# write_hist_robinson(gray_oak_atln, "oak_atln", 100)
# write_hist_robinson(gray_oak_atln, "oak_atln", 150)

# write_hist_hog(gray_ivy, "ivy", [2.0, 3.0, 5.0], [5, 15, 2])
# write_combo_gauss_sobel(gray_ivy, "ivy", 3, 2, 20)
# write_combo_gauss_sobel(gray_ivy, "ivy", 3, 2, 40)
# write_combo_gauss_sobel(gray_ivy, "ivy", 3, 2, 128)
# write_combo_gauss_sobel(gray_ivy, "ivy", 3, 2, 60)
# write_combo_gauss_sobel(gray_ivy, "ivy", 3, 2, 80)
# write_combo_gauss_sobel(gray_ivy, "ivy", 3, 2, 100)
# write_hist_robinson(gray_ivy, "ivy", 30)
# write_hist_robinson(gray_ivy, "ivy", 50)
# write_hist_robinson(gray_ivy, "ivy", 100)
# write_hist_robinson(gray_ivy, "ivy", 150)

# write_hist_hog(gray_ivy2, "ivy2", [2.0, 3.0, 5.0], [5, 15, 2])
# write_combo_gauss_sobel(gray_ivy2, "ivy2", 3, 2, 20)
# write_combo_gauss_sobel(gray_ivy2, "ivy2", 3, 2, 40)
# write_combo_gauss_sobel(gray_ivy2, "ivy2", 3, 2, 128)
# write_combo_gauss_sobel(gray_ivy2, "ivy2", 3, 2, 60)
# write_combo_gauss_sobel(gray_ivy2, "ivy2", 3, 2, 80)
# write_combo_gauss_sobel(gray_ivy2, "ivy2", 3, 2, 100)
# write_hist_robinson(gray_ivy2, "ivy2", 30)
# write_hist_robinson(gray_ivy2, "ivy2", 50)
# write_hist_robinson(gray_ivy2, "ivy2", 100)
# write_hist_robinson(gray_ivy2, "ivy2", 150)

# write_hist_canny(oak_atln, "oak_atln", 20, 120)
# write_hist_canny(oak_atln, "oak_atln", 20, 200)
# write_hist_canny(oak_atln, "oak_atln", 30, 60)
# write_hist_canny(oak_atln, "oak_atln", 50)
# write_hist_canny(oak_atln, "oak_atln", 100, 200)
# write_hist_canny(oak_atln, "oak_atln", 180, 220)
# write_hist_canny(oak_atln, "oak_atln", 80, 250)

# write_hist_hog(gray_oak_atln, "oak_atln", [1.5, 2.0, 5.0], [5,15,2])
# write_laplacian_et_gauss(gray_ivy2, "ivy2", [1, 2, 3, 4], [15])
# write_laplacian_et_gauss(gray_ivy, "ivy", [1, 2, 3, 4], [15])
# show_laplacian(gray_ivy2, 1)
# show_gaussian_edge(gray_ivy2, 1)
# plt.show()
# write_hist_hog(gray_ivy, "ivy", [1.5, 2.0, 5.0], [5,15,2])
# write_double_hog(gray_ivy, "ivy", [(1.5, 120), (2, 120)])
# write_hist_robinson(gray_ivy, "ivy", 30)
# write_hist_robinson(gray_ivy, "ivy", 50)
# write_hist_robinson(gray_ivy, "ivy", 100)
# write_hist_robinson(gray_ivy, "ivy", 150)

# write_hist_canny(gray_ivy, "ivy", 20, 120)
# write_hist_canny(gray_ivy, "ivy", 20, 200)
# write_hist_canny(gray_ivy, "ivy", 30, 60)
# write_hist_canny(gray_ivy, "ivy", 50)
# write_hist_canny(gray_ivy, "ivy", 100, 200)
# write_hist_canny(gray_ivy, "ivy", 180, 220)
# write_hist_canny(gray_ivy, "ivy", 80, 250)

# show_laplacian(apple)
# show_gaussian_edge(apple, 3, 2, 0)
# write_hist_gauss_sobel(gray_ivy, "ivy", 3, 2, 20)
# write_hist_gauss_sobel(gray_ivy, "ivy", 3, 2, 40)
# write_hist_gauss_sobel(gray_ivy, "ivy", 3, 2, 128)
# write_hist_gauss_sobel(gray_ivy, "ivy", 3, 2, 60)
# write_hist_gauss_sobel(gray_ivy, "ivy", 3, 2, 80)
# write_hist_gauss_sobel(gray_ivy, "ivy", 3, 2, 100)
# cv2.imwrite("enhance/gaussfilt3-s2_ivy.png", cv2.GaussianBlur(gray_ivy, (3,3), sigmaX=2))
# alg.save_hist(cv2.GaussianBlur(gray_ivy, (3,3), sigmaX=2), "gaussfilt3-s2_ivy.png")

# write_global_t(gray_oak_atln, "oak_atln", [15, 45, 75, 125, 200])
# multi_multi_t(gray_ivy, "ivy", [(5, 120), (30, 70), (45, 135), (5, 35)])
# multi_multi_t(gray_ivy2, "ivy2", [(30, 100), (30, 163), (63, 135), (150, 235)])
# write_adaptive_t(gray_oak_atln, "oak_atln", [3, 5, 9, 15, 35, 65])
# write_niblack_t(gray_oak_atln, "oak_atln", [15, 30, 60, 100], [-0.2, -0.3, -0.6])
# write_niblack_t(gray_ivy, "ivy", [15, 30, 60, 100, 300], [-0.2, -0.3, -0.6])
# write_bernsen_t(gray_oak_atln, "oak_atln", [15, 30, 60, 100, 300], [15, 20, 40])
# write_sauvola_t(gray_oak_atln, "oak_atln", [15, 30, 60, 100, 300], [0.2, 0.34, 0.5])
# plt.imshow(sauvola(gray_ivy), cmap="gray")
# plt.show()

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
# def best_t(f):
# T = 0.5*(np.min(f) + np.max(f))

## Initialize done to False
# done = False

## Loop until done is True
# while not done:
## Binarize the image based on the current threshold
# g = f >= T

## Calculate the mean intensity values of the foreground and background regions
# foreground_mean = np.mean(f[g])
# background_mean = np.mean(f[~g])

## Calculate the next threshold value
# Tnext = 0.5*(foreground_mean + background_mean)

## Check if the difference between T and Tnext is greater than 0.5
# if abs(T - Tnext) > 0.5:
## Update T and continue the loop
# T = Tnext
# else:
## Exit the loop
# done = True
# return T
# def local_threshold(gray, T, window_size=11):
# height, width = gray.shape
# half_size = window_size // 2
# padding_size = half_size + 1
# gray = cv2.copyMakeBorder(
# gray, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_REFLECT
# )
# output = np.zeros((height, width), dtype=np.uint8)
# for i in range(padding_size, height + padding_size):
# for j in range(padding_size, width + padding_size):
# window = gray[
# i - half_size:i + half_size + 1, j - half_size:j + half_size + 1
# ]
# _, threshold = cv2.threshold(
# window, T, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
# )
# output[i - padding_size, j - padding_size] = threshold[half_size, half_size]
# return output
