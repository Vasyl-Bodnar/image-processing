"""
Defines many functions and abstractions to help show or save more images more easily
Most are conjured rather horribly
"""
import matplotlib.pyplot as plt
import cv2
import numpy as np
import algos as alg
from util import identity


def test(img, fs, l=1, *args):
    multi_show([f(img, *args) for f in fs], l)
    plt.show()


def multi_show(imgs, l=1, n=None):
    n = len(imgs) if n is None else n
    _, axis = plt.subplots(l, n // l)
    if n > 1:
        i = 0
        for row in axis:
            if l > 1:
                for col in row:
                    col.imshow(imgs[i], cmap="gray")
                    i += 1
            else:
                row.imshow(imgs[i], cmap="gray")  # type:ignore
                i += 1
    else:
        axis.imshow(imgs[0], cmap="gray")  # type:ignore


# Rather complex base function for saving things
def multi_save(loc, name_imgs, name_fs, name_args=[], preprocess=("", identity)):
    arg = []
    fdata = ""
    preprocess = (".", preprocess[1]) if preprocess[0] == "" else preprocess
    if len(name_args) != 0:
        for k, n in name_args:
            arg.append(k)
            fdata += "-" + n
    for iname, i in name_imgs:
        for fname, f in name_fs:
            if fname.endswith("_"):
                print(fname[:-1] + f" for {iname}:")
                f(preprocess[1](i), *arg)
            else:
                cv2.imwrite(
                    f"{loc}/{iname}/{preprocess[0]}/{fname}{fdata}.png",
                    f(preprocess[1](i), *arg),
                )


def show_sob(img, T):
    multi_show(alg.sobel(img, T), 4)


def show_rob(img, T):
    multi_show(alg.robinson(img, T), 2)


def show_laplacian(img, s=2, T=30):
    multi_show(alg.laplacian(img, s, T), 3)


def show_gaussian_edge(img, s=2, T=50):
    multi_show(alg.gaussian_edge(img, s, T), 3)


def write_hist_sobel(img, img_name, T):
    x, y, mag, ang, fin = alg.sobel(img, T)
    cv2.imwrite("edge/sobel-x_{}.png".format(img_name), x)
    cv2.imwrite("edge/sobel-y_{}.png".format(img_name), y)
    cv2.imwrite("edge/sobel-mag_{}.png".format(img_name), mag)
    cv2.imwrite("edge/sobel-ang_{}.png".format(img_name), ang)
    cv2.imwrite("edge/sobel-adapt_{}.png".format(img_name), alg.adaptive_thresh(mag))
    cv2.imwrite("edge/sobel-otsu_{}.png".format(img_name), alg.otsu(mag))  # type:ignore
    cv2.imwrite("edge/sobel{}_{}.png".format(T, img_name), fin)
    alg.save_hist(x, "sobel-x_{}.png".format(img_name))
    alg.save_hist(y, "sobel-y_{}.png".format(img_name))
    alg.save_hist(mag, "sobel-mag_{}.png".format(img_name))
    alg.save_hist(ang, "sobel-ang_{}.png".format(img_name))


def write_hist_gauss_sobel(img, img_name, k, s, T):
    img = cv2.GaussianBlur(img, (k, k), sigmaX=s)
    x, y, mag, ang, fin = alg.sobel(img, T)
    cv2.imwrite("edge/gsobel-k{}-s{}-x_{}.png".format(k, s, img_name), x)
    cv2.imwrite("edge/gsobel-k{}-s{}-y_{}.png".format(k, s, img_name), y)
    cv2.imwrite("edge/gsobel-k{}-s{}-mag_{}.png".format(k, s, img_name), mag)
    cv2.imwrite("edge/gsobel-k{}-s{}-ang_{}.png".format(k, s, img_name), ang)
    cv2.imwrite(
        "edge/gsobel-k{}-s{}-adapt_{}.png".format(k, s, img_name),
        alg.adaptive_thresh(mag),
    )
    cv2.imwrite(
        "edge/gsobel-k{}-s{}-otsu_{}.png".format(k, s, img_name), alg.otsu(mag)
    )  # type:ignore
    cv2.imwrite("edge/gsobel-k{}-s{}{}_{}.png".format(k, s, T, img_name), fin)
    alg.save_hist(x, "gsobel-k{}-s{}-x_{}.png".format(k, s, img_name))
    alg.save_hist(y, "gsobel-k{}-s{}-y_{}.png".format(k, s, img_name))
    alg.save_hist(mag, "gsobel-k{}-s{}-mag_{}.png".format(k, s, img_name))
    alg.save_hist(ang, "gsobel-k{}-s{}-ang_{}.png".format(k, s, img_name))


def write_combo_gauss_sobel(img, img_name, k, s, T):
    x, y, mag, ang, fin = alg.sobel(img, T)
    cv2.imwrite("edge/sobel-x_{}.png".format(img_name), x)
    cv2.imwrite("edge/sobel-y_{}.png".format(img_name), y)
    cv2.imwrite("edge/sobel-mag_{}.png".format(img_name), mag)
    cv2.imwrite("edge/sobel-ang_{}.png".format(img_name), ang)
    cv2.imwrite(
        "edge/sobel-adapt_{}.png".format(img_name),
        alg.adaptive_thresh(mag.astype(np.uint8)),
    )
    cv2.imwrite(
        "edge/sobel-otsu_{}.png".format(img_name), alg.otsu(mag.astype(np.uint8))
    )  # type:ignore
    cv2.imwrite("edge/sobel{}_{}.png".format(T, img_name), fin)
    alg.save_hist(x, "sobel-x_{}.png".format(img_name))
    alg.save_hist(y, "sobel-y_{}.png".format(img_name))
    alg.save_hist(mag, "sobel-mag_{}.png".format(img_name))
    alg.save_hist(ang, "sobel-ang_{}.png".format(img_name))
    img = cv2.GaussianBlur(img, (k, k), sigmaX=s)
    x, y, mag, ang, fin = alg.sobel(img, T)
    cv2.imwrite("edge/gsobel-k{}-s{}-x_{}.png".format(k, s, img_name), x)
    cv2.imwrite("edge/gsobel-k{}-s{}-y_{}.png".format(k, s, img_name), y)
    cv2.imwrite("edge/gsobel-k{}-s{}-mag_{}.png".format(k, s, img_name), mag)
    cv2.imwrite("edge/gsobel-k{}-s{}-ang_{}.png".format(k, s, img_name), ang)
    cv2.imwrite(
        "edge/gsobel-k{}-s{}-adapt_{}.png".format(k, s, img_name),
        alg.adaptive_thresh(mag.astype(np.uint8)),
    )
    cv2.imwrite(
        "edge/gsobel-k{}-s{}-otsu_{}.png".format(k, s, img_name),
        alg.otsu(mag.astype(np.uint8)),
    )  # type:ignore
    cv2.imwrite("edge/gsobel-k{}-s{}{}_{}.png".format(k, s, T, img_name), fin)
    alg.save_hist(x, "gsobel-k{}-s{}-x_{}.png".format(k, s, img_name))
    alg.save_hist(y, "gsobel-k{}-s{}-y_{}.png".format(k, s, img_name))
    alg.save_hist(mag, "gsobel-k{}-s{}-mag_{}.png".format(k, s, img_name))
    alg.save_hist(ang, "gsobel-k{}-s{}-ang_{}.png".format(k, s, img_name))


def write_hist_canny(img, img_name, T1, T2=None):
    fin = alg.canny(img, T1, T2)
    cv2.imwrite("edge/canny{}-{}_{}.png".format(T1, T2, img_name), fin)


def write_hist_robinson(img, img_name, T):
    mag, ang, fin = alg.robinson(img, T)
    cv2.imwrite("edge/robinson-mag_{}.png".format(img_name), mag)
    cv2.imwrite("edge/robinson-ang_{}.png".format(img_name), ang)
    cv2.imwrite("edge/robinson{}_{}.png".format(T, img_name), fin)
    cv2.imwrite(
        "edge/robinson-adapt_{}.png".format(img_name),
        alg.adaptive_thresh(mag.astype(np.uint8)),
    )
    cv2.imwrite(
        "edge/robinson-otsu_{}.png".format(img_name), alg.otsu(mag.astype(np.uint8))
    )  # type:ignore
    alg.save_hist(mag, "robinson-mag_{}.png".format(img_name))
    alg.save_hist(ang, "robinson-ang_{}.png".format(img_name))


def write_hist_hog(img, img_name, gs, Ts):
    fin = alg.hog(img)
    cv2.imwrite("edge/hog_{}.png".format(img_name), fin)  # type: ignore
    alg.save_hist(fin, "hog_{}.png".format(img_name))
    for g in gs:
        cv2.imwrite("edge/hog{}_{}.png".format(g, img_name), fin**g)  # type: ignore
        alg.save_hist(fin**g, "hog{}_{}.png".format(g, img_name))  # type: ignore
    for T in Ts:
        cv2.imwrite("edge/hog{}_{}.png".format(T, img_name), alg.global_thresh(fin, T))  # type: ignore
    cv2.imwrite("edge/hog-adapt_{}.png".format(img_name), alg.adaptive_thresh(fin.astype(np.uint8)))  # type: ignore
    cv2.imwrite("edge/hog-otsu_{}.png".format(img_name), alg.otsu(fin.astype(np.uint8)))  # type: ignore


def write_double_hog(img, name, gsEtTs):
    fin = alg.hog(img)
    for g, T in gsEtTs:
        cv2.imwrite("edge/hog-g{}T{}_{}.png".format(g, T, name), fin)  # type: ignore


def write_laplacian_et_gauss(img, img_name, ss, Ts):
    for s in ss:
        for T in Ts:
            lapl = alg.laplacian(img, s, T)
            gaus = alg.gaussian_edge(img, s, T)
            # cv2.imwrite("edge/laplacian-gauss-s{}_{}.png".format(s, img_name), lapl[0]) # type: ignore
            # cv2.imwrite("edge/laplacian-lapl-s{}_{}.png".format(s, img_name), lapl[1]) # type: ignore
            # cv2.imwrite("edge/laplacian-T{}-s{}_{}.png".format(T, s, img_name), lapl[2]) # type: ignore
            # cv2.imwrite("edge/gauss-gauss-edge-s{}_{}.png".format(s, img_name), gaus[0]) # type: ignore
            # cv2.imwrite("edge/gauss-first-edge-s{}_{}.png".format(s, img_name), gaus[1]) # type: ignore
            # cv2.imwrite("edge/gauss-edge-T{}-s{}_{}.png".format(T, s, img_name), gaus[2]) # type: ignore
            cv2.imwrite("edge/laplacian-lapl-s{}_{}.png".format(s, img_name), lapl[1] * 255)  # type: ignore
            cv2.imwrite("edge/gauss-first-edge-s{}_{}.png".format(s, img_name), gaus[1] * 255)  # type: ignore


def write_global_t(img, name, vs):
    for v in vs:
        cv2.imwrite(
            "segment/global-v{}-{}.png".format(v, name), alg.global_thresh(img, v)
        )


def write_adaptive_t(img, name, ss):
    for s in ss:
        cv2.imwrite(
            "segment/adaptive-s{}-{}.png".format(s, name), alg.adaptive_thresh(img, s)
        )


def write_niblack_t(img, name, win_sizes, ks):
    for w in win_sizes:
        for k in ks:
            cv2.imwrite(
                "segment/niblack-w{}-k{}-{}.png".format(w, k, name),
                alg.niblack(img, w, k),
            )


def write_bernsen_t(img, name, win_sizes, cs):
    for w in win_sizes:
        for c in cs:
            cv2.imwrite(
                "segment/bernsen-w{}-c{}-{}.png".format(w, c, name),
                alg.bernsen(img, w, c),
            )


def write_sauvola_t(img, name, win_sizes, ks):
    for w in win_sizes:
        for k in ks:
            cv2.imwrite(
                "segment/sauvola-w{}-k{}-{}.png".format(w, k, name),
                alg.sauvola(img, w, k),
            )


def multi_multi_t(img, name, vps):
    for v, u in vps:
        cv2.imwrite(
            "segment/multi-v({},{})-{}.png".format(v, u, name),
            alg.multilevel_thresh(img.copy(), (v, u)),
        )


def alpha_trim(img, name):
    sp, gauss, full = combo_get_no_noise(name)
    meds = alg.alpha_trim_mean(img)
    meds_sp = alg.alpha_trim_mean(sp)
    meds_gauss = alg.alpha_trim_mean(gauss)
    meds_full = alg.alpha_trim_mean(full)
    for i in range(len(meds)):
        cv2.imwrite("enhance/atrim{}_{}.png".format(i, name), meds[i])
        cv2.imwrite(
            "enhance/atrim{}_gauss{}-{}_{}.png".format(i, 1, 50, name), meds_gauss[i]
        )
        cv2.imwrite("enhance/atrim{}_s&p{}_{}.png".format(i, 0.05, name), meds_sp[i])
        cv2.imwrite(
            "enhance/atrim{}_gauss_s&p{}-{}-{}_{}.png".format(i, 1, 50, 0.05, name),
            meds_full[i],
        )
        alg.save_hist(
            meds[i],
            "atrim{}_{}.png".format(i, name),
        )
        alg.save_hist(
            meds_gauss[i],
            "atrim{}_gauss{}-{}_{}.png".format(i, 1, 50, name),
        )
        alg.save_hist(meds_sp[i], "atrim{}_s&p{}_{}.png".format(i, 0.05, name))
        alg.save_hist(
            meds_full[i],
            "atrim{}_full{}-{}-{}_{}.png".format(i, 1, 50, 0.05, name),
        )


def weighted_median(img, name):
    sp, gauss, full = combo_get_no_noise(name)
    meds = alg.weighted_median(img)
    meds_sp = alg.weighted_median(sp)
    meds_gauss = alg.weighted_median(gauss)
    meds_full = alg.weighted_median(full)
    for i in range(len(meds)):
        cv2.imwrite("enhance/wmed{}_{}.png".format(i, name), meds[i])
        cv2.imwrite(
            "enhance/wmed{}_gauss{}-{}_{}.png".format(i, 1, 50, name), meds_gauss[i]
        )
        cv2.imwrite("enhance/wmed{}_s&p{}_{}.png".format(i, 0.05, name), meds_sp[i])
        cv2.imwrite(
            "enhance/wmed{}_gauss_s&p{}-{}-{}_{}.png".format(i, 1, 50, 0.05, name),
            meds_full[i],
        )
        alg.save_hist(
            meds[i],
            "wmed{}_{}.png".format(i, name),
        )
        alg.save_hist(
            meds_gauss[i],
            "wmed{}_gauss{}-{}_{}.png".format(i, 1, 50, name),
        )
        alg.save_hist(meds_sp[i], "wmed{}_s&p{}_{}.png".format(i, 0.05, name))
        alg.save_hist(
            meds_full[i],
            "wmed{}_full{}-{}-{}_{}.png".format(i, 1, 50, 0.05, name),
        )


def f_filter(img, name, f, fn, kernels=[3, 7, 15]):
    sp, gauss, full = combo_get_no_noise(name)
    for k in kernels:
        cv2.imwrite("enhance/{}{}_{}.png".format(fn, k, name), f(img, k))
        cv2.imwrite(
            "enhance/{}{}_gauss{}-{}_{}.png".format(fn, k, 1, 50, name),
            f(gauss, k),
        )
        cv2.imwrite("enhance/{}{}_s&p{}_{}.png".format(fn, k, 0.05, name), f(sp, k))
        cv2.imwrite(
            "enhance/{}{}_gauss_s&p{}-{}-{}_{}.png".format(fn, k, 1, 50, 0.05, name),
            f(full, k),
        )
        alg.save_hist(
            f(img, k),
            "{}{}_{}.png".format(fn, k, name),
        )
        alg.save_hist(
            f(gauss, k),
            "{}{}_gauss{}-{}_{}.png".format(fn, k, 1, 50, name),
        )
        alg.save_hist(f(sp, k), "{}{}_s&p{}_{}.png".format(fn, k, 0.05, name))
        alg.save_hist(
            f(full, k),
            "{}{}_full{}-{}-{}_{}.png".format(fn, k, 1, 50, 0.05, name),
        )


def max_min_filter(img, name, kernels=[3, 7, 15]):
    sp, gauss, full = combo_get_no_noise(name)
    kernels = [(k, np.ones((k, k), np.uint8)) for k in kernels]
    for f, fn in [(cv2.erode, "min"), (cv2.dilate, "max")]:
        for k, kern in kernels:
            cv2.imwrite("enhance/{}{}_{}.png".format(fn, k, name), f(img, kern))
            cv2.imwrite(
                "enhance/{}{}_gauss{}-{}_{}.png".format(fn, k, 1, 50, name),
                f(gauss, kern),
            )
            cv2.imwrite(
                "enhance/{}{}_s&p{}_{}.png".format(fn, k, 0.05, name), f(sp, kern)
            )
            cv2.imwrite(
                "enhance/{}{}_gauss_s&p{}-{}-{}_{}.png".format(
                    fn, k, 1, 50, 0.05, name
                ),
                f(full, kern),
            )
            alg.save_hist(
                f(img, kern),
                "{}{}_{}.png".format(fn, k, name),
            )
            alg.save_hist(
                f(gauss, kern),
                "{}{}_gauss{}-{}_{}.png".format(fn, k, 1, 50, name),
            )
            alg.save_hist(f(sp, kern), "{}{}_s&p{}_{}.png".format(fn, k, 0.05, name))
            alg.save_hist(
                f(full, kern),
                "{}{}_full{}-{}-{}_{}.png".format(fn, k, 1, 50, 0.05, name),
            )


def median_filter(img, name, kernels=[3, 7, 15]):
    sp, gauss, full = combo_get_no_noise(name)
    for k in kernels:
        cv2.imwrite("enhance/med{}_{}.png".format(k, name), cv2.medianBlur(img, k))
        cv2.imwrite(
            "enhance/med{}_gauss{}-{}_{}.png".format(k, 1, 50, name),
            cv2.medianblur(gauss, k),
        )
        cv2.imwrite(
            "enhance/med{}_s&p{}_{}.png".format(k, 0.05, name), cv2.medianBlur(sp, k)
        )
        cv2.imwrite(
            "enhance/med{}_gauss_s&p{}-{}-{}_{}.png".format(k, 1, 50, 0.05, name),
            cv2.medianBlur(full, k),
        )
        alg.save_hist(
            cv2.medianBlur(img, k),
            "med{}_{}.png".format(k, name),
        )
        alg.save_hist(
            cv2.medianBlur(gauss, k),
            "med{}_gauss{}-{}_{}.png".format(k, 1, 50, name),
        )
        alg.save_hist(cv2.medianBlur(sp, k), "med{}_s&p{}_{}.png".format(k, 0.05, name))
        alg.save_hist(
            cv2.medianBlur(full, k),
            "med{}_full{}-{}-{}_{}.png".format(k, 1, 50, 0.05, name),
        )


def combo_get_no_noise(name, m=1, v=50, p=0.05):
    (sp, _), (gauss, _), full = combo_get(name, m, v, p)
    return sp, gauss, full


def combo_get(name, m=1, v=50, p=0.05):
    return (
        (
            cv2.imread("noisy/s&p{}_{}.png".format(p, name), cv2.IMREAD_GRAYSCALE),
            cv2.imread("noise/s&p{}_{}.png".format(p, name), cv2.IMREAD_GRAYSCALE),
        ),
        (
            cv2.imread(
                "noisy/gauss{}-{}_{}.png".format(m, v, name),
                cv2.IMREAD_GRAYSCALE,
            ),
            cv2.imread(
                "noise/gauss{}-{}_{}.png".format(m, v, name),
                cv2.IMREAD_GRAYSCALE,
            ),
        ),
        cv2.imread(
            "noisy/gauss_s&p{}-{}-{}_{}.png".format(m, v, p, name),
            cv2.IMREAD_GRAYSCALE,
        ),
    )


def noise_estimation(img, name, m, v, p, regions, thickness=2):
    (sp, sp_noise), (gauss, gauss_noise), full = combo_get(name)
    for region in regions:
        print("\n" + str(region))
        print("Real {}".format(alg.estimate_noise(img, region)))
        print(
            "S&P {}: p={}: {}".format(
                name,
                p,
                alg.estimate_noise(
                    sp,
                    region,
                ),
            )
        )
        print(
            "Gauss {}: m={}, v={}, {}".format(
                name,
                m,
                v,
                alg.estimate_noise(
                    gauss,
                    region,
                ),
            )
        )
        print(
            "Full {}: m={}, v={}, p={}, {}".format(
                name,
                m,
                v,
                p,
                alg.estimate_noise(
                    full,
                    region,
                ),
            )
        )
        print(
            "Noise S&P {}: p={}: {}".format(
                name,
                p,
                alg.estimate_noise(
                    sp_noise,
                    region,
                ),
            )
        )
        print(
            "Noise Gauss {}: m={}, v={}, {}".format(
                name,
                m,
                v,
                alg.estimate_noise(
                    gauss_noise,
                    region,
                ),
            )
        )

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    sp = cv2.cvtColor(sp, cv2.COLOR_GRAY2BGR)
    gauss = cv2.cvtColor(gauss, cv2.COLOR_GRAY2BGR)
    sp_noise = cv2.cvtColor(sp_noise, cv2.COLOR_GRAY2BGR)
    gauss_noise = cv2.cvtColor(gauss_noise, cv2.COLOR_GRAY2BGR)
    full = cv2.cvtColor(full, cv2.COLOR_GRAY2BGR)
    for region in regions:
        if region is not None:
            x, y, w, h = region
            # roi = img[y : y + h, x : x + w] # noqa
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness)
            cv2.rectangle(sp, (x, y), (x + w, y + h), (0, 0, 255), thickness)
            cv2.rectangle(gauss, (x, y), (x + w, y + h), (0, 0, 255), thickness)
            cv2.rectangle(sp_noise, (x, y), (x + w, y + h), (0, 0, 255), thickness)
            cv2.rectangle(gauss_noise, (x, y), (x + w, y + h), (0, 0, 255), thickness)
            cv2.rectangle(full, (x, y), (x + w, y + h), (0, 0, 255), thickness)
            cv2.imwrite("modf/regions_{}.png".format(name), img)
            cv2.imwrite("modf/regions_{}_sp.png".format(name), sp)
            cv2.imwrite("modf/regions_{}_gauss.png".format(name), gauss)
            cv2.imwrite("modf/regions_{}_full.png".format(name), full)
            cv2.imwrite("modf/noise-regions_{}_sp.png".format(name), sp_noise)
            cv2.imwrite("modf/noise-regions_{}_gauss.png".format(name), gauss_noise)


def write_hist_noise(img, img_name, m, v, p):
    full, (sp, sp_noise), (gauss, gauss_noise) = alg.combo_gauss_sp_noise(img, m, v, p)
    cv2.imwrite("noisy/gauss{}-{}_{}.png".format(m, v, img_name), gauss)
    cv2.imwrite("noisy/s&p{}_{}.png".format(p, img_name), sp)
    cv2.imwrite("noisy/gauss_s&p{}-{}-{}_{}.png".format(m, v, p, img_name), full)
    cv2.imwrite("noise/gauss{}-{}_{}.png".format(m, v, img_name), gauss_noise)
    cv2.imwrite("noise/s&p{}_{}.png".format(p, img_name), sp_noise)
    alg.save_hist(gauss, "gauss{}-{}_{}.png".format(m, v, img_name))
    alg.save_hist(sp, "s&p{}_{}.png".format(p, img_name))
    alg.save_hist(full, "gauss_s&p{}-{}-{}_{}.png".format(m, v, p, img_name))
    alg.save_hist(gauss_noise, "noise-gauss{}-{}_{}.png".format(m, v, img_name))
    alg.save_hist(sp_noise, "noise-s&p{}_{}.png".format(p, img_name))


def arithm_kern(img, name, kerns, m, v, p):
    sp, gauss, full = combo_get_no_noise(name)
    for k in kerns:
        cv2.imwrite("enhance/filt-{}_{}.png".format(k, name), alg.arithm_filter(img, k))
        cv2.imwrite(
            "enhance/filt-{}_{}_sp.png".format(k, name), alg.arithm_filter(sp, k)
        )
        cv2.imwrite(
            "enhance/filt-{}_{}_gauss.png".format(k, name), alg.arithm_filter(gauss, k)
        )
        cv2.imwrite(
            "enhance/filt-{}_{}_full.png".format(k, name), alg.arithm_filter(full, k)
        )
        alg.save_hist(
            alg.arithm_filter(img, k),
            "filt-{}_{}.png".format(k, name),
        )
        alg.save_hist(
            alg.arithm_filter(gauss, k),
            "filt-{}_gauss{}-{}_{}.png".format(k, m, v, name),
        )
        alg.save_hist(
            alg.arithm_filter(sp, k), "filt-{}_s&p{}_{}.png".format(k, p, name)
        )
        alg.save_hist(
            alg.arithm_filter(full, k),
            "filt-{}_full{}-{}-{}_{}.png".format(k, m, v, p, name),
        )