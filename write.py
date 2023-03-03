import matplotlib.pyplot as plt
import cv2
import numpy as np
import algos as alg


def alpha_trim(img, name):
    sp, gauss, full = combo_get_no_noise(name)
    meds = alg.alpha_trim_mean(img)
    meds_sp = alg.alpha_trim_mean(sp)
    meds_gauss = alg.alpha_trim_mean(gauss)
    meds_full = alg.alpha_trim_mean(full)
    for i in range(len(meds)):
        cv2.imwrite("enhance/atrim{}_{}.png".format(i, name), meds[i])
        cv2.imwrite(
            "enhance/atrim{}_gauss{}-{}_{}.png".format(i, 1, 50, name),
            meds_gauss[i]
        )
        cv2.imwrite(
            "enhance/atrim{}_s&p{}_{}.png".format(i, 0.05, name), meds_sp[i]
        )
        cv2.imwrite(
            "enhance/atrim{}_gauss_s&p{}-{}-{}_{}.png".format(i, 1, 50, 0.05, name),
            meds_full[i]
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
            "enhance/wmed{}_gauss{}-{}_{}.png".format(i, 1, 50, name),
            meds_gauss[i]
        )
        cv2.imwrite(
            "enhance/wmed{}_s&p{}_{}.png".format(i, 0.05, name), meds_sp[i]
        )
        cv2.imwrite(
            "enhance/wmed{}_gauss_s&p{}-{}-{}_{}.png".format(i, 1, 50, 0.05, name),
            meds_full[i]
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
        cv2.imwrite(
            "enhance/{}{}_s&p{}_{}.png".format(fn, k, 0.05, name), f(sp, k)
        )
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
        for (k, kern) in kernels:
            cv2.imwrite("enhance/{}{}_{}.png".format(fn, k, name), f(img, kern))
            cv2.imwrite(
                "enhance/{}{}_gauss{}-{}_{}.png".format(fn, k, 1, 50, name),
                f(gauss, kern),
            )
            cv2.imwrite(
                "enhance/{}{}_s&p{}_{}.png".format(fn, k, 0.05, name), f(sp, kern)
            )
            cv2.imwrite(
                "enhance/{}{}_gauss_s&p{}-{}-{}_{}.png".format(fn, k, 1, 50, 0.05, name),
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
            cv2.medianBlur(gauss, k),
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


# def geom_kern(img, name, kerns, m, v, p):
#     sp, gauss, full = combo_get(name)
#     for k in kerns:
#         cv2.imwrite(
#             "enhance/filtg-{}_{}.png".format(k, name), alg.geometric_filter(img, k)
#         )
#         cv2.imwrite(
#             "enhance/filtg-{}_{}_sp.png".format(k, name), alg.geometric_filter(sp, k)
#         )
#         cv2.imwrite(
#             "enhance/filtg-{}_{}_gauss.png".format(k, name),
#             alg.geometric_filter(gauss, k),
#         )
#         cv2.imwrite(
#             "enhance/filtg-{}_{}_full.png".format(k, name),
#             alg.geometric_filter(full, k),
#         )
#         alg.save_hist(
#             alg.geometric_filter(img, k),
#             "filtg-{}_{}.png".format(k, name),
#         )
#         alg.save_hist(
#             alg.geometric_filter(gauss, k),
#             "filtg-{}_gauss{}-{}_{}.png".format(k, m, v, name),
#         )
#         alg.save_hist(
#             alg.geometric_filter(sp, k), "filtg-{}_s&p{}_{}.png".format(k, p, name)
#         )
#         alg.save_hist(
#             alg.geometric_filter(full, k),
#             "filtg-{}_full{}-{}-{}_{}.png".format(k, m, v, p, name),
#         )


# def weight_kern(img, name, kerns, m, v, p):
#     sp, gauss, full = combo_get(name)
#     kerns = zip([alg.wf_3x3, alg.wf_7x7, alg.wf_9x9], kerns)
#     for f, k in kerns:
#         cv2.imwrite("enhance/wfilt-{}_{}.png".format(k, name), f(img))
#         cv2.imwrite("enhance/wfilt-{}_{}_sp.png".format(k, name), f(sp))
#         cv2.imwrite("enhance/wfilt-{}_{}_gauss.png".format(k, name), f(gauss))
#         cv2.imwrite("enhance/wfilt-{}_{}_full.png".format(k, name), f(full))
#         alg.save_hist(
#             f(img),
#             "wfilt-{}_{}.png".format(k, name),
#         )
#         alg.save_hist(
#             f(gauss),
#             "wfilt-{}_gauss{}-{}_{}.png".format(k, m, v, name),
#         )
#         alg.save_hist(f(sp), "wfilt-{}_s&p{}_{}.png".format(k, p, name))
#         alg.save_hist(
#             f(full),
#             "wfilt-{}_full{}-{}-{}_{}.png".format(k, m, v, p, name),
#         )


# # weight_kern(gray_ivy2, "ivy2", [3, 7, 9], 1, 50, 0.05)
# # geom_kern(gray_ivy2, "ivy2", [3, 7, 9, 15], 1, 50, 0.05)
