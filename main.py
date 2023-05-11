"""
This is an old file that defines some images, functions, and operations, used to be old.py
Folders throughout the project may be hardcoded, I also have a special structure for my two databases which are used throughout
These databases are:
Leaves and Their Health Data (Database A) - https://www.kaggle.com/datasets/piantic/plantpathology-apple-dataset
Toxic and Non-Toxic Plants (Database B) - https://www.kaggle.com/datasets/hanselliott/toxic-plant-classification
"""
import matplotlib.pyplot as plt
import cv2
import numpy as np
from multiprocess import pool

from util import identity
import algos as alg
import write as wrt

ivy = cv2.imread("orig/poison-ivy.jpg")
ivy2 = cv2.imread("orig/poison-ivy2.jpg")
oak_atln = cv2.imread("orig/atlantic-poison-oak.jpg")
oak_east = cv2.imread("orig/eastern-poison-oak.jpg")
sumac = cv2.imread("toxic_images/poison_sumac/361.jpg")
gray_ivy = cv2.cvtColor(ivy, cv2.COLOR_BGR2GRAY)
gray_ivy2 = cv2.cvtColor(ivy2, cv2.COLOR_BGR2GRAY)
gray_oak_atln = cv2.cvtColor(oak_atln, cv2.COLOR_BGR2GRAY)
gray_oak_east = cv2.cvtColor(oak_east, cv2.COLOR_BGR2GRAY)
gray_sumac = cv2.cvtColor(sumac, cv2.COLOR_BGR2GRAY)
apple = cv2.imread("orig/apple.png")
apple = cv2.cvtColor(apple, cv2.COLOR_BGR2GRAY)

example_images = [gray_ivy, gray_ivy2, gray_oak_atln, gray_oak_east, gray_sumac, apple]
test_image_numbers = [
    (str(x) if x >= 100 else ("0" + str(x) if x >= 10 else "00" + str(x))) + ".jpg"
    for x in [y for y in range(30)]
]
test_images_paths = [
    [p + img for img in test_image_numbers]
    for p in [
        "toxic_images/poison_ivy/",
        "toxic_images/poison_ivy_west/",
        "toxic_images/poison_oak_east/",
        "toxic_images/poison_oak_west/",
        "toxic_images/poison_sumac/",
        "nontoxic_images/bear_oak/",
        "nontoxic_images/boxelder/",
        "nontoxic_images/fragrant_sumac/",
        "nontoxic_images/virginia_creeper/",
        "nontoxic_images/jack_in_the_pulpit/",
    ]
]


def model(img, test_imgs, enhance, segment, histogram, compare):
    """
    img: (IName, Img)
    test_imgs: [(TName, Img)]
    enhance: Img -> Img
    segment: Img -> Img
    histogram: Img -> Hist
    compare: Hist, Hist -> Float
    return [(IName, TName, Float)]
    """
    full_apply = lambda img: histogram(enhance(img) * segment(img))
    processed_img = full_apply(img)
    fin = []
    for test in test_imgs:
        fin.append(compare(processed_img, full_apply(test)))
    return fin


def blur_sharp(gray):
    # Apply a Gaussian filter to create a blurred version of the image
    blurred = cv2.GaussianBlur(gray, (3, 3), 1)

    # Subtract the blurred version of the image from the original image to create a mask
    mask = cv2.subtract(gray, blurred)

    # Multiply the mask by a factor (0.5 in this example) and add it back to the original image
    sharpened = cv2.addWeighted(gray, 1.5, mask, 0.5, 0)
    return blurred, sharpened


def combine_enh_seg(img):
    return alg.enhance(img) * alg.segment(img)


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


def thread_party(f=alg.enhance, g=alg.segment):
    collect = []
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
                if (not c[0].__contains__("poison"))
                and (not c[1][0][1].__contains__("poison"))
                else 0
                for c in dflatten(collect)
            ]
        ),
        sum(
            [
                1
                if c[0].__contains__("poison")
                and (not c[1][0][1].__contains__("poison"))
                else 0
                for c in dflatten(collect)
            ]
        ),
        sum(
            [
                1
                if (not c[0].__contains__("poison"))
                and c[1][0][1].__contains__("poison")
                else 0
                for c in dflatten(collect)
            ]
        ),
    )


# if __name__ == '__main__':
# collect = thread_party(lambda img: equalizeHist(enhance(img)), lambda img: alg.global_thresh(img, 130))
# print(count_successes_fails(collect))
# print([x/300 for x in count_successes_fails(collect)])