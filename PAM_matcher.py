import os
import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
import cv2
import numpy as np

from feature_model import model_NN
model_NN_weights = "/home/avinoam/workspace/YAM_HAMELACH/results/17_05/weights/cp.weights.h5"
from tqdm import tqdm
np.set_printoptions(2)

def save_match_figure(fn_1, fn_2, path):
    plt.subplot(1, 2, 1)
    plt.imshow(plt.imread(fn_1))
    plt.title(fn_1.parts[-1])
    plt.subplot(1,2,2)
    plt.imshow(plt.imread(fn_2))
    plt.title(fn_2.parts[-1])

    plt.savefig(path)
def to_patch_fn(base_dir, PAM, box_id):
    basepath = Path(base_dir, PAM)
    fn = Path(basepath, f"{PAM}_{box_id}").with_suffix(".jpg")
    return fn

class PatchMatcher(object):
    def __init__(self, args):
        self._patches_path = args.patches


MIN_GOOD_FEATURES_MATCH = 6
class TwoImagesMatchFeatures():
    '''
    A class that can calculate the match score of two images
    by trying to apply homography between them.
    '''
    def __init__(self, im1_fn, im2_fn):
        self._im1_fn = im1_fn
        self._im2_fn = im2_fn
        self._model_NN = model_NN
        self._model_NN.load_weights(model_NN_weights)

    def calc_mathces(self):
        self._img1 = cv2.imread(str(self._im1_fn), cv2.IMREAD_GRAYSCALE) # queryImage
        self._img2 = cv2.imread(str(self._im2_fn), cv2.IMREAD_GRAYSCALE) # trainImage

        # print(self._img1.shape[:2], self._img2.shape[:2])
        # Initiate SIFT detector
        sift = cv2.SIFT_create()

        # find the keypoints and descriptors with SIFT
        self._kp1, des1 = sift.detectAndCompute(self._img1,None)
        self._kp2, des2 = sift.detectAndCompute(self._img2,None)

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        self._matches = bf.knnMatch(des1,des2,k=2)

        # Apply ratio test
        self._good = []
        for m,n in self._matches:
            if m.distance < 0.75*n.distance:
                self._good.append(m)

        # cv.drawMatchesKnn expects list of lists as matches.
    def find_homography(self):
        if len(self._good) < MIN_GOOD_FEATURES_MATCH: # There is not reason to look for homography..
            return False
        if (self._im1_fn.parts[-2]) == (self._im2_fn.parts[-2]):
            return False
        keepPercent = 30
        keep_min = 15
        matches = sorted(self._good, key=lambda x:x.distance)
        # keep only the top matches
        keep = max((keep_min, int(len(matches) * keepPercent)))
        matches = matches[:keep]

        # allocate memory for the keypoints (x, y)-coordinates from the
        # top matches -- we'll use these coordinates to compute our
        # homography matrix
        self._ptsA = np.zeros((len(matches), 2), dtype="float")
        self._ptsB = np.zeros((len(matches), 2), dtype="float")
        for (i, m) in enumerate(matches):
		# indicate that the two keypoints in the respective images
		# map to each other
            self._ptsA[i] = self._kp1[m.queryIdx].pt
            self._ptsB[i] = self._kp2[m.trainIdx].pt

        self._H, mask = cv2.findHomography(self._ptsB, self._ptsA, method=cv2.RANSAC)

    def plot_homography(self):
        (h, w) = self._img1.shape[:2]
        aligned = cv2.warpPerspective(self._img2, self._H, (w, h))

        # new_im = cv2.hconcat(new_im, self._img2)
        plt.subplot(1,2,1)
        plt.imshow(aligned, cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(self._img1, cmap='gray')

        plt.show()
        return aligned

    def calc_features(self):
        self._errors = np.zeros(len(self._good))
        for (i, m) in enumerate(self._good):
            # indicate that the two keypoints in the respective images
            # map to each other
            ptsA = np.array(self._kp1[m.queryIdx].pt)
            ptsB = np.array(self._kp2[m.trainIdx].pt)
            pt_b_homogenous = np.concatenate((ptsB, np.array([1])))
            projected_p = self._H.dot(pt_b_homogenous)
            projected_p_homogenous = projected_p / projected_p[2]
            projected_p = projected_p_homogenous[:2]
            error = np.linalg.norm(projected_p - ptsA)
            self._errors[i] = error
        self._bins = np.power(2, np.array([0,  2, 4, 6, 8, 12], dtype=np.float32))
        self._bins = np.concatenate((np.array([0]), self._bins))
        self._errors_hist = np.histogram(self._errors, bins=self._bins)[0]

        self._errors_hist = self._errors_hist/self._errors_hist.sum()
    def match(self):
        if len(self._good) < MIN_GOOD_FEATURES_MATCH:
            return False
        if (self._im1_fn.parts[-2]) == (self._im2_fn.parts[-2]):
            return False

        self.calc_features()

        features = np.array([self.to_features()[0]])
        predict = self._model_NN(features).numpy()
        return [False, True][np.argmax(predict)]
        # implement here any features based model
        # return self._errors_hist[:7].sum()>0.3

    def to_features(self):
        features = []
        features_names = []

        features.append(len(self._good))

        features.extend(list(self._errors_hist))
        features_names.extend( [f"error_{self._bins[i]}-{self._bins[i+1]}" for i in range(len(self._bins)-1)])

        return features, features_names

    def plot(self):
        good = [[g] for g in self._good]
        img3 = cv2.drawMatchesKnn(self._img1,self._kp1,self._img2,self._kp2, good,None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        plt.imshow(img3),plt.show()
class FeatureBasedMatcher(PatchMatcher):
    def __init__(self, fn_1, fn_2):
        self._feature_matcher = TwoImagesMatchFeatures(fn_1, fn_2)

    def match(self):
        try:
            self._feature_matcher.calc_mathces()
            self._feature_matcher.find_homography()
            match  = self._feature_matcher.match()
            return match
        except:
            return False # if no match is found, so it probably it is a bad match..

    def features(self):
        features, features_names = self._feature_matcher.to_features()
        return features, features_names

class CSVMatcher(PatchMatcher):
    '''
    This is a 'fake' matcher, used for ground truth parsing
    '''
    def __init__(self, args):
        PatchMatcher.__init__(self, args)
        self._csv = args.csv_fn
        self._raw_csv = pd.read_csv(self._csv)
        self._valid_csv = self._raw_csv[self._raw_csv.Frg.str.isdigit() == True]
        self._valid_csv = self._valid_csv[self._valid_csv.Box.str.isdigit() == True]
        self._valid_csv = self._add_patch_fp(self._valid_csv)
        self._valid_csv[["Scroll", "Frg"]].value_counts() # [lambda x: x>20].index.tolist()
        self._grouped = self._valid_csv.groupby(["Scroll", "Frg"])

        self._has_copy = []
        for i,df in self._grouped:
            n = df.shape[0]
            if n<2:
                continue
            self._has_copy.extend(df.patch_file_path)
        pass

    def _add_patch_fp(self, df):
        ''' add file path to data-frame'''
        def row_to_fp(row):
           return to_patch_fn(self._patches_path, row.File, row.Box)

        patch_file_path = df.apply(row_to_fp, axis=1)
        df['patch_file_path'] = patch_file_path
        return df


    def show_matches(self):
        for i,df in self._grouped:
            n = df.shape[0]
            if n<2:
                continue
            i = 1
            print (df)
            imgs_fn = []
            for index, row in df.iterrows():
                # print(row)
                fn = row.patch_file_path
                imgs_fn.append(fn)
                try:
                    im = plt.imread(fn)
                    plt.subplot(1,n,i)
                    plt.title(f"{row.File}, {row.Box}")
                    plt.imshow(im)
                    i += 1
                except FileNotFoundError as e:
                    print (e)
                    continue
            plt.show()



        pass

    def is_match(self, fn1, fn2):
        if fn1 == fn2:
            return True
        if fn1 not in self._has_copy:
            return False
        if fn2 not in self._has_copy:
            return False
        for i,df in self._grouped:
            n = df.shape[0]
            if n==1:
                continue
            file_nams = df.patch_file_path
            contain1 = (file_nams == fn1).sum()
            if not contain1:
                continue
            contain2 = (file_nams == fn2).sum()
            if contain2:
                return True

            return False
        pass

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--patches")
    parser.add_argument("--csv_fn", default=False)
    parser.add_argument("--features_out", default=False)

    args = parser.parse_args()

    if args.csv_fn:
        csv_matcher = CSVMatcher(args)
        # csv_matcher.show_matches()

    patch_path = []
    for PAM in ['M43485-1-E', 'M43003-1-E', 'M43510-1-E',
                'M43500-1-E', 'M43505-1-E', 'M42809-1-E',
                'M43473-1-E', 'M42970-1-E', 'M43992-1-E',
                'M43195-1-E', 'M43448-1-E']:
        PAM_paths = list(Path(args.patches,PAM).glob("*.jpg"))
        patch_path.extend(PAM_paths)

    header = False
    # features_out = open(args.features_out, 'w')
    # missed = open(args.features_out + "miss", 'w')
    p_bar = tqdm(patch_path)
    for i,fn_1 in enumerate(patch_path):
        p_bar.update(1)
        p_bar.refresh()
        for (j,fn_2) in enumerate(patch_path[i+1:]):
            csv_match = csv_matcher.is_match(fn_1, fn_2)
            # if not csv_match:
            #     if np.random.random()>0.005:
            #         continue

            matcher = FeatureBasedMatcher(fn_1, fn_2)
            match = matcher.match()
            # print(match)
            if csv_match and not match:
                print(f"f{fn_1.name},{fn_2.name} missed ")
                # missed.write(f"f{fn_1.name},{fn_2.name},{os.linesep}")
                path = Path("/home/avinoam/workspace/YAM_HAMELACH/results/17_05/match_images/miss",
                            f"{fn_1.parts[-1]}-{fn_2.parts[-1]}").with_suffix(".jpg")
                save_match_figure(fn_1, fn_2, path)
            if match:
                features, features_names = matcher.features()
                if not header:
                    # features_out.write("im1,im2,label,n_good,"+",".join(features_names)+os.linesep)
                    header = True
                features = [f"{f:.3f}" for f in features]
                m = fn_1.name, fn_2.name, ["False_pair","True_pair"][csv_match], *features
                m = ",".join(m)
                print (m)
                # features_out.write(m + os.linesep)
                if csv_match:
                    path = Path("/home/avinoam/workspace/YAM_HAMELACH/results/17_05/match_images/true",
                              f"{fn_1.parts[-1]}-{fn_2.parts[-1]}").with_suffix(".jpg")
                else:
                    path = Path("/home/avinoam/workspace/YAM_HAMELACH/results/17_05/match_images/false",
                              f"{fn_1.parts[-1]}-{fn_2.parts[-1]}").with_suffix(".jpg")
                save_match_figure(fn_1, fn_2, path)

                # plt.show()

                # matcher._feature_matcher.plot()
                # matcher._feature_matcher.plot_homography()


            # feature_mathch = TwoImagesMatchFeatures
            # feature_mathch.calc_mathces()
            # feature_mathch.find_homography()
            # error_hist = feature_mathch.calc_homography_errors()
            # print(error_hist[:5], match)
    # features_out.close()
    # missed.close()
