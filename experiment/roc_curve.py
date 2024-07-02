#ai generated... just to see a random implementation of roc curve


import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def process_frame(frame, ground_truth, thresholds):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tpr_list = []
    fpr_list = []

    for threshold in thresholds:
        _, thresholded_frame = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)

        # Flatten the arrays
        thresholded_flat = thresholded_frame.flatten()
        ground_truth_flat = ground_truth.flatten()

        # Calculate TPR and FPR
        fpr, tpr, _ = roc_curve(ground_truth_flat, thresholded_flat)

        tpr_list.append(tpr[1])  # True Positive Rate for current threshold
        fpr_list.append(fpr[1])  # False Positive Rate for current threshold

    return tpr_list, fpr_list

def plot_roc_curve(tpr_list, fpr_list, thresholds):
    plt.figure()
    for i, threshold in enumerate(thresholds):
        plt.plot(fpr_list[i], tpr_list[i], label=f'Threshold: {threshold}')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.show()

def main(video_path, ground_truth_path, thresholds):
    frames = extract_frames(video_path)
    ground_truth_frames = extract_frames(ground_truth_path)

    assert len(frames) == len(ground_truth_frames), "Number of frames and ground truth frames must be equal"

    all_tpr = []
    all_fpr = []

    for i in range(len(frames)):
        frame = frames[i]
        ground_truth = cv2.cvtColor(ground_truth_frames[i], cv2.COLOR_BGR2GRAY)
        tpr_list, fpr_list = process_frame(frame, ground_truth, thresholds)
        all_tpr.append(tpr_list)
        all_fpr.append(fpr_list)

    avg_tpr = np.mean(all_tpr, axis=0)
    avg_fpr = np.mean(all_fpr, axis=0)

    plot_roc_curve([avg_tpr], [avg_fpr], thresholds)

if __name__ == "__main__":
    video_path = 'input_video.mp4'         # Path to the input video
    ground_truth_path = 'ground_truth_video.mp4' # Path to the ground truth video
    thresholds = np.linspace(0, 255, num=10)   # List of threshold values to evaluate

    main(video_path, ground_truth_path, thresholds)
