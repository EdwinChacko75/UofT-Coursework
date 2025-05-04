import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_iou(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.
    Each box is defined as (x, y, w, h).
    """
    xA, yA, wA, hA = boxA
    xB, yB, wB, hB = boxB
    xA2, yA2 = xA + wA, yA + hA
    xB2, yB2 = xB + wB, yB + hB

    xI1 = max(xA, xB)
    yI1 = max(yA, yB)
    xI2 = min(xA2, xB2)
    yI2 = min(yA2, yB2)
    if xI2 < xI1 or yI2 < yI1:
        return 0.0
    interArea = (xI2 - xI1) * (yI2 - yI1)
    boxAArea = wA * hA
    boxBArea = wB * hB
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def process_video_hue(video_path):
    """Part 1: Use hue histogram and mean shift tracking, comparing with Viola–Jones detections."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video")
        return

    ret, frame = cap.read()
    if not ret:
        print("Cannot read video")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        print("No face detected in the first frame.")
        return
    x, y, w, h = faces[0]
    
    track_window = (x, y, w, h)

    hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0, 30, 32)), np.array((180, 255, 255)))
    roi = hsv_roi[y:y+h, x:x+w]
    mask_roi = mask[y:y+h, x:x+w]
    roi_hist = cv2.calcHist([roi], [0], mask_roi, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    iou_list = []
    frame_nums = []
    frame_counter = 1  

    sample_high = None  
    sample_low = None   

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_counter += 1

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        ret_val, track_window = cv2.meanShift(back_proj, track_window,
                                                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))
        x_t, y_t, w_t, h_t = track_window

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            x_d, y_d, w_d, h_d = faces[0]
        else:
            x_d, y_d, w_d, h_d = (0, 0, 0, 0)

        iou_val = compute_iou((x_t, y_t, w_t, h_t), (x_d, y_d, w_d, h_d))
        iou_list.append(iou_val)
        frame_nums.append(frame_counter)

        if sample_high is None and iou_val > 0.5:
            sample_high = frame.copy()
            cv2.rectangle(sample_high, (x_t, y_t), (x_t+w_t, y_t+h_t), (0, 255, 0), 2)  # tracked: green
            cv2.rectangle(sample_high, (x_d, y_d), (x_d+w_d, y_d+h_d), (0, 0, 255), 2)  # detection: red

        if sample_low is None and iou_val < 0.1:
            sample_low = frame.copy()
            cv2.rectangle(sample_low, (x_t, y_t), (x_t+w_t, y_t+h_t), (0, 255, 0), 2)
            cv2.rectangle(sample_low, (x_d, y_d), (x_d+w_d, y_d+h_d), (0, 0, 255), 2)

        # cv2.imshow("Hue Tracking", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    # cv2.destroyAllWindows()

    high_iou_count = sum(1 for i in iou_list if i > 0.5)
    percentage = (high_iou_count / len(iou_list)) * 100
    print("Hue-based tracking: Percentage of frames with IoU > 50%: {:.2f}%".format(percentage))

    plt.figure()
    plt.plot(frame_nums, iou_list)
    plt.xlabel("Frame Number")
    plt.ylabel("IoU")
    plt.title("IoU Over Time (Hue Histogram)")
    plt.savefig("./assets/hue.png")
    plt.show()

    if sample_high is not None:
        cv2.imwrite("./assets/sample_high_iou_hue.png", sample_high)
    if sample_low is not None:
        cv2.imwrite("./assets/sample_low_iou_hue.png", sample_low)

def process_video_gradient(video_path):
    """Part 2 (Modified): Use sum-of-gradients histogram and mean shift tracking."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video")
        return

    # Read first frame and detect face
    ret, frame = cap.read()
    if not ret:
        print("Cannot read video")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        print("No face detected in the first frame.")
        return
    x, y, w, h = faces[0]
    track_window = (x, y, w, h)

    # Compute gradients in the ROI
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    Ix = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(Ix, Iy, angleInDegrees=True)

    roi_mag = magnitude[y:y+h, x:x+w]
    roi_angle = angle[y:y+h, x:x+w]

    max_mag = np.max(roi_mag)
    threshold = 0.05 * max_mag
    mask = roi_mag >= threshold

    roi_hist = np.zeros((24,), dtype=np.float32)
    bin_indices = (roi_angle[mask] // 15).astype(np.int32)
    bin_indices = np.clip(bin_indices, 0, 23)
    for i, bin_idx in enumerate(bin_indices):
        roi_hist[bin_idx] += roi_mag[mask][i]

    roi_hist = roi_hist.reshape(-1, 1)
    roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    iou_list = []
    frame_nums = []
    frame_counter = 1
    sample_high = None
    sample_low = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_counter += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
        Ix = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(Ix, Iy, angleInDegrees=True)

        x_t, y_t, w_t, h_t = track_window
        roi_mag_current = magnitude[y_t:y_t+h_t, x_t:x_t+w_t]
        if roi_mag_current.size > 0:
            max_mag_current = np.max(roi_mag_current)
        else:
            max_mag_current = 0
        curr_threshold = 0.01 * max_mag_current

        bin_indices = (angle // 15).astype(np.int32)
        bin_indices = np.clip(bin_indices, 0, 23)
        back_proj = roi_hist[bin_indices]

        back_proj[magnitude < curr_threshold] = 0
        back_proj = back_proj.astype(np.uint8)

        ret_val, track_window = cv2.meanShift(back_proj, track_window,
                                              (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.5))
        x_t, y_t, w_t, h_t = track_window

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            x_d, y_d, w_d, h_d = faces[0]
        else:
            x_d, y_d, w_d, h_d = (0, 0, 0, 0)

        iou_val = compute_iou((x_t, y_t, w_t, h_t), (x_d, y_d, w_d, h_d))
        iou_list.append(iou_val)
        frame_nums.append(frame_counter)

        if sample_high is None and iou_val > 0.7:
            sample_high = frame.copy()
            cv2.rectangle(sample_high, (x_t, y_t), (x_t+w_t, y_t+h_t), (0, 255, 0), 2)
            cv2.rectangle(sample_high, (x_d, y_d), (x_d+w_d, y_d+h_d), (0, 0, 255), 2)

        if sample_low is None and iou_val < 0.5:
            sample_low = frame.copy()
            cv2.rectangle(sample_low, (x_t, y_t), (x_t+w_t, y_t+h_t), (0, 255, 0), 2)
            cv2.rectangle(sample_low, (x_d, y_d), (x_d+w_d, y_d+h_d), (0, 0, 255), 2)

    cap.release()

    high_iou_count = sum(1 for i in iou_list if i > 0.5)
    percentage = (high_iou_count / len(iou_list)) * 100
    print("Gradient-based tracking (Sum of Gradients): IoU > 50% in {:.2f}% of frames".format(percentage))

    plt.figure()
    plt.plot(frame_nums, iou_list)
    plt.xlabel("Frame Number")
    plt.ylabel("IoU")
    plt.title("IoU Over Time (Gradient Histogram - Sum of Gradients)")
    plt.savefig("./assets/gradient_sum.png")
    plt.show()

    if sample_high is not None:
        cv2.imwrite("./assets/sample_high_iou_gradient_sum.png", sample_high)
    if sample_low is not None:
        cv2.imwrite("./assets/sample_low_iou_gradient_sum.png", sample_low)


if __name__ == '__main__':
    video_path = "KylianMbappe.mp4"  
    print("Part 1: Hue-based tracking")
    process_video_hue(video_path)
    print("Part 2: Gradient-based tracking")
    process_video_gradient(video_path)
