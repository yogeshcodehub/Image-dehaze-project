import cv2
import numpy as np
import argparse
import time


def compute_dark_channel(image, window_size=15, downscale_factor=0.5):
   
    downsampled = cv2.resize(image, (0, 0), fx=downscale_factor, fy=downscale_factor)
    min_channel = np.min(downsampled, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return cv2.resize(dark_channel, (image.shape[1], image.shape[0]))


def estimate_atmospheric_light(image, dark_channel, top_percent=0.1):
   
    num_pixels = int(dark_channel.size * top_percent)
    indices = np.argpartition(dark_channel.flatten(), -num_pixels)[-num_pixels:]
    return np.median(image.reshape(-1, 3)[indices], axis=0)


def dehaze_frame(frame, w=0.95, guided_filter_radius=40, eps=1e-3):
  
    I = frame.astype(np.float32) / 255.0
    dark_channel = compute_dark_channel(I)
    A = estimate_atmospheric_light(I, dark_channel)

    # Transmission estimation
    normalized_I = I / A
    transmission = 1 - w * compute_dark_channel(normalized_I)

    # Guided filter refinement
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    transmission = cv2.ximgproc.guidedFilter(gray, transmission, guided_filter_radius, eps)
    transmission = np.clip(transmission, 0.1, 1.0)

    # Scene recovery
    J = np.empty_like(I)
    for i in range(3):
        J[:, :, i] = (I[:, :, i] - A[i]) / transmission + A[i]

    return np.clip(J * 255, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description='Real-Time Image/Video Dehazing using DCP')
    parser.add_argument('--input', type=str, default='webcam', help='Input path (image/video) or "webcam"')
    args = parser.parse_args()

    # Input handling
    if args.input.lower() == 'webcam':
        cap = cv2.VideoCapture(0)
    elif args.input.endswith(('.jpg', '.png', '.jpeg')):
        image = cv2.imread(args.input)
        if image is None:
            print(f"Error: Could not read image {args.input}")
            return
        dehazed = dehaze_frame(image)
        cv2.imshow('Original', image)
        cv2.imshow('Dehazed', dehazed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    else:  # Video file
        cap = cv2.VideoCapture(args.input)

    # Video processing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        dehazed_frame = dehaze_frame(frame)
        fps = 1 / (time.time() - start_time)

        cv2.putText(dehazed_frame, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Hazy Input', frame)
        cv2.imshow('Dehazed Output', dehazed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
