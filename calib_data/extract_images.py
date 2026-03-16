import cv2
import cv2.aruco as aruco
import rosbag
from cv_bridge import CvBridge
import cv2

bridge = CvBridge()

if len(sys.argv) < 2:
    print("Usage: python extract_images.py <bag_name.bag>")
    sys.exit(1)

bag_name = sys.argv[1]

#change name of bag
with rosbag.Bag(bag_name, 'r') as bag:
    count = 0
    for topic, msg, t in bag.read_messages(topics=['/camera/image_raw']):
        cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        detector_params = aruco.DetectorParameters_create()
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)
        num_markers = len(corners) if ids is not None else 0
        if num_markers == 4:
            count += 1
            if count % 50 == 0:
                cv2.imwrite(f'scene{count}.png', cv_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                print(f"✓ Saved image: {cv_img.shape[1]}x{cv_img.shape[0]}, {cv_img.dtype}")
print(f"Saved {count //50} images.")
                
           