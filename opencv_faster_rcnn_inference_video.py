import cv2
import time
import matplotlib.pyplot as plt

cv_net = cv2.dnn.readNetFromTensorflow('./pretrained/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb',
                                       './pretrained/config_graph.pbtxt')

# OpenCV Tensorflow Faster-RCNN용
labels_to_names_0 = {0:'person',1:'bicycle',2:'car',3:'motorcycle',4:'airplane',5:'bus',6:'train',7:'truck',8:'boat',9:'traffic light',
                    10:'fire hydrant',11:'street sign',12:'stop sign',13:'parking meter',14:'bench',15:'bird',16:'cat',17:'dog',18:'horse',19:'sheep',
                    20:'cow',21:'elephant',22:'bear',23:'zebra',24:'giraffe',25:'hat',26:'backpack',27:'umbrella',28:'shoe',29:'eye glasses',
                    30:'handbag',31:'tie',32:'suitcase',33:'frisbee',34:'skis',35:'snowboard',36:'sports ball',37:'kite',38:'baseball bat',39:'baseball glove',
                    40:'skateboard',41:'surfboard',42:'tennis racket',43:'bottle',44:'plate',45:'wine glass',46:'cup',47:'fork',48:'knife',49:'spoon',
                    50:'bowl',51:'banana',52:'apple',53:'sandwich',54:'orange',55:'broccoli',56:'carrot',57:'hot dog',58:'pizza',59:'donut',
                    60:'cake',61:'chair',62:'couch',63:'potted plant',64:'bed',65:'mirror',66:'dining table',67:'window',68:'desk',69:'toilet',
                    70:'door',71:'tv',72:'laptop',73:'mouse',74:'remote',75:'keyboard',76:'cell phone',77:'microwave',78:'oven',79:'toaster',
                    80:'sink',81:'refrigerator',82:'blender',83:'book',84:'clock',85:'vase',86:'scissors',87:'teddy bear',88:'hair drier',89:'toothbrush',
                    90:'hair brush'}

# VideoCapture 와 VideoWriter 설정하기
# - VideoCapture 를 이용하여 Video 를 frame 별로 capture 할 수 있도록 설정
# - VideoCapture 의 속성을 이용하여 Video Frame 의 크기 및 FPS 설정
# - VideoWriter 를 위한 인코딩 코덱 설정 및 영상 write를 위한 설정

video_input_path = './data/John_Wick_small.mp4'

cap = cv2.VideoCapture(video_input_path)
codec = cv2.VideoWriter_fourcc(*'XVID')

frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('총 Frame 개수:', frame_cnt)

video_output_path = './data/John_Wick_small_cv01.mp4'
vid_size = (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
vid_fps = cap.get(cv2.CAP_PROP_FPS)

vid_writer = cv2.VideoWriter(video_output_path, codec, vid_fps, vid_size)

frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('총 Frame 개수:', frame_cnt)

# 총 frame 별로 iteration 하면서 object detection 수행. 개별 frame 별로 단일 이미지 Object Detection 과 유사

# bounding box 의 테두리와 caption 글자색 지정
green_color = (0, 255, 0)
red_color = (0, 0, 255)

# while True:
#     # 더 이상 읽을 frame 이 없을 시 hasFrame 은 None 으로 반환된다.
#     hasFrame, img_frame = cap.read()
#     if not hasFrame:
#         print('더 이상 처리할 frame이 없습니다.')
#         break
#
#     rows = img_frame.shape[0]
#     cols = img_frame.shape[1]
#
#     # 원본 이미지 배열 RGB 로 변환
#     cv_net.setInput(cv2.dnn.blobFromImage(img_frame, swapRB=True, crop=False))
#
#     start = time.time()
#     # object detection 수행하여 결과를 cv_out 으로 반환
#     cv_out = cv_net.forward()
#     frame_index = 0
#     # detected 된 object 들을 iteration 하면서 정보 추출
#     for detection in cv_out[0,0,:,:]:
#         score = float(detection[2])
#         class_id = int(detection[1])
#         # detected된 object들의 score가 0.5 이상만 추출
#         if score > 0.5:
#             # detected된 object들은 scale된 기준으로 예측되었으므로 다시 원본 이미지 비율로 계산
#             left = detection[3] * cols
#             top = detection[4] * rows
#             right = detection[5] * cols
#             bottom = detection[6] * rows
#             # labels_to_names_0딕셔너리로 class_id값을 클래스명으로 변경.
#             caption = "{}: {:.4f}".format(labels_to_names_0[class_id], score)
#             #print(class_id, caption)
#             #cv2.rectangle()은 인자로 들어온 draw_img에 사각형을 그림. 위치 인자는 반드시 정수형.
#             cv2.rectangle(img_frame, (int(left), int(top)), (int(right), int(bottom)), color=green_color, thickness=2)
#             cv2.putText(img_frame, caption, (int(left), int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color, 1)
#     print('Detection 수행 시간:', round(time.time()-start, 2),'초')
#     vid_writer.write(img_frame)
# # end of while loop
#
# vid_writer.release()
# cap.release()

####################################################################################

# do_detected_video 함수에서 사용될 이미지 frame 별 처리 함수
def get_detected_img(cv_net, img_array, score_threshold, use_copied_array=True, is_print=True):

    rows = img_array.shape[0]
    cols = img_array.shape[1]

    draw_img = None
    if use_copied_array:
        draw_img = img_array.copy()
    else:
        draw_img = img_array

    cv_net.setInput(cv2.dnn.blobFromImage(img_array, swapRB=True, crop=False))

    start = time.time()
    cv_out = cv_net.forward()

    green_color=(0, 255, 0)
    red_color=(0, 0, 255)

    # detected 된 object들을 iteration 하면서 정보 추출
    for detection in cv_out[0,0,:,:]:
        score = float(detection[2])
        class_id = int(detection[1])
        # detected된 object들의 score가 함수 인자로 들어온 score_threshold 이상만 추출
        if score > score_threshold:
            # detected된 object들은 scale된 기준으로 예측되었으므로 다시 원본 이미지 비율로 계산
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            # labels_to_names 딕셔너리로 class_id값을 클래스명으로 변경. opencv에서는 class_id + 1로 매핑해야함.
            caption = "{}: {:.4f}".format(labels_to_names_0[class_id], score)
            print(caption)
            #cv2.rectangle()은 인자로 들어온 draw_img에 사각형을 그림. 위치 인자는 반드시 정수형.
            cv2.rectangle(draw_img, (int(left), int(top)), (int(right), int(bottom)), color=green_color, thickness=2)
            cv2.putText(draw_img, caption, (int(left), int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, red_color, 1)
    if is_print:
        print('Detection 수행시간:',round(time.time() - start, 2),"초")

    return draw_img
# video detection 전용 함수 생성.
def do_detected_video(cv_net, input_path, output_path, score_threshold, is_print):

    cap = cv2.VideoCapture(input_path)

    codec = cv2.VideoWriter_fourcc(*'XVID')

    vid_size = (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    vid_fps = cap.get(cv2.CAP_PROP_FPS)

    vid_writer = cv2.VideoWriter(output_path, codec, vid_fps, vid_size)

    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('총 Frame 갯수:', frame_cnt)

    green_color=(0, 255, 0)
    red_color=(0, 0, 255)
    while True:
        hasFrame, img_frame = cap.read()
        if not hasFrame:
            print('더 이상 처리할 frame이 없습니다.')
            break

        img_frame = get_detected_img(cv_net, img_frame, score_threshold=score_threshold, use_copied_array=False, is_print=is_print)

        vid_writer.write(img_frame)
    # end of while loop

    vid_writer.release()
    cap.release()

do_detected_video(cv_net, './data/V2.mp4', './data/V2_01.mp4', 0.2, True)
