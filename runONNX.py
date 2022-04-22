import argparse
import cv2
import os
import sys
import numpy as np
import onnxruntime
import time
from pytorch.runONNXyolo5_py import mainYOLOv5
from loguru import logger
import datetime
import random
import colorsys
#from tf1.runONNXFasterRCNN import mainFasterRCNN
from DatasetClass import Ship_classNames
from tqdm import tqdm
from thop import profile

def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors
def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return rgb_colors

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default='', help='image file, eg:--img PATH/demo.jpg')
    parser.add_argument('--video',  type=str, default='', help='video file, eg:--video PATH/demo.mp4')
    parser.add_argument('--modelname', type=str, default='', help='file of detection model, eg:PATH/demo.model')#'./FasterRCNN.model'
    parser.add_argument('--flagShow',action='store_true',help='whether to show the results,eg:--flagShow',)
    parser.add_argument('--saveDir',type=str,default='',help='if want to save the result of image or video, Please place the Path,eg:--saveDir Path')
    parser.add_argument('--log', type=str, default='',
                        help='Using default name if None')
    config = parser.parse_args()

    NUM = len(Ship_classNames)  # - 1
    colors = np.array(ncolors(NUM))

    imgPath = config.img  # MVI_1592_VIS_00462
    videoPath=config.video
    onnxfilename=config.modelname
    flagShow=config.flagShow
    saveDir=config.saveDir
    log=config.log
    if not log:
        log='log'+datetime.datetime.now().strftime('%Y%m%d%H%M%S')+'.log'
    logger.add(log, mode='a')
    logger.info(str(config))
    # ImageWriter=None
    if videoPath is None and imgPath is None:
        parser.print_help()
        logger.warning("Both videoPath and imgPath are None!")
        sys.exit(1)
        #raise ValueError('Error:None of img and video')
    try:
        onnxSession=onnxruntime.InferenceSession(onnxfilename)
    except:
        logger.error("Error for Reading the detection model:%s" % (onnxfilename))
        sys.exit(1)

    if imgPath:
        logger.info('imgPath:%s'%(imgPath))
        img = cv2.imread(imgPath)
        if img is None:
            logger.error("img is None:The imgPath doesn't exist!")
            sys.exit(1)
        logger.info("Reading the detection model:%s"%(onnxfilename))
        logger.info("detecting..." )
        a=time.time()
        outImg,det=mainYOLOv5(img,onnxSession,colors)
        b=time.time()
        logger.info("detection finished:%fs"%(b-a))
        if flagShow:
            logger.warning("please input any key for continue!!!")
            cv2.imshow("imgResult",outImg)
            cv2.waitKey(10)
        if saveDir:
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
            basename=os.path.basename(imgPath)
            savingfile=os.path.join(saveDir, basename)
            logger.info("Saving the image%s"%(savingfile))
            cv2.imwrite(savingfile,outImg)

    VideoWriter = None
    if videoPath:
        logger.info('videoPath:%s' % (videoPath))
        cap=cv2.VideoCapture(videoPath)
        if not cap.isOpened():
            logger.error("Video is None:The videofile doesn't exist!")
            sys.exit(1)
        totalFrame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.warning("Total frame number is %d"%(totalFrame))
        if saveDir:
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
            fourcc=cv2.VideoWriter_fourcc(*'XVID')
            fps=round(cap.get(cv2.CAP_PROP_FPS))
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            basename = os.path.basename(videoPath)
            basename=os.path.splitext(basename)[0]
            savingfile = os.path.join(saveDir, basename+'.avi')
            logger.info("write the video results:%s"%(savingfile))
            VideoWriter=cv2.VideoWriter(savingfile,fourcc,fps,(width,height),True)

        logger.info('reading the video!')

        total_time=0
        #count=0
        for count in tqdm(range(totalFrame)):
            ret, frame = cap.read()
            if not ret:
                break
            logger.info("Video detecting.....")
            a = time.time()
            outImg, det = mainYOLOv5(frame, onnxSession, colors)
            b = time.time()
            logger.info("frame %d finished:%fs" % (count,b - a))
            total_time+=(b-a)
            if flagShow:
                #logger.warning("please input any key for continue!!!")
                cv2.imshow("VideoResult", outImg)
                waitK=cv2.waitKey(2)
                if waitK==ord('q'):
                    break
            if VideoWriter:
                logger.info("writing the video!")
                VideoWriter.write(outImg)
            cap.read()

        cap.release()
        VideoWriter.release()
        cv2.destroyAllWindows()
        logger.info("video detection average tiem per frame:%f" % (total_time / count))







    # else:
    #     print('FasterRCNN')
    #     #onnxfilename = './fasterRCNN.onnx'  # ’fasterRCNN.onnx‘
    #     onnxSession = onnxruntime.InferenceSession(onnxfilename)
    #     mainFasterRCNN(img,onnxSession,flagShow)

