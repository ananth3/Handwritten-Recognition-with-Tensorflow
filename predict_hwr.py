from darkflow.net.build import TFNet
import cv2
import numpy as np
import pytesseract
# import pyocr
from PIL import Image
# import pyocr.builders
import pandas as pd

from HTR.src import main
from HTR.src.main import infer
from HTR.src import Model
from HTR.src.Model import Model,DecoderType

from HDR.hwr_digit import hwr_digit,hwr_digits

options = {"model": "cfg/tiny-yolo-voc11.cfg", "load": "bin/tiny-yolo-voc.weights", "threshold": 0.01, "load": 4800}
tfnet = TFNet(options)


def yolo_pred(image):
    imgcv = cv2.imread(image)
    result = tfnet.return_predict(imgcv)
    df = pd.DataFrame(result)
    opt = []
    # output={}
    for i in df.label.unique():
        tdf=df[df['label'] == i]
        tmax = tdf.confidence.max()
        tdf = tdf[tdf['confidence'] == tmax]
        opt.append([i,tdf.iloc[0]['topleft'],tdf.iloc[0]['bottomright']])
    print(opt)

# bank name
    image=imgcv[opt[0][1]['y']:opt[0][2]['y'],opt[0][1]['x']:opt[0][2]['x']]
    bank_name=tesseract(image)
    bank_name=bank_name.split('\n')[-1]

#ifsc
    image=imgcv[opt[1][1]['y']:opt[1][2]['y'],opt[1][1]['x']:opt[1][2]['x']]
    h,w=image.shape[:2]
    image=cv2.resize(image,(w,int(h*1.5)),interpolation=cv2.INTER_AREA)
    ifsc=tesseract(image)
    ifsc=ifsc.split('\n')[-1]
    ifsc=ifsc.split(':')[-1]

# pay name
    image=imgcv[opt[2][1]['y']:opt[2][2]['y'],opt[2][1]['x']:opt[2][2]['x']]
    name=htr_predict(image)

# amount
    image=imgcv[opt[3][1]['y']:opt[3][2]['y'],opt[3][1]['x']:opt[3][2]['x']]
    amt=htr_digit(image)

#account number
    image=imgcv[opt[4][1]['y']:opt[4][2]['y'],opt[4][1]['x']:opt[4][2]['x']]
    acc_no=htr_digits(image)

#sign
    image=imgcv[opt[5][1]['y']:opt[5][2]['y'],opt[5][1]['x']:opt[5][2]['x']]
    sign=tesseract(image)

#cheque number
    image=imgcv[opt[6][1]['y']:opt[6][2]['y'],opt[6][1]['x']:opt[6][2]['x']]
    chq_no=micr_font(image)
    chq_no=chq_no.replace('<','')

#micr number
    image=imgcv[opt[7][1]['y']:opt[7][2]['y'],opt[7][1]['x']:opt[7][2]['x']]
    micr_no=micr_font(image)
    micr_no=micr_no.replace('<','')
    micr_no=micr_no.split(':')[0]

    output={'bank name': bank_name, 'ifsc': ifsc, 'pay name': name, 'amount': amt, 'acc no': acc_no, 'sign': sign, 'cheque no': chq_no, 'micr no': micr_no  }

    return output

    # for i in range(len(opt)):
    #     if opt[i][0] == 'bank_name' or opt[i][0] == 'cheque_no' or opt[i][0] == 'micr_no':
    #         image=imgcv[opt[i][1]['y']:opt[i][2]['y'],opt[i][1]['x']:opt[i][2]['x']]
    #         data=tesseract(image)
    #         return data
    #
    #     if opt[i][0] == 'ifsc':
    #         image=imgcv[opt[i][1]['y']:opt[i][2]['y'],opt[i][1]['x']:opt[i][2]['x']]
    #         (h,w)=image.shape[:2]
    #         image=cv2.resize(image, (w, int(h*1.5)), interpolation=cv2.INTER_AREA)
    #         # cv2.imwrite('ifsc1.jpg',image)
    #         # image=cv2.imread('/home/ananth/Downloads/Text_extraction_chqimages/yolo/darkflow/sample_img/test/ifsc.jpg')
    #         data=tesseract(image)
    #         return data
    #     #
    #     if opt[i][0] == 'p_name':
    #         image=imgcv[opt[i][1]['y']:opt[i][2]['y'],opt[i][1]['x']:opt[i][2]['x']]
    #         # image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #         # ret,image=cv2.threshold(image,130,255,cv2.THRESH_BINARY)
    #         # cv2.imwrite('pname.jpg',image)
    #         image=cv2.imread('/home/ananth/Downloads/Text_extraction_chqimages/yolo/darkflow/pname_1_crop.png')
    #         htr_predict(image)
    #
    #     if opt[i][0] == 'amount':
    #         image=imgcv[opt[i][1]['y']:opt[i][2]['y'],opt[i][1]['x']:opt[i][2]['x']]
    #         # cv2.imwrite('dig.jpg',image)
    #         data=htr_digit(image)
    #         return data
    #
    #     if opt[i][0] == 'acc_no':
    #         image=imgcv[opt[i][1]['y']:opt[i][2]['y'],opt[i][1]['x']:opt[i][2]['x']]
    #         data=htr_digits(image)
    #         return data

        # print(opt[i][0])
    # return opt


def htr_predict(image):
    decoderType = DecoderType.BestPath
    model = Model(open('/home/ananth/Downloads/Text_extraction_chqimages/yolo/darkflow/HTR/model/charList.txt').read(), decoderType, mustRestore=True, dump=False)
    # cv2.imshow('roi', image)
    # cv2.waitKey(2000)
    # cv2.destroyAllWindows()
    hwr_text=infer(model,image)
    return hwr_text

def htr_digit(image):
    # cv2.imshow('roi', image)
    # cv2.waitKey(2000)
    # cv2.destroyAllWindows()
    h_digit=hwr_digit(image)
    return h_digit

def htr_digits(image):
    # cv2.imshow('roi', image)
    # cv2.waitKey(2000)
    # cv2.destroyAllWindows()
    h_digits=hwr_digits(image)
    return h_digits

def tesseract(image):
    # cv2.imshow('roi', image)
    # cv2.waitKey(2000)
    # cv2.destroyAllWindows()
    text_data=pytesseract.image_to_string(image)
    return text_data

def micr_font(image):
    # cv2.imshow('roi', image)
    # cv2.waitKey(2000)
    # cv2.destroyAllWindows()
    text_data=pytesseract.image_to_string(image,lang='e13b')
    return text_data

data=yolo_pred('/home/ananth/Downloads/Text_extraction_chqimages/yolo/darkflow/sample_img/hw_test/chq1.jpeg')

print('#'*30+'output'+'#'*30)
print('\n')

for k in data.keys():
    val=data[k]
    print(k,': ',val)
