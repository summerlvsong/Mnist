import os
import re
import numpy as np
from math import pow

# input the raw data
# output the feature data and label
def GetSeqMnistAndCalSig(labelPath, dataPath, sampleClip, logSigDgree, logSigDim, sigFun):

    #read the ground-truth
    grountruthPATH = labelPath # 'D:/England/pythonWork/python/python/testlabels.txt'
    grountruthFile = open(grountruthPATH)
    grountruth = grountruthFile.read()
    grountruthFile.close()
    grountruthAll = grountruth.split('\n')
    grountruthAllV = [int(x) for x in grountruthAll]
    mnistlabel = np.zeros((sampleClip,10))
    count = 0;
    for x in grountruthAllV:
        if(count == sampleClip):
            break
        mnistlabel[count,x] = 1;
        count = count + 1;

    #read the mnist data
    mnistPath = dataPath # 'D:/England/pythonWork/python/python/test.txt';
    mnistf = open(mnistPath)
    mnist = mnistf.read()
    mnistf.close()
    mnistSeq = mnist.split('\n')
    mnistData = np.zeros((sampleClip,logSigDim))
    count = 0;
    for x in mnistSeq:
        if(count%100==0):
            print(str(count)+' / '+str(sampleClip))
        if(count == sampleClip):
            break

        #data preprocessing; split data, remove '' and -1 -2, and string-to-digit
        tmpList = re.split(r',| *',x)
        if(tmpList[-1] == ''):
            tmpList.pop()
        if(tmpList[-1] == '\r'):
            tmpList.pop()        
        tmpListV = [int(y) for y in tmpList]
        tmpListV = [x for x in tmpListV if x != -1]
        tmpListV.pop()
        tmpListV.pop()
        tmpL = int(len(tmpListV)/2)
    
        #extract log signature : normalize
        logSigArray = np.zeros((tmpL,2))
        xV = tmpListV[1::2]
        yV = tmpListV[::2]
        minx = min(xV)
        maxx = max(xV)
        miny = min(yV)
        maxy = max(yV)
        scale = float(2)/max([(maxx-minx),(maxy-miny)])
        logSigArray[:,0] = [(x-(maxx+minx)/2)*scale for x in xV]
        logSigArray[:,1] = [(y-(maxy+miny)/2)*scale for y in yV]
        b = sigFun(logSigArray,logSigDgree)
        #stroe data in a big arrary
        mnistData[count,:] = b;
        count = count + 1;
        #visulize the normalize digit
        #plt.plot(logSigArray[:,0],logSigArray[:,1])
        #plt.show()
    
    #visulize the normalize digit
    #print(mnistData[1:5,:])
    return [mnistData,mnistlabel]













# get the raw data, i.e., the x, y, points
# return listof label listof train data
def GetSeqMnistRawData(labelPath, dataPath, sampleClip):
    #read the ground-truth
    grountruthPATH = labelPath # 'D:/England/pythonWork/python/python/testlabels.txt'
    grountruthFile = open(grountruthPATH)
    grountruth = grountruthFile.read()
    grountruthFile.close()
    grountruthAll = grountruth.split('\n')
    grountruthAllV = [int(x) for x in grountruthAll]
    mnistlabel = np.zeros((sampleClip,10))
    count = 0;
    for x in grountruthAllV:
        if(count == sampleClip):
            break
        mnistlabel[count,x] = 1;
        count = count + 1;

    #read the mnist data
    mnistPath = dataPath # 'D:/England/pythonWork/python/python/test.txt';
    mnistf = open(mnistPath)
    mnist = mnistf.read()
    mnistf.close()
    mnistSeq = mnist.split('\n')
    mnistData = [0]*sampleClip
    count = 0;
    for sample in mnistSeq:
        if(count%10000==0):
            print(str(count)+' / '+str(sampleClip))
        if(count == sampleClip):
            break

        #data preprocessing; split data, remove '' and -1 -2, and string-to-digit
        tmpList = re.split(r',| *',sample)
        if(tmpList[-1] == ''):
            tmpList.pop()
        if(tmpList[-1] == '\r'):
            tmpList.pop()        
        tmpListV = [int(y) for y in tmpList]
        tmpListV = [x for x in tmpListV if x != -1]
        tmpListV.pop()
        tmpListV.pop()
        tmpL = int(len(tmpListV)/2)
        
        xV = tmpListV[1::2]
        yV = tmpListV[::2]
        minx = min(xV)
        maxx = max(xV)
        miny = min(yV)
        maxy = max(yV)
        scale = float(2)/max([(maxx-minx),(maxy-miny)])
        list_x = [(x-(maxx-minx)/2)*scale for x in xV]
        list_y = [(y-(maxx-minx)/2)*scale for y in yV]

        mnistData[count] = list_x+list_y
        count = count + 1
    return [mnistData,mnistlabel]



# get the raw data, i.e., the x, y, points
# return basic fea for LSTM
def GetSeqMnistGenerateLSTMFea(dataPath, sampleClip):

    #read the mnist data
    mnistPath = dataPath # 'D:/England/pythonWork/python/python/test.txt';
    mnistf = open(mnistPath)
    mnist = mnistf.read()
    mnistf.close()
    mnistSeq = mnist.split('\n')
    mnistData = [0]*sampleClip
    count = 0;
    max_len = 0;
    for sample in mnistSeq:
        if(count%5000==0):
            print(str(count)+' / '+str(sampleClip))
        if(count == sampleClip):
            break

        #data preprocessing; split data, remove '' and -1 -2, and string-to-digit
        tmpList = re.split(r',| *',sample)
        if(tmpList[-1] == ''):
            tmpList.pop()
        if(tmpList[-1] == '\r'):
            tmpList.pop()        
        tmpListV = [int(y) for y in tmpList]
        tmpListV.pop()
        tmpListV.pop()


        xVN = []
        yVN = []
        penUpV = []
        penDownV = []
        fd = 0;
        xV = tmpListV[1::2]
        yV = tmpListV[::2]

        for index in range(len(xV)):
            x = xV[index];
            y = yV[index];
            if x != -1:
                xVN.append(x)
                yVN.append(y)
                penUpV.append(0)
                if(fd == -1):
                    penDownV.append(1)
                else:
                    penDownV.append(0)
                fd = x;
            else:
                penUpV[-1] = 1

        penDownV[0] = 1
        penUpV[-1] = 1

        minx = min(xVN)
        maxx = max(xVN)
        miny = min(xVN)
        maxy = max(xVN)
        scale = float(2)/max([(maxx-minx),(maxy-miny)])
        list_x = [(x-(maxx-minx)/2)*scale for x in xVN]
        list_y = [(y-(maxx-minx)/2)*scale for y in yVN]

        deltaXV = []
        deltaYV = []
        for index in range(len(list_x)-1):
            deltaXV.append(list_x[index+1]-list_x[index])
            deltaYV.append(list_y[index+1]-list_y[index])
        deltaXV.append(0)
        deltaYV.append(0)

        mnistData[count] = list_x+list_y+deltaXV+deltaYV+penUpV+penDownV
        if(max_len < len(list_x)):
            max_len = len(list_x)
        count = count + 1
    return mnistData, max_len


# get the raw data, i.e., the x, y, points
# return basic fea for LSTM
def GetSeqMnistGenerateLSTMFeaSparse(dataPath, sampleClip, sparse):

    #read the mnist data
    mnistPath = dataPath # 'D:/England/pythonWork/python/python/test.txt';
    mnistf = open(mnistPath)
    mnist = mnistf.read()
    mnistf.close()
    mnistSeq = mnist.split('\n')
    mnistData = [0]*sampleClip
    count = 0;
    max_len = 0;
    for sample in mnistSeq:
        if(count%5000==0):
            print(str(count)+' / '+str(sampleClip))
        if(count == sampleClip):
            break

        #data preprocessing; split data, remove '' and -1 -2, and string-to-digit
        tmpList = re.split(r',| *',sample)
        if(tmpList[-1] == ''):
            tmpList.pop()
        if(tmpList[-1] == '\r'):
            tmpList.pop()        
        tmpListV = [int(y) for y in tmpList]
        tmpListV.pop()
        tmpListV.pop()


        xVN = []
        yVN = []
        penUpV = []
        penDownV = []
        fd = 0;
        xV = tmpListV[1::2]
        yV = tmpListV[::2]

        for index in range(len(xV)):
            x = xV[index];
            y = yV[index];
            if x != -1:
                xVN.append(x)
                yVN.append(y)
                penUpV.append(0)
                if(fd == -1):
                    penDownV.append(1)
                else:
                    penDownV.append(0)
                fd = x;
            else:
                penUpV[-1] = 1

        penDownV[0] = 1
        penUpV[-1] = 1

        minx = min(xVN)
        maxx = max(xVN)
        miny = min(xVN)
        maxy = max(xVN)
        scale = float(2)/max([(maxx-minx),(maxy-miny)])
        list_x = [(x-(maxx-minx)/2)*scale for x in xVN]
        list_y = [(y-(maxx-minx)/2)*scale for y in yVN]

        deltaXV = []
        deltaYV = []
        for index in range(len(list_x)-1):
            deltaXV.append(list_x[index+1]-list_x[index])
            deltaYV.append(list_y[index+1]-list_y[index])
        deltaXV.append(0)
        deltaYV.append(0)

        mnistData[count] = list_x[::sparse]+list_y[::sparse]+deltaXV[::sparse]+deltaYV[::sparse]+penUpV[::sparse]+penDownV[::sparse]
        if(max_len < len(list_x)):
            max_len = len(list_x)
        count = count + 1
    return mnistData, max_len


# output the label data
def GetLabelFromFile(labelPath, sampleN):
    #read the ground-truth
    grountruthPATH = labelPath # 'D:/England/pythonWork/python/python/testlabels.txt'
    grountruthFile = open(grountruthPATH)
    grountruth = grountruthFile.read()
    grountruthFile.close()
    grountruthAll = grountruth.split('\n')
    grountruthAllV = [int(x) for x in grountruthAll]
    mnistlabel = np.zeros((sampleN,10))
    count = 0;
    for x in grountruthAllV:
        if(count == sampleN):
            break
        mnistlabel[count,x] = 1;
        count = count + 1;
    return mnistlabel    


# output the label data
def GetLabelNumFromFile(labelPath, sampleN):
    #read the ground-truth
    grountruthPATH = labelPath # 'D:/England/pythonWork/python/python/testlabels.txt'
    grountruthFile = open(grountruthPATH)
    grountruth = grountruthFile.read()
    grountruthFile.close()
    grountruthAll = grountruth.split('\n')
    grountruthAllV = [int(x) for x in grountruthAll]
    mnistlabel = np.zeros((sampleN,1))
    count = 0;
    mnistlabel[:,0] = grountruthAllV
    return mnistlabel        








if __name__ == "__main__":
    main()