import random
from numpy import *

def loadDataSet(fileName):
    """
    数据加载
    :param fileName:
    :return:
    """
    dataMat = []
    lableMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        print(lineArr)
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        lableMat.append(float(lineArr[2]))

    return dataMat,lableMat


def selectJrand(i,m):
    """
    随机选择第二个变量j
    :param i:
    :param j:
    :return:
    """
    j=i
    while(j==i):
       j = int(random.uniform(0,m))
    return j


def clipAlpha(aj,H,L):
    if aj>H:
        aj = H
    if aj<L:
        aj = L
    return aj


def smoSimple(dataMatIn,classLables,C,toler,maxIter):
    """
    简易的smo算法主体
    :param dataMatIn:
    :param classLables:
    :param C:
    :param toler:
    :param maxIter:
    :return:
    """
    dataMat = mat(dataMatIn)
    lableMat = mat(classLables).transpose()
    b = 0
    m,n = shape(dataMat)
    alphas = mat(zeros((m,1)))
    iter = 0
    while(iter<maxIter):
        #记录是否有alpha被优化
        alphaPairsChanged = 0
        for i in range(m):
            # a = multiply(alphas,lableMat.T).T
            # b = dataMat*dataMat[i,:].T
            # print("a",shape(a))
            # y = multiply(alphas,lableMat.T).T*(dataMat*dataMat[i,:].T)
            # print("b",shape(b))
            # print("y",shape(y))
            fXi =float(multiply(alphas,lableMat).T*(dataMat*dataMat[i,:].T)) +b
            Ei = fXi - float(lableMat[i,])
            #简易的选择条件
            if((lableMat[i]*Ei<-toler and alphas[i]<C) or \
                    (lableMat[i]*Ei>toler and alphas[i]>0)):
                #随机选择第二个
                j = selectJrand(i,m)
                fXj = float(multiply(alphas,lableMat).T*(dataMat*dataMat[j,:].T)) +b
                Ej = fXj - float(lableMat[j])
                alphaiOld = alphas[i].copy()
                alphajOld = alphas[j].copy()

                #对j进行修改
                if(lableMat[i]!=lableMat[j]):
                    L = max(0,alphas[j]-alphas[i])
                    H = min(C,C+alphas[j]-alphas[i])
                else:
                    L = max(0,alphas[i]+alphas[j]-C)
                    H = min(C,alphas[i]+alphas[j])
                if L == H :print("L==H"); continue
                eta = 2.0*dataMat[i,:]*dataMat[j,:].T - \
                      dataMat[i,:]*dataMat[i,:].T - dataMat[j,:]*dataMat[j,:].T
                if(eta>=0): print("eta>=0");continue
                alphas[j] -=lableMat[j]*(Ei-Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                #对i进行修改
                if (abs(alphas[j]-alphajOld)<0.0001):print("j not moving enough");continue
                alphas[i] +=lableMat[i]*lableMat[j]*(alphajOld-alphas[j])

                #修改b
                bi = b-Ei-lableMat[i]*dataMat[i,:]*dataMat[i,:].T*(alphas[i]-alphaiOld) - \
                       lableMat[j]*dataMat[j,:]*dataMat[i,:].T*(alphas[j]-alphajOld)

                bj = b-Ei - lableMat[i]*dataMat[i,:]*dataMat[j,:].T*(alphas[i]-alphaiOld)-\
                       lableMat[j]*dataMat[j,:]*dataMat[j,:].T*(alphas[j]-alphajOld)

                if(0<alphas[i]) and (C>alphas[i]):b = bi
                elif(0<alphas[j]) and (C>alphas[j]):b = bj
                else: b = (bi+bj)/2.0

                alphaPairsChanged +=1
                print ("iter: %d i: %d,pairschanged %d" %(iter,i,alphaPairsChanged))
        if(alphaPairsChanged==0): iter+= 1
        else: iter = 0
        print("iteration number: %d",iter)
    return b,alphas


if __name__=='__main__':
    fileName = "E:\\ML\\SVM\\testSet_line.txt"
    dataMatIn,classLables =loadDataSet(fileName)
    b,alphas = smoSimple(dataMatIn,classLables,200,0.001,40)
    print(alphas[alphas>0])
    print(b)

