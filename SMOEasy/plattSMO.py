from numpy import *
from SMOEasy.svmMLiA import *
import matplotlib.pyplot as plt

class optStruct:
    def __init__(self,dataMatIn,classLables,C,toler,kTup):
        self.X = mat(dataMatIn)
        self.lableMat = mat(classLables)
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2)))
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X,self.X[i,:],kTup)


def drawDataMap(dataArr,labelArr,b,alphas):
    #获取alpha大于0的索引列表
    svInd = nonzero(alphas.A>0)[0]
    #分类数据点
    classfied_pts = {'+1':[],'-1':[]}
    for point,label in zip(dataArr,labelArr):
        if label ==1.0:
            classfied_pts['+1'].append(point)
        else:
            classfied_pts['-1'].append(point)

    #绘制数据点
    for label,pts in classfied_pts.items():
        pts = array(pts)
        plt.plot(pts[:,0],pts[:,1],'o',label='point')
        #ax.scatter(pts[:,0],pts[:,1],label=label)

    #绘制分离超平面
    # w1,w2 = calcws(alphas,dataArr,mat(labelArr))
    # x1 = linspace(-6,4)
    # x2 = w1/w2*x1
    # plt.plot(x1,x2)
    #绘制支持向量
    for i in svInd:
        x,y = dataArr[i]
        print("支持向量",x,y)
        plt.plot(x,y,'v',label='support')
        #ax.scatter(x,y,c = 'b',marker ='v')
    plt.legend(loc='lower right')
    plt.show()




def calcEk(oS,k):
    fXk = float(multiply(oS.alphas,oS.lableMat).T*oS.K[:,k]) +oS.b
    Ek = fXk - float(oS.lableMat[k])
    return Ek

def selectJ(oS,i):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    Ei = calcEk(oS,i)
    oS.eCache[i] = [i,Ei]
    validECacheList = nonzero(oS.eCache[:,0].A)[0]
    if(len(validECacheList)>0):
        for k in validECacheList:
            if k == i :continue
            deltaE = abs(Ei-Ej)
            if(deltaE>maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
        return maxK,Ej
    else:
        j = selectJrand(i,oS.m)
        Ej = calcEk(oS,j)
        return j,Ej

def updateEk(oS,k):
    """
    将更新好的Ek添加进缓存中，这里的1指的是有效的意思，也就是计算好了
    :param oS:
    :param k:
    :return:
    """
    Ek = calcEk(oS,k)
    oS.eCache[k] = [1,Ek]


def innerL(i,oS):
    """
    内层循环，在机器学习实战第六章的内层循环的筛选条件我认为无法与李航的统计机器学习实战的条件完全匹配
    故我使用李航的筛选条件
    :param i:
    :param oS:
    :return:
    """
    Ei = calcEk(oS,i)
    # if ((0<oS.alphas[i] and oS.alphas[i]<oS.C) and abs(oS.lableMat[i]*Ei-1)>oS.tol) or \
    #      (oS.alphas[i]==0 and oS.lableMat[i]*Ei-1<oS.tol) or \
    #         (oS.alphas[i]==oS.C and oS.lableMat[i]*Ei-1>oS.tol):
    if ((oS.lableMat[i] * Ei < -oS.tol and oS.alphas[i] < oS.C) or \
                (oS.lableMat[i] * Ei > oS.tol and oS.alphas[i] > 0)):
        j,Ej = selectJ(oS,i)
        alphaiOld = oS.alphas[i].copy()
        alphajOld = oS.alphas[j].copy()

        if(oS.lableMat[i]!=oS.lableMat[j]):
            L = max(0,oS.lableMat[j]-oS.lableMat[i])
            H = min(oS.C,oS.C+oS.lableMat[j]-oS.lableMat[i])
        else:
            L = max(0,oS.lableMat[i]+oS.lableMat[j]-oS.C)
            H = min(oS.C,oS.lableMat[i]+oS.lableMat[j])
        if(L==H):
            print("L==H")
            return 0
        #分母：K11+K22-2*K11*K22
        #eta = oS.X[i,:]*oS.X[i,:].T+oS.X[j,:]*oS.X[j,:].T-2.0*oS.X[i,:]*oS.X[j,:].T
        #添加核函数
        eta = oS.K[i,i]+oS.K[j,j]-2.0*oS.K[i,j]
        if(eta<=0): print("eta<=0"); return 0

        #更新alpha
        oS.alphas[j] +=oS.lableMat[j]*(Ei-Ej)/eta
        oS.alphas[j]  =clipAlpha(oS.alphas[j],H,L)
        if(abs(oS.alphas[j]-alphajOld)<0.0001):print("j not moving enough");return 0
        updateEk(oS,j)
        oS.alphas[i] +=oS.lableMat[i]*oS.lableMat[j]*(alphajOld-oS.alphas[j])
        updateEk(oS,i)

        #更新b
        b1 =oS.b - Ei - oS.lableMat[i]*(oS.alphas[i]-alphaiOld)*oS.K[i,i] -\
                     oS.lableMat[j]*(oS.alphas[j]-alphajOld)*oS.K[i,j]
        b2 =oS.b - Ej - oS.lableMat[j]*(oS.alphas[i]-alphaiOld)*oS.K[i,j] -\
                     oS.lableMat[j]*(oS.alphas[j]-alphajOld)*oS.K[j,j]

        if(0<oS.alphas[i]) and (oS.alphas[i]<oS.C): b=b1
        elif(0 < oS.alphas[j]) and (oS.alphas[j] < oS.C): b = b2
        else: b = (b1+b2)/2.0

        return 1
    else:
        return 0


def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
    """
    完整的SMO算法
    :param dataMatIn:
    :param classLabels:
    :param C:
    :param toler:
    :param maxIter:
    :param kTup:
    :return:
    """
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler,kTup)
    entireSet = True;
    alphaPairsChanged = 0
    iter =0
    while ((iter<maxIter) and (alphaPairsChanged>0)) or (entireSet):
        alphaPairsChanged = 0
        #全遍历
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged+=innerL(i,oS)
            print("fullSet: iter : %d i :%d,pairs changed: %d",iter,i,alphaPairsChanged)
            iter +=1
        #遍历0<alpha<C
        else:
            #非边界数据ID
            nonBoundIs = nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
            for i in nonBoundIs:
                 alphaPairsChanged +=innerL(i,oS)
                 print("non-bound,iter: %d i:%d pairs changed :%d",iter,i,alphaPairsChanged)
            iter +=1
        if(entireSet):
            entireSet = False
        elif(alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d",iter)
    return oS.b,oS.alphas


def calcws(alphas,dataArr,classLables):
    X = mat(dataArr)
    lableMat = mat(classLables).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*lableMat[i],X[i,:].T)
    print("shape(w): ",shape(w))
    return w

def kernelTrans(X,A,kTup):
    m,_ = shape(X)
    K = zeros((m,1))
    if kTup[0] =='lin':K = X*A.T
    elif  kTup[0] =='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = -deltaRow*deltaRow.T
        K = exp(K/-kTup[1]**2)
    else:print("The kernel is not recognized")
    return K


def test(data_train):
    dataArr, labelArr = loadDataSet(data_train)  # 读取训练数据
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 100, ('lin', 1.3))  # 通过SMO算法得到b和alpha
    print(alphas[alphas>0])
    drawDataMap(dataArr,labelArr,b,alphas)

if __name__=='__main__':
    test("E:\\ML\\SVM\\testSet_FSY.txt")
