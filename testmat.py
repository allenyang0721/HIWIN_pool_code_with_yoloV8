import math
import numpy as np
import cv2

homo = cv2.imread('00.png')
w=(140,360)
r=(380,430)

holes = [(0, 0), (0, 500), (0, 1000), (500, 0), (500, 500), (500, 1000)]
obs=[(300,400),(400,400)]

#算長度
def getL(a,b):
    xy=math.sqrt((a*a)+(b*b))
    return xy

#算向量
def vector(a,b):
    x1=b[0]
    x2=a[0]
    y1=b[1]
    y2=a[1]
    x=x1-x2
    y=y1-y2
    vct=[x,y]
    return vct

#假想球座標
def fakeball(b,c):
    Vb=vector(b,c)
    Lb=getL(Vb[0],Vb[1])
    a=2*28/Lb
    F=[b[0]-(Vb[0]*a),b[1]-(Vb[1]*a)]
    G=(int(F[0]),int(F[1]))
    return G

#算角度
def getA(a,b,c):
    F1,F2=vector(a,b)
    S1,S2=vector(b,c)
    AB=np.array(b)-np.array(a)
    BC=np.array(c)-np.array(b)
    cos=(np.dot(AB,BC))/(getL(F1,F2)*getL(S1,S2))
    theta=math.acos(cos)*(180/math.pi)
    return  theta

#畫點
def DOT(a):
    cv2.circle(homo,a,5,(200,0,0),-1)

#碰撞
def crush(a,b,c):
    #adot bdot c=ball y=ax+b
    ass=0
    m=(b[1]-a[1])/(b[0]-a[0])
    y=a[1]
    x=a[0]
    z=(b[1]-m*b[0])
    (m*x)-y+z==0
    for i in range(len(obs)):
        gay=abs(m*c[i-1][0]-c[i-1][1]+z)
        ans=gay/getL(m,-1)
    
        if 0<=ans<=56:
           
            
            if x<=c[i-1][0]<=b[0] or y<=c[i-1][1]<=b[1]:
                ass=ass+1
                
    if ass>0:
        return True
    else:
        return False        
    
#畫反彈點
def rebound(a,b):#假想，母
    p1=((b[0]+a[0])/2,28)
    p2=((b[0]+a[0])/2,972)
    p3=(28,(b[1]+a[1])/2)
    p4=(472,(b[1]+a[1])/2)
    int(p1[0])
    int(p1[1])
    int(p2[0])
    int(p2[1])
    int(p3[0])
    int(p3[1])
    int(p4[0])
    int(p4[1])
    p1to4=[p1,p2,p3,p4]
    return p1to4
    
    
#反彈點與假想球
def rtof(a,b,c,d):#母，子，袋口，反彈點
    anal=[]

    for i in range(3):
        for j in range(5):
            x=d[i]
            y=c[j]
            if crush(a,x,obs)==False:#母球到反彈
                if crush(x,fakeball(b,y),obs)==False:#反彈到洞口
                    anal.append((fakeball(b, y), obs)) #都沒有阻礙就把兩個座標回傳
                    if not anal:
                        return [(0,0),(0,0)]
                    else:
                        return anal #anal[0]=[(x1,y1),(x2,y2)] anal[0][0]=(x1,y1) anal[0][0][0]=x1

#反彈點與假想球
def rto(mom,kid,pocket,rebound,obs):#母，子，袋口，反彈點，障礙物
    ret=[]
    for k in range(len(obs)):#障礙loop
        for i in range(5):
            for j in range(3):
                if crush(mom,rebound(fakeball(pocket[i],kid),mom)[j],obs[k])== True:#母到反
                    if crush(rebound(fakeball(pocket[i],kid))[j],fakeball(pocket[i],kid),obs[k])==True:#反到假
                        if crush(fakeball(pocket[i],kid),pocket[i],obs)==True:#假到袋
                            ret.append(rebound(fakeball(pocket[i],kid),mom)[j],fakeball(pocket[i],kid),pocket[i])#反to假to袋
    if not ret:
        return [(0,0),(0,0),(0,0)]
    else:
        return ret
                    
#反彈點
def analreb(mom,kid,hole):
    fb_list=[]
    for i in range(5):
        fb_list.append(fakeball(kid,hole[i]))
    reb_list=[]
    for i in range(5):
        for j in range(3):
            reb_list.append(rebound(fb_list[i],mom)[j])
    print(reb_list)
    return reb_list

    


#設定turnL90為homo左轉90度
turnL90=cv2.rotate(homo,cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imshow('poolball',turnL90)
cv2.waitKey(0)




    

    





