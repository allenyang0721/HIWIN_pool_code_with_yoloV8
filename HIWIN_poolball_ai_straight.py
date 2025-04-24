import math
import numpy as np
import cv2

#-------⚡手動輸入區(圖片 母球 子球)⚡---------
homo = cv2.imread('pool_data/case2.png') # 00空桌,01直擊,02反彈,03組合
wball = (250,750) #母 120 740
rball = (250,125) #子 310 810
obs=[(28,250),(250,250)]#障礙 400,925

#-----⚡函式區⚡-------

#算長度
def getLong(a,b):
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
    Vector_b=vector(b,c)
    Long_b=getLong(Vector_b[0],Vector_b[1])
    a=2*28/Long_b
    F=[b[0]-(Vector_b[0]*a),b[1]-(Vector_b[1]*a)]
    G=(int(F[0]),int(F[1]))
    return G

#算角度
def getAngle(a,b,c):
    F1,F2=vector(a,b)
    S1,S2=vector(b,c)
    AB=np.array(b)-np.array(a)
    BC=np.array(c)-np.array(b)
    cos=(np.dot(AB,BC))/(getLong(F1,F2)*getLong(S1,S2))
    theta=math.acos(cos)*(180/math.pi)

    return  theta

#畫點
def DOT(a):
    cv2.circle(homo,a,5,(200,0,0),-1)
#畫線
def line(a,b):
    cv2.line(homo, a,b, (150, 70,130), 3)
#碰撞
def crush(a,b,c):
    #adot bdot c=ball y=ax+b
    ass=0
    y=a[1]
    x=a[0]
    if b[0]==a[0]:
        for i in range(len(c)):
            ans=abs(a[1]-c[i][1])
            
    else:
        m=(b[1]-a[1])/(b[0]-a[0])
    
        z=(b[1]-m*b[0])
        (m*x)-y+z==0
        for i in range(len(c)):
            
            gay=abs(m*(c[i][0])-(c[i][1])+z)
            ans=gay/getLong(m,-1)
            #print("距離",str(ans))

            if 0<=ans<=57:
                
                #ass=ass+1
                bigx=max(x-28,b[0]+28)
                smallx=min(x-28,b[0]+28)
                bigy=max(y-28,b[1]+28)
                smally=min(y-28,b[1]+28)
                if smallx<=c[i][0]<=bigx or smally<=c[i][1]<=bigy:
                    ass=ass+1
                    
            
    if ass>0:
        return True
    else:
        return False        


#碰撞單球體
def crush_one(a,b,c):
    #adot bdot c=ball y=ax+b
    ass=0
    m=(b[1]-a[1])/(b[0]-a[0])
    y=a[1]
    x=a[0]
    z=(b[1]-m*b[0])
    (m*x)-y+z==0
    gay=abs(m*(c[0])-(c[1])+z)
    ans=gay/getLong(m,-1)

    if 0<=ans<=28:
        if x<c[0]<b[0] or y<c[1]<b[1]:
            ass=ass+1
    if ass>0:
        return True
    else:
        return False        

#畫反彈點
def rebound(a,b):#假想，母
    p1=((b[0]+a[0])/2,28)#|
    p2=((b[0]+a[0])/2,972)# |
    p3=(28,(b[1]+a[1])/2)#_
    p4=(472,(b[1]+a[1])/2)#-
    p1to4=[p1,p2,p3,p4]
    return p1to4
    
#反彈點
def rebpoint_list(mom,kid,hole):
    fb_list=[]
    for i in range(6):
        fb_list.append(fakeball(kid,hole[i]))
    reb_list=[]
    for i in range(6):
        
        for j in range(4):
            reb_list.append(rebound(fb_list[i],mom)[j])
    return reb_list

#假球list
def fakeball_list(kid,hole):
    fb_list=[]
    for i in range(4):
        i
        for j in range(6):
            fb_list.append(fakeball(kid,hole[j]))
    return fb_list



#-----⚡常用變數區⚡-------

#洞口位置
holes =  [(0, 0), (0, 500), (0, 1000), (500, 0), (500, 500), (500, 1000)]

#向量
Vwr=vector(wball,rball)

#母球to子球距離
W_R=(getLong(Vwr[0],Vwr[1]))

#-------⚡洞口座標判斷(直擊)⚡-------
hol=[]
for i in range(6):
    angle=abs(getAngle(wball,rball,holes[i]))
    #print(str(angle))
    if angle is not None and 0<=angle<=90:
        #print(str(angle))
        hol.append((holes[i]))
        
#⚡-----⚡⚡⚡執行區⚡⚡⚡-----⚡
reb = False
mix = False
#直球判斷
print("可選的直擊袋口",str(hol))
if hol:
    h=min(hol)
    print(crush(wball,fakeball(rball,h),obs))
    print(crush(fakeball(rball,h),h,obs))
    if crush(wball,fakeball(rball,h),obs)==False:
        if crush(fakeball(rball,h),h,obs)==False:
            straight=True
            print("可直擊")#回報是否能直接擊打
        else:
            straight=False
            mix=True #組合
            print("組合")
    else:
        straight=False
        reb=True #反彈
        print("反彈")

    #畫點,圓
    DOT(wball)
    DOT(rball)
    for i in range(len(obs)):
        DOT(obs[i])
    for i in range(5):
        DOT(holes[i])

    if straight==True:
        for i in range(len(hol)):
            cv2.line(homo, hol[i], fakeball(rball,hol[i]), (150, 170,30), 3)  # 洞口>>假想球 draw line
            cv2.line(homo, wball, fakeball(rball,hol[i]), (150, 170,30), 3)  # 白球>>假想球 draw line
            cv2.circle(homo,fakeball(rball,hol[i]),28,(200,100,0),3) #畫假想圓
            DOT(fakeball(rball,hol[i]))#畫假想圓心
            
    elif straight==False:
        print("不可直接擊打")

    else:print("error!!!\n你三小拉")

else:
    print("沒有可供擊打的角度，重新填數值試試?")

#反彈
if reb==True:
    joker=[]
    rebf=(fakeball_list(rball,holes))
    rebb=(rebpoint_list(wball,rball,holes))
    for h in range(5):
        hnh=holes[h]       
        for r in range(24):
            ana=rebf[r]
            bnb=rebb[r]          
            joker.append((ana,bnb,hnh))  
    analpp=[]
    for i in range(len(joker)):
        
        if crush(wball,joker[i][1],obs)==False:
            if crush(joker[i][1],joker[i][0],obs)==False:
                if crush(joker[i][0],joker[i][2],obs)==False:
                    if crush_one(joker[i][0],joker[i][1],rball)==False:
                         if crush(rball,joker[i][2],obs)==False:
                            if getAngle((joker[i][0]),rball,joker[i][2])<=10:
                                #if getAngle((joker[i][1]),(joker[i][0]),joker[i][2])<=90:
                                 #print(str(getA((joker[i][0]),rball,joker[i][2])))
                                 analpp.append(joker[i])
                                 print(joker[i])
                            
    #print(str(analpp))
    if len(analpp)!=0:
        len_of_ap=len(analpp)
        
        for i in range(len_of_ap):
            #i=0
            cv2.circle(homo,((round(analpp[i][0][0])),round(analpp[i][0][1])),28,(200,100,0),3)#反彈假想
            DOT(((round(analpp[i][0][0])),round(analpp[i][0][1])))
            DOT(((round(analpp[i][1][0])),round(analpp[i][1][1])))
            line(((round(analpp[i][0][0])),round(analpp[i][0][1])),((round(analpp[i][1][0])),round(analpp[i][1][1])))
            line(((rball[0],rball[1])),((round(analpp[i][0][0])),round(analpp[i][0][1])))
            line(((round(analpp[i][2][0])),round(analpp[i][2][1])),(((rball[0],rball[1]))))
            line(((wball[0],wball[1])),((round(analpp[i][1][0])),round(analpp[i][1][1])))
    else:
        print("沒有適合打擊的角度，重設座標")

if mix==True:
    #組合球 單球
    mixf1=fakeball(obs[0],h)
    mixf2=fakeball(rball,mixf1)
    cv2.circle(homo,mixf1,28,(200,100,0),3)
    cv2.circle(homo,mixf2,28,(200,100,0),3)
    DOT(mixf2)
    DOT(mixf1)
    line(wball,mixf2)
    line(mixf2,rball)
    line(rball,mixf1)
    line(mixf1,obs[0])
    line(obs[0],h)
print(fakeball_list(rball,holes))
cv2.circle(homo,wball,28,(200,255,200),3)
cv2.circle(homo,rball,28,(0,0,255),3)
for i in range(len(obs)):
    cv2.circle(homo,obs[i],28,(255,0,255),3)

                 
#設定turnL90為homo左轉90度.
turnL90=cv2.rotate(homo,cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imshow('poolball',turnL90)
cv2.waitKey(0)