import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import math
import os

# aplicare HoG pe imagini

# definitie HoG la nivel de imagine
def HOG(img, T, o):
    # o da tipul filtrului: o=0-> sobel, o=1-> -11, o=2-> -101
    l = img.shape[0]
    c = img.shape[1]
    
    # pt a det valoarea care trebuie scazuta la modul general 
    # val = max din cele 2 dimensiuni -1
    Gv = np.zeros([l-2,c-2])
    Gh = np.zeros([l-2,c-2])
    # magnitudini si orientari
    M = np.zeros([l-2,c-2])
    O = np.zeros([l-2,c-2])
    
    eps = 1e-5
    ## aplicare filtre
    if(o==0):
        Gv = -img[0:l-2,0:c-2]-2*img[1:l-1,0:c-2]-img[2:l,0:c-2]+img[0:l-2,2:c]+2*img[1:l-1,2:c]+img[2:l,2:c]
        Gh = -img[0:l-2,0:c-2]+img[2:l,0:c-2]-2*img[0:l-2,1:c-1]+2*img[2:l,1:c-1]-img[0:l-2,2:c]+img[2:l,2:c]
    elif(o==1):
        # se scade in plus o linie/ coloana pt a avea matrice patratica -> 299*299
        Gv = -img[0:l,0:c-1]+img[0:l,1:c]
        Gh = -img[0:l-1,0:c]+img[1:l,0:c]
        # se scad 2 linii/coloane -> 288*288
    elif(o==2):
        Gv = -img[0:l,0:c-2]+img[0:l,2:c]
        Gh = -img[0:l-2,0:c]+img[2:l,0:c]
    
    
    # calcul magnitudini si orientari
    
    for i in range(Gv.shape[0]):
        for j in range(Gv.shape[1]):
            M[i,j] = math.sqrt(np.power(Gv[i,j],2)+np.power(Gh[i,j],2))
            O[i,j] = math.atan(Gv[i,j]/(Gh[i,j]+eps))
    
    #M = np.power((np.power(Gv,2)+np.power(Gh,2)), 0.5)
    
    ## vectorizare magnitudine si orientari
    M = M.flatten()
    O = O.flatten()
    O = O*180/np.pi
    
    #stergere din orientari a valorilor corespunzatoare unor magnitudini maimici decat pragul
    #val intre 0 si 1 pt magn
    O[M<T] = -1
    O=O[O!=-1]

    #aplicare histograma pe orientari
    O = np.round(O)
    rez = np.histogram(O,bins = 9)[0]
    rez = rez/(sum(rez)+eps)
    
    return rez

plt.close("all")  

poze = os.listdir('./Mari')
# masca de ponderare
mat_descriptor = np.zeros([12,441]) # 9 bini * 49 ferestre 
mask = np.ones([7,7])
mask[3:7,0] = 0
mask[3:7,6] = 0
mask[2:4,3] = 0
mask[0:2,0] = 2
mask[0:2,6] = 2
mask[5,3] = 2
mask[1,1:3] = 4
mask[1,4:6] = 4

T=0.05

for k,nume in enumerate(poze):
    l=[]
    img = io.imread('./Mari/'+nume)
    gray = color.rgb2gray(img)
        
    val = math.ceil(img.shape[0]/7)
    for i in range(7):
        for j in range(7):
            fer = gray[i*val:(i+1)*val, j*val:(j+1)*val]
            # hog aplicat pe fiecare fereastra din cele 49
            hog = HOG(fer,T,0)
            # histograma direct in functie, deci doar ponderare
            rez = hog * mask[i,j]
            l.append(rez)
            
    l = np.asarray(l)
    l = l.flatten()
    mat_descriptor[k,:] = l
    
'''
#check to see if it keeps contours
plt.figure()
plt.imshow(Gv, cmap = "gray")

plt.figure()
plt.imshow(Gh, cmap = "gray")
'''
# calcul distante intre descriptori
dist = np.zeros([mat_descriptor.shape[0],mat_descriptor.shape[0]]) 
arrange = np.zeros([mat_descriptor.shape[0],mat_descriptor.shape[0]])

for i in range(mat_descriptor.shape[0]):
    for j in range(mat_descriptor.shape[0]):
        dist[i,j] = math.sqrt(sum((mat_descriptor[i]-mat_descriptor[j])**2))

    # sortare id img in functie de distante
    arrange[i] = np.argsort(dist[i,:])

print(arrange[:,0:3])
