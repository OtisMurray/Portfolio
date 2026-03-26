import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import pandas as pd
from scipy import stats
from numpy import array, cov, corrcoef
from math import sqrt
from sklearn import datasets, linear_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "..", "Data")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "Output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

filepath1 = os.path.join(DATA_DIR, "Figure 1.xlsx")

df = pd.read_excel("Data/Figure 1.xlsx")
print(df.columns)
print(df.dtypes)
print(df.head())


def getdata(filepath1):
    df = pd.read_excel(filepath1)
    x = df.iloc[:, 0].to_numpy()
    y = df.iloc[:, 1].to_numpy()

    return x, y

    
def RMSE(x, y):  
    rmse = np.sqrt(np.mean((x - y) ** 2))
    rmse = round(rmse,2)
    return str('%.2f'%(rmse))  

def NRMSE(x, y):  
    rmse = np.sqrt(np.mean((x - y) ** 2))
    rmse = round(rmse,2)
    nrmse = rmse/np.mean(x)
    return str('%.2f'%(nrmse))
    
def MAE(x, y):  
    mae = np.mean(abs(x - y)) 
    mae = round(mae,2)
    return str('%.2f'%(mae)) 

def MRE(x, y):  
    mpe = np.mean(abs(x - y)/x)*100.0 
    return str('%.2f'%(mpe))  
    
def multipl(a,b):
    sumofab=0.0
    for i in range(len(a)):
        temp=a[i]*b[i]
        sumofab+=temp
    return sumofab      

def correlation(x,y):
    n=len(x)
    sum1=sum(x)
    sum2=sum(y)
    sumofxy=multipl(x,y)
    sumofx2 = sum([pow(i,2) for i in x])
    sumofy2 = sum([pow(j,2) for j in y])
    num=sumofxy-(float(sum1)*float(sum2)/n)
    den=sqrt((sumofx2-float(sum1**2)/n)*(sumofy2-float(sum2**2)/n))   
    r = num/den
    r2 = r*r
    return str('%.2f'%(r2))

def regress(x,y):
    regr = linear_model.LinearRegression()  
    t = np.array(x)
    t = np.array([x]).T  
    regr.fit(t, y) 
    k = regr.coef_
    b = regr.intercept_
    KB = [k,b]
    return KB
              
def plot_data(x,y,ax):    
    z_min=0 
    z_max=200
    nbins=100
    global plot
    H, xedges, yedges = np.histogram2d(x,y,bins=nbins)
    H = np.rot90(H)
    H = np.flipud(H)    
    Hmasked = np.ma.masked_where(H==0,H)
    plot = ax.pcolormesh(xedges,yedges,Hmasked,cmap='RdYlBu_r', vmin=z_min, vmax=z_max)   

    xx=[0,1000]
    yy=[0,1000]
    ax.plot(xx,yy,'k--',linewidth=2.)              
    
    aa = regress(x,y)
    ax.plot(x,x*aa[0]+aa[1],'r',linewidth=1.) 
    k = str('%.2f'%(aa[0]))
    b = str('%.2f'%(aa[1]))
    ax.set_ylim(0, 1000)
    ax.set_xlim(0, 1000)
    ax.set_xticklabels(('', '', '', '', '','',''),fontname='Calibri',fontsize=18) 
    ax.set_yticklabels(('', '', '', '', '','',''),fontname='Calibri',fontsize=18)
    ax.text(84, 900, r"Y = "+k+"X + "+b, {'color': 'k', 'fontname': 'Calibri', 'fontsize': 18,'fontweight':'bold'})
    ax.text(84, 830, r"R$^2$ = "+correlation(x,y), {'color': 'k', 'fontname': 'Calibri', 'fontsize': 18,'fontweight':'bold'})
    ax.text(84, 760, r"RMSE = "+RMSE(x,y), {'color': 'k', 'fontname': 'Calibri', 'fontsize': 18,'fontweight':'bold'})
    ax.text(84, 690, r"NRMSE = "+NRMSE(x,y), {'color': 'k', 'fontname': 'Calibri', 'fontsize': 18,'fontweight':'bold'})
    ax.text(84, 620, r"MAE = "+MAE(x,y), {'color': 'k', 'fontname': 'Calibri', 'fontsize': 18,'fontweight':'bold'})
    
if __name__ == "__main__": 
    path = DATA_DIR
    filepath1 = os.path.join(DATA_DIR, "Figure 1.xlsx")
    x1,y1 = getdata(filepath1)
            
    fig1 = plt.figure(figsize=(5,6))
    G = gridspec.GridSpec(1, 1) 
                       
    ax1 = plt.subplot(G[0, 0]) 
                                   
    plot_data(x1,y1,ax1)
    plt.tight_layout(pad=5, w_pad=1.0, h_pad=1.0)  
 
    aa = regress(x1,y1)
    regr = linear_model.LinearRegression()  
    t = np.array(x1)
    t = np.array([x1]).T  
    regr.fit(t, y1) 

    ax1.set_xlabel(r"Measured PM$_{2.5}$ (μg m$^{-3}$)",fontname='Calibri',fontsize=18,fontweight='bold') 
    ax1.set_xticklabels(('0', '200', '400', '600', '800', '1000'),fontname='Calibri',fontsize=18)    
    ax1.set_ylabel(r"Estimated PM$_{2.5}$ (μg m$^{-3}$)",fontname='Calibri',fontsize=18,fontweight='bold')
    ax1.set_yticklabels(('0', '200', '400', '600', '800', '1000'),fontname='Calibri',fontsize=18)   

    ax1.text(620,60, r"(c) Daily", {'color': 'k', 'fontname': 'Calibri', 'fontsize': 20,'fontweight':'bold'})                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    fig1.subplots_adjust(right=0.79)
    cbar_ax = fig1.add_axes([0.81, 0.2, 0.03, 0.6])
    cbar = fig1.colorbar(plot,cax=cbar_ax) 
    cbar.ax.set_yticklabels(('0','','50','','100','','150','','200'),fontname='Calibri',fontsize=18,fontweight='bold')  
                                             
    savepath = os.path.join(OUTPUT_DIR, "Sample_CV-Day.pdf")
    plt.savefig(savepath, dpi =300, bbox_inches='tight')           