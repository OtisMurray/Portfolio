import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from numpy import array, cov, corrcoef
from math import sqrt
from sklearn import datasets, linear_model

def getdata(filepath):
    martix = np.loadtxt(filepath)
    x= martix[:,0]
    y= martix[:,1]
    return x,y
    
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
    z_max=160
    nbins=100
    global plot
    H, xedges, yedges = np.histogram2d(x,y,bins=nbins)
    H = np.rot90(H)
    H = np.flipud(H)    
    Hmasked = np.ma.masked_where(H==0,H)
    plot = ax.pcolormesh(xedges,yedges,Hmasked,cmap='RdYlBu_r', vmin=z_min, vmax=z_max)    

    xx=[0,500]
    yy=[0,500]
    ax.plot(xx,yy,'k--',linewidth=2.)              
    
    aa = regress(x,y)
    ax.plot(x,x*aa[0]+aa[1],'r',linewidth=1.) 
    k = str('%.2f'%(aa[0]))
    b = str('%.2f'%(aa[1]))
    ax.set_ylim(0, 500)
    ax.set_xlim(0, 500)
    ax.set_xticklabels(('', '', '', '', '','',''),fontname='Calibri',fontsize=18) 
    ax.set_yticklabels(('', '', '', '', '','',''),fontname='Calibri',fontsize=18)
    ax.text(42, 450, r"Y = "+k+"X + "+b, {'color': 'k', 'fontname': 'Calibri', 'fontsize': 18,'fontweight':'bold'})
    ax.text(42, 415, r"R$^2$ = "+correlation(x,y), {'color': 'k', 'fontname': 'Calibri', 'fontsize': 18,'fontweight':'bold'})
    ax.text(42, 380, r"RMSE = "+RMSE(x,y), {'color': 'k', 'fontname': 'Calibri', 'fontsize': 18,'fontweight':'bold'})
    ax.text(42, 345, r"NRMSE = "+NRMSE(x,y), {'color': 'k', 'fontname': 'Calibri', 'fontsize': 18,'fontweight':'bold'})
    ax.text(42, 310, r"MAE = "+MAE(x,y), {'color': 'k', 'fontname': 'Calibri', 'fontsize': 18,'fontweight':'bold'})
    
if __name__ == "__main__": 
    path = r'E:\Figure 1'   
    filepath1 = path + "/Sample_CV-Month.txt"
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
    ax1.set_xticklabels(('0', '100', '200', '300', '400', '500'),fontname='Calibri',fontsize=18)    
    ax1.set_yticklabels(('0', '100', '200', '300', '400', '500'),fontname='Calibri',fontsize=18) 

    ax1.text(250,30, r"(d) Monthly", {'color': 'k', 'fontname': 'Calibri', 'fontsize': 20,'fontweight':'bold'})                                                                                                                                                                                                                                                          
                                                                                                                                                                                                             
    fig1.subplots_adjust(right=0.79)
    cbar_ax = fig1.add_axes([0.81, 0.2, 0.03, 0.6])
    cbar = fig1.colorbar(plot,cax=cbar_ax) 
    cbar.ax.set_yticklabels(('0','','40','','80','','120','','160'),fontname='Calibri',fontsize=18,fontweight='bold')  
                                             
    savepath = path + '\Sample_CV-Month-New.pdf'
    plt.savefig(savepath, dpi =300)           