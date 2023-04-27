#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 12:22:06 2023

@author: phykc
"""
import cv2 as cv2
from math import pi
import numpy as np
import os as os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import pandas as pd

class FileManager:
    def __init__(self, file, path):
        self.file=file
        self.path=path
        self.filename=os.path.join(self.path, self.file)

class OpenVideo(FileManager):
    def __init__(self, file, path):
        super().__init__(file, path)
        self.frames=[]
       
    def run(self, ri, glass_ri, landa, nthframe, fig, ax, pixelpermicron, startatframe, endframe, timeperframe):
        self.pixelpermicron=pixelpermicron
        self.tpf=timeperframe
        self.nthframe=nthframe
        self.endframe=endframe
        self.cap =cv2.VideoCapture(self.filename)
        self.maxframes=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
        self.maxframes=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.startatframe=startatframe
        self.glass_ri=glass_ri
        print('File: ',self.filename)
        print('Size: ',self.width,' x ',self.height)
        print('Frames: ',self.maxframes)
        framenumber=0
        # self.maxframes=5
        terminate=False
        while framenumber<self.maxframes and not terminate and framenumber<self.endframe:
            print('frame: {}'.format(framenumber))
            #Red each frame ret id True if there is a frame
            ret, frame = self.cap.read()
            #This ends the movie when no frame is present or q is pressed
            if not ret:
                print('End of frames')
                cv2.waitKey(1)
                break
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                cv2.waitKey(1)
                break
            if framenumber<1:
                center,radius=self.find_center(frame)
            if framenumber%self.nthframe==0 and framenumber>=self.startatframe:
                print('Frame ',framenumber,' Analysing')
                self.frames.append(Frame(frame, framenumber, center,radius, ri, self.glass_ri, landa, fig, ax, self.pixelpermicron, self.tpf))
                terminate=self.frames[-1].end
            framenumber+=1
            
            
        cv2.destroyAllWindows()
        cv2.waitKey(10)
       
        
    def find_center(self,frame):
        self.gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        equ = cv2.equalizeHist(self.gray)
        
        self.filter=cv2.bilateralFilter(equ,5,50,50)
        # dst = cv2.Canny(self.filter, 0, 250, None, 3)
        # cv2.imshow('canny',dst)
        circles = cv2.HoughCircles(self.filter, cv2.HOUGH_GRADIENT, 1.8,1000,param1=70,param2=50, minRadius=20, maxRadius=500)
        print(len(circles))
        self.filter1=self.filter.copy()
        self.filter2=self.filter.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv2.circle(self.filter, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv2.circle(self.filter, center, radius, (255, 0, 255), 3)
            cv2.imshow('circles',self.filter)
            cv2.waitKey(100)
            x0,y0=center[0], center[1]
            print(center)
        while True:
            k = cv2.waitKey(100)
            if k ==ord('y'):
                # print('Frame number: ',self.frameno)
                break
            # move up
            elif k==0:
                y0-=1
             
             # move down  
            elif k==1:
                y0+=1
            # move left
            elif k==2:
                x0-=1
            # move right
            elif k==3:
                x0+=1
                
            elif k==ord('o'):
                radius-=5
            
            elif k==ord('p'):
                radius+=5
            center=(x0,y0)
            # you can remove this once keys are setup correctly
            if k!=-1:
                print(k)
            self.filter1=self.filter2.copy()
            cv2.circle(self.filter1, center, 1, (0, 100, 100), 3)
            cv2.circle(self.filter1, center, radius, (255, 0, 255), 3)
            cv2.imshow('moved',self.filter1)
        
        return (int(x0),int(y0)), radius

class Height:
    def __init__(self, fringe_number, ri, glass_ri, landa, dark):
        self.dark=dark
        self.fringenumber=fringe_number
        self.heightlist=[]
        self.ri=ri
        self.glass_ri=glass_ri
        self.landa=landa
        self.delta_h=self.landa/(2*self.ri)
        if self.glass_ri>self.ri:
            if self.dark:
                self.heightlist=[(self.landa*(x+0.5))/(2*self.ri) for x in range(0,self.fringenumber)]
            else:
                self.heightlist=[(self.landa*x)/(2*self.ri) for x in range(0,self.fringenumber)]
        else:
            if self.dark:
                self.heightlist=[(self.landa*(x))/(2*self.ri) for x in range(0,self.fringenumber)]
            else:
                self.heightlist=[(self.landa*(x+0.5))/(2*self.ri) for x in range(0,self.fringenumber)]
            
        self.heightlist.reverse() 
        self.h_array=np.array(self.heightlist)
                 

class Profile(Height):
    def __init__(self, center, rdata, angle, warp_radius, radius, ri, glass_ri, landa,pixelpermicron, dark):
        super().__init__(len(rdata),ri,glass_ri,landa, dark)
        self.pixelpermicron=pixelpermicron
        self.radius=radius
        self.warp_radius=warp_radius
        self.rdata=self.radius*np.array(rdata, dtype=float)/self.warp_radius
        # print(self.rdata)
        self.center=center
        self.angle=angle
        # in radians is easier
        self.angle=self.angle*pi/180
        self.dradius=self.rdata/self.pixelpermicron
        # Add h data array
        self.z=self.h_array.copy()
       
        #determine the pixels in x,y of the image
        self.x=self.rdata*np.cos(self.angle)+self.center[0]
        self.y=self.rdata*np.sin(self.angle)+self.center[1]
        # print(self.x)
        
        self.map=zip(self.x,self.y,self.z)
        if len(self.dradius)>3:
            self.contactangle=np.arctan(self.z[-3]/(self.dradius[-1]-self.dradius[-3]))*180/pi
        else:
            self.contactangle=0

class Frame(Height):
    def __init__(self, frame, frameno, center, radius, ri, glass_ri, landa, fig, ax, pixelpermicron, tpf):
        self.pixelpermicron=pixelpermicron
        self.ax=ax
        self.fig=fig
        self.end=False
        self.frameno=frameno
        self.time=frameno*tpf
        self.frame=frame
        self.profilesdark=[]
        self.profileslight=[]
        self.center=center
        self.radius=radius
        self.collected=True
        self.ri=ri
        self.glass_ri=glass_ri
        self.landa=landa
        self.gray=cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.equ = cv2.equalizeHist(self.gray)
        self.filter=cv2.bilateralFilter(self.equ,15,20,20)
        self.heightmap=np.zeros(self.gray.shape)
        self.warp_radius=1000
        flags = cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR
        self.warp=cv2.warpPolar(self.filter, (self.warp_radius, 3600), self.center,self.radius, flags)
        self.equw = cv2.equalizeHist(self.warp)
        
        # self.filterw=cv2.GaussianBlur(self.equw,(21,21),0)
        self.filterw=cv2.bilateralFilter(self.equw,11,130,130)
        self.filterw_dark=self.filterw.copy()
        self.filterw_light=self.filterw.copy()
        
        self.find_fringes()
        self.plotdata()
        
    def find_fringes(self):
        xdata=np.arange(self.filterw.shape[1])
        astepsize=100
        startangle=0
        endangle=3600
        
        fringetypes=[True, False]
        
        self.radiusposmeandark=[]
        self.radiusposmeanlight=[]
        self.stderrordark=[]
        self.stderrorlight=[]
        self.h_arraydark=[]
        self.h_arraylight=[]
        
        for dark in fringetypes:
            for angle in range(startangle,endangle,astepsize):
                # Take the mean of a slice of the polar warped image
                ydata=np.mean(self.filterw[angle:angle+astepsize,:],axis=0)
                # plt.plot(xdata,ydata)
                # plt.show()
                # ydata=np.flip(ydata)
                
                # Find the 'troughs'
                if dark:
                    peaks=find_peaks(ydata*-1, distance=20,prominence=5)
                else:
                    peaks=find_peaks(ydata, distance=20,prominence=5)
                    
                peakarray=peaks[0]
                peaklist=[]
                # Draw the found fringes and list the peak indicies
                for pks in peakarray:
                    if abs(xdata[pks])>10:
                        peaklist.append(pks)
                        if dark:
                            cv2.line(self.filterw_dark,(int(xdata[pks]),angle),(int(xdata[pks]),angle+astepsize),color=(255,255,0),thickness=3)
                        else:
                            cv2.line(self.filterw_light,(int(xdata[pks]),angle),(int(xdata[pks]),angle+astepsize),color=(255,255,0),thickness=3)
                        
                xdrop=[]
                for pk in peaklist:
                    xdrop.append(xdata[pk])
                
                # create a profile for the angle range as long as there are more than three fringes. If less then it assumes the drop video is ending.
                if dark:
                    self.profilesdark.append(Profile(self.center,xdrop,angle/10,self.warp_radius,self.radius, self.ri, self.glass_ri, self.landa,self.pixelpermicron, dark))
                else:
                    self.profileslight.append(Profile(self.center,xdrop,angle/10,self.warp_radius,self.radius, self.ri, self.glass_ri, self.landa,self.pixelpermicron, dark))
                    
                # else:
                #     print('skipped frame')
                # Show the images.
            
                
                # Process only xth frame
                # create a 2D array of the profiles of zeros, placing the rdata to the right of the array for each line. then take mean along the
                
                profilesizes=[]
                maxnopoints=0
                
                # just need the dark and light profiles and then combine them
                if dark:
                    self.profiles=self.profilesdark
                else:
                    self.profiles=self.profileslight
                    
                for pro in self.profiles:
                    if len(pro.dradius)>maxnopoints:
                        maxnopoints=len(pro.dradius)
                    profilesizes.append(len(pro.dradius))
            
            # Find mean size of the profiles (no. of points)
            
            psizearray=np.array(profilesizes)
            # print('psize',psizearray)
            
            if len(psizearray)>0:
                meansize=int(np.nanmean(psizearray))
                if meansize>3:
                    # meansize-=1
                    # Create an array
                    # print('mean size', meansize)
                    profilesarray=np.zeros((len(self.profiles),maxnopoints))
                    profilesarray[profilesarray==0]=np.nan
                    for i,pro in enumerate(self.profiles):
                        profilesarray[i,maxnopoints-len(pro.dradius):]=pro.dradius
                        
                    radiusposmean=np.nanmean(profilesarray, axis=0)
                    stderrradiuspos=np.nanstd(profilesarray, axis=0, ddof=1)/np.sqrt(np.size(profilesarray, axis=0))
                    # Slice so that only nonzero values are included to avoid distortion.
                    # Use the height class to get the height array
                    super().__init__(len(radiusposmean[maxnopoints-meansize:]), self.ri, self.glass_ri, self.landa, dark)
                    if dark:
                        self.radiusposmeandark=radiusposmean[maxnopoints-meansize:]
                        self.stderrordark=stderrradiuspos[maxnopoints-meansize:]
                        self.h_arraydark=self.h_array.copy()
                    else:
                        self.radiusposmeanlight=radiusposmean[maxnopoints-meansize:]
                        self.stderrorlight=stderrradiuspos[maxnopoints-meansize:]
                        self.h_arraylight=self.h_array.copy()
                    # combine light and dark arrays
                else:
                    self.end=True
        cv2.imshow('warp', self.filterw)
        if dark:
            cv2.imshow('warp dark', self.filterw_dark)
        else:
            cv2.imshow('warp light', self.filterw_light)
            
        cv2.imshow('Frame',self.frame)
        cv2.waitKey(1)    
        
        self.test='hi!'
        self.radiusposmean=np.concatenate((self.radiusposmeandark,self.radiusposmeanlight))
        self.stderror=np.concatenate((self.stderrordark,self.stderrorlight))
        self.h_array=np.concatenate((self.h_arraydark,self.h_arraylight))
        zipped=zip(self.radiusposmean,self.h_array,self.stderror)
        listzip=list(zipped)
        sortedlist=sorted(listzip, key =lambda x:x[0])
        self.completearray=np.array(sortedlist)
        try:
            self.C_A=self.findCA(self.completearray[:,0],self.completearray[:,1])
        except:
            print('not enough data for CA ')
    def findCA(self, xdata, ydata):
        f,v=curve_fit(self.quadfunc,xdata,ydata)
        x1=self.Qsolve(f)
        gradient=-1*self.gradient(x1,f[0], f[1])
        return np.arctan(gradient)*180/pi
    
    def Qsolve(self,f):
        delta=1E-4
        a,b,c =f[0],f[1],f[2]
        d=np.sqrt(b**2-4*a*c)

        if -b+d<delta:
            x2=(-b-d)/(2*a)
            x1=2*c/(-b-d)
        else:
            x2= 2*c/(-b+d)
            x1=(-b+d)/(2*a)
        return max([x1,x2])
    
    def gradient(self,x,a,b):
        return 2*a*x+b
        
    def quadfunc(self,x,a,b,c):
        return a*x**2+b*x+c
        
    def plotdata(self):
                # fit the data and plot it 
        try:
            if len(self.completearray[:,0])>3:
                popt, pcov = curve_fit(self.dropfunc,self.completearray[:,0],self.completearray[:,1],p0=[max(self.radiusposmean),10*pi/180])
                
                self.ax[0].errorbar(self.completearray[:,0],self.completearray[:,1],xerr=self.completearray[:,2],capsize=3,label='t = '+str(self.time),marker='o')
                self.ax[1].errorbar(self.completearray[:,0],self.completearray[:,1],xerr=self.completearray[:,2], marker='o',capsize=3, label='t = '+str(self.time)+' s')
                # plot best fir to spherical droplet
                r_data=np.linspace(-popt[0],popt[0],100)
                self.ax[0].plot(r_data,self.dropfunc(r_data, *popt),linestyle='dotted', color='black',label='best fit')
                
                perr = np.sqrt(np.diag(pcov))
                h0=popt[0]*np.tan(popt[1]/2)
                
                self.volume=pi*h0*(3*popt[0]**2+h0**2)/6
                # print('Volume = ',volume)
                self.CA=popt[1]*180/pi
                self.ax[0].set_xlabel('radial distance /µm')
                self.ax[0].set_ylabel('height distance /µm')
                self.ax[1].set_xlabel('radial distance /µm')
                self.ax[1].set_ylabel('height distance /µm')
            else:
                self.volume=np.nan
                self.CA=np.nan
        except:
            print('no data')
    
        
        
       
        
    def dropfunc(self,r,R, theta):
        return np.sqrt((R**2/np.sin(theta)**2)-r**2)-R/(np.tan(theta))
    def checkcentre(self):
        img=self.filterw.copy()
        dst = cv2.Canny(img, 50, 250, None, 3)
        linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, 80, 30, 10)

        if linesP is not None:
            print(len(linesP))
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(img, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
                        
            cv2.imshow('hough lines',dst)
            cv2.waitKey(10)
                
        cv2.waitKey(1)


class DropAnalysis:
    def __init__(self, file, path, ri, glass_ri, landa, nthframe, pixelpermicron, startatframe, endframe, timeperframe):
        self.file=file
        self.path=path
        vid=OpenVideo(self.file,self.path)
        fig, ax =plt.subplots(2)
        vid.run(ri=1.4, glass_ri=glass_ri, landa=0.450, nthframe=nthframe, fig=fig, ax=ax, pixelpermicron=pixelpermicron, startatframe=startatframe, endframe=endframe, timeperframe=timeperframe)
        plt.show()
        self.tdata=[]
        self.voldata=[]
        self.CAdata=[]
        
        for frame in vid.frames:
            try:    
                self.voldata.append(frame.volume)
                self.CAdata.append(frame.C_A)
                self.tdata.append(frame.time)
            except:

                print('Vol data missing')
        try:        
            plt.scatter(self.tdata,self.voldata)
            plt.xlabel('Time /s')
            plt.ylabel("volume/ cubic µm")
            plt.ylim(0,max(self.voldata))
            plt.show()
            plt.scatter(self.tdata,self.CAdata)
            plt.xlabel('Time /s')
            plt.ylabel("Contact Angle")
            plt.show()
        except:
            print('no data to plot')
        data={'time /s':self.tdata,'Contact angle':self.CAdata, 'Volume':self.voldata}
        self.df1=pd.DataFrame(data=data)
        self.savedata(vid)
    def savedata(self, vid):
        datasets=[]
        for frame in vid.frames:
            if frame.end ==False:

                data={'t = '+str(frame.time)+' r':frame.completearray[:,0],'t = '+str(frame.time)+' r error':frame.completearray[:,2], 't = '+str(frame.time)+' h(r)':frame.completearray[:,1]}
                datasets.append(pd.DataFrame(data))
        
        self.df=pd.concat(datasets, axis=1)
        xlsfilename=self.file[:-4]+'_drop.xlsx'
        newfilename=os.path.join(self.path, xlsfilename)
        print('Data saved',newfilename)
        self.df.to_excel(newfilename)
        xlsfilename1=self.file[:-4]+'_volume_CA.xlsx'
        newfilename1=os.path.join(self.path, xlsfilename1)
        self.df1.to_excel(newfilename1)
        
        

if __name__ =='__main__':
    
    file='20 DegC_400 um_RT_20X_2.mp4'
    path='/Users/phykc/OneDrive - University of Leeds/Min Fu'
    ri=1.5160635
    landa=0.480
    nthframe=50
    pixelpermicron=3.03
    startatframe=0
    endframe=3500
    timeperframe=0.121
    glass_ri=1.51

    DA=DropAnalysis(file, path, ri, glass_ri, landa, nthframe, pixelpermicron, startatframe, endframe, timeperframe)


    
    

