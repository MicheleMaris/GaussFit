__DESCRIPTION__="""2d Gauss Fit of a single gaussian
The model function would be

R=[
[ cos(psi_ell),sin(psi_ell)]
[-sin(psi_ell),cos(psi_ell)]
]

C=diag(lambda0,lambda1)

R0 = (x0;y0)

is
log(y)=-0.5*(A*x**2+2*B*x*y+C*y**2+K+E*x+F*y)

the fwhm of two axis comes from eigenvectors of matrix
AA=[[A,B],[B,C]]

the center x0,y0 from 

(x0;y0)=inv(AA)*(E;F)

the zero point

"""

class MapProfile :
   """ Handles the profile of a GRD Map """
   def __init__(self,psi_deg_array,radius_array) :
      """MPF=MapProfile(psi_deg_array,radius_array)
         psi_deg_array=list of angles along wich to compute profiles
         radius_array=list of radii to sample the profiles
      """
      import copy
      from collections import OrderedDict
      import numpy as np
      self.M=OrderedDict()
      self.psi_deg=copy.deepcopy(psi_deg_array)
      self.psi=np.deg2rad(self.psi_deg)
      self.radius=copy.deepcopy(radius_array)
      self.M['_x']=self._template()
      self.M['_y']=self._template()
      self.M['_cos_psi']=self._template()
      self.M['_sin_psi']=self._template()
      for i in range(len(self.psi)) :
         self.M['_x'][i]=self.radius*np.cos(self.psi[i])
         self.M['_y'][i]=self.radius*np.sin(self.psi[i])
         self.M['_cos_psi'][i]=np.cos(self.psi[i])
         self.M['_sin_psi'][i]=np.sin(self.psi[i])
   def _template(self,dtype='float'):
      import numpy as np
      return np.zeros([len(self.psi),len(self.radius)],dtype=dtype)
   def __getitem__(self,this) :
      try :
         return self.M[this]
      except :
         return None
   def __setitem__(self,this,that) :
      self.M[this]=that
   def keys() :
      return self.M.keys()
   def fill(self,name,GRDMap,argument) :
      "extracts profiles of a map of given argument along a list of directions"
      self.M[name]=self._template()
      for ipsi in range(len(self.psi)) :
         self.M[name][ipsi]=GRD.bilinearXY(argument,self.M['_x'][ipsi],self.M['_y'][ipsi])
   def fwhm(self,name,returnItp=False,threshold=0.5,returnStats=True) :
      """extracts the fwhm (beware it assumes profiles can be sorted)
         if returnItp=True (default False) returns also the value of the profile at the threshold point
         if returnStats=True (default False) returns statistics as:
            min, max, sqrt(min*max), sqrt(max/min),mean,rotation,amplitude
      """
      import numpy as np
      pp=np.zeros(len(self.psi))
      tt=np.zeros(len(self.psi))
      for ipsi in range(len(self.psi)) :
         yv=self[name][ipsi]
         idx=np.argsort(yv)
         yv=self[name][ipsi][idx]
         xv=self.radius[idx]
         pp[ipsi]=np.interp(threshold,yv,xv)
         tt[ipsi]=np.interp(pp[ipsi],self.radius,self[name][ipsi])
      pp=2*pp
      if returnStats :
         Min=pp.min()
         Max=pp.max()
         A=(np.cos(self.psi)*(pp-pp.mean())).sum()
         B=(np.sin(self.psi)*(pp-pp.mean())).sum()
         return [Min,Max,(Min*Max)**0.5,(Max/Min)**0.5,pp.mean(),np.rad2deg(np.arctan2(A,B)),(A**2+B**2)**0.5]
      if returnItp : 
         return pp,tt
      return pp

class NoCentNoZer :
   def __init__(self,U,V,Y,Yth=None,doNotScaleAxis=True) :
      import numpy as np
      import numpy.linalg as linalg
      if Yth == None :
         self.YTh = 10**(-0.3)
      else :
         self.YTh = Yth*1
      self.YTh
      self.peak=Y.max()
      lmap=Y/self.peak
      lmap.shape=lmap.size
      idx = np.where(lmap>=self.YTh)[0]
      lmap = np.log(lmap[idx])
      u=U*1
      v=V*1
      u.shape=u.size
      v.shape=v.size
      if doNotScaleAxis :
         self.uscal=1.
         self.vscal=1.
      else :
         self.uscal=u.max()-u.min()
         self.vscal=v.max()-v.min()
      u=u[idx]/self.uscal
      v=v[idx]/self.vscal
      self.n=len(idx)
      self.S=np.zeros([3,3]) 
      self.VV=np.zeros(3) 
      self.S[0,0]= 0.25*(u**4).sum()
      self.S[0,1]= 0.5*((u**3)*v).sum()
      self.S[0,2]= 0.25*(u**2*v**2).sum()
      self.S[1,1]=(u**2*v**2).sum()
      self.S[1,2]=0.5*(u*v**3).sum()
      self.S[2,2]=0.25*(v**4).sum()
      self.S[2,3]=-0.5*(v**2).sum()
      for r in range(len(self.VV)) :
         for c in range(len(self.VV)) :
            if r > c :
               self.S[r,c]=self.S[c,r]*1
      self.VV[0] = -0.5*(lmap*u**2).sum()
      self.VV[1] = -(lmap*v*u).sum()
      self.VV[2] = -0.5*(lmap*v**2).sum()
      self.pars=np.dot(linalg.inv(self.S),self.VV)
      self.inv=linalg.inv(self.S)
      self.det=linalg.det(self.S)
      self.y=lmap
      self.u=u
      self.v=v
      self.res = (self.pars[0]*self.u**2+2*self.pars[1]*self.u*self.v+self.pars[2]*self.v**2)
      self.ksq = self.res**2
      self.A=np.zeros([2,2])
      self.A[0][0]=self.pars[0]/self.uscal**2
      self.A[0][1]=self.pars[1]/self.uscal/self.vscal
      self.A[1][0]=self.pars[1]/self.uscal/self.vscal
      self.A[1][1]=self.pars[2]/self.vscal**2
      self.heighen_val,hv=linalg.eigh(self.A)
      self.semiaxis_fwhm=2.*np.sqrt(2.*(np.log(2.)-self.pars[3])/self.heighen_val)*180./np.pi
      self.fwhm=(self.semiaxis_fwhm[0]*self.semiaxis_fwhm[1])**0.5
      self.ellipticity=self.semiaxis_fwhm.max()/self.semiaxis_fwhm.min()
      self.rot=np.transpose(hv/linalg.det(hv))
      self.psi_ell=np.arctan2(self.rot[0][1],self.rot[0][0])*180./np.pi
      self.gauss_peak=self.peak
   def mdl(self,U,V) :
      acc = self.pars[0]*(U/self.uscal)**2
      acc += 2*self.pars[1]*self.pars[2]*(U/self.uscal)*(V/self.vscal)
      acc += self.pars[2]*(V/self.vscal)**2
      return -0.5*acc+self.pars[3]


class NoCent :
   def __init__(self,U,V,Y,YTh=None,doNotScaleAxis=True,allowed_radius_deg=None) :
      import numpy as np
      import numpy.linalg as linalg
      if YTh == None :
         self.YTh = 1e-3
      else :
         self.YTh = YTh*1
      self.peak=Y.max()
      lmap=Y/self.peak
      lmap.shape=lmap.size
      #
      radius=np.rad2deg((U**2+V**2)**0.5)
      radius.shape=radius.size
      if allowed_radius_deg == None :
         idx = np.where(lmap>=self.YTh)[0]
         print "Select by YTH",len(idx),lmap[idx].min(),lmap.max()
         self.allowed_radius=radius[idx].max()
      else :
         idx = np.where(radius<=allowed_radius_deg)[0]
         self.allowed_radius=allowed_radius_deg
         self.YTh=lmap[idx].min()
      #
      lmap = np.log(lmap[idx])
      u=U*1
      v=V*1
      u.shape=u.size
      v.shape=v.size
      if doNotScaleAxis :
         self.uscal=1.
         self.vscal=1.
      else :
         self.uscal=u.max()-u.min()
         self.vscal=v.max()-v.min()
      u=u[idx]/self.uscal
      v=v[idx]/self.uscal
      self.n=len(idx)
      self.N=len(idx)
      self.S=np.zeros([4,4]) 
      self.VV=np.zeros(4) 
      self.S[0,0]= 0.25*(u**4).sum()
      self.S[0,1]= 0.5*((u**3)*v).sum()
      self.S[0,2]= 0.25*(u**2*v**2).sum()
      self.S[0,3]= -0.5*(u**2).sum()
      self.S[1,1]=(u**2*v**2).sum()
      self.S[1,2]=0.5*(u*v**3).sum()
      self.S[1,3]=-(u*v).sum()
      self.S[2,2]=0.25*(v**4).sum()
      self.S[2,3]=-0.5*(v**2).sum()
      self.S[3,3]=float(len(idx))
      for r in range(len(self.VV)) :
         for c in range(len(self.VV)) :
            if r > c :
               self.S[r,c]=self.S[c,r]*1
      self.VV[0] = -0.5*(lmap*u**2).sum()
      self.VV[1] = -(lmap*v*u).sum()
      self.VV[2] = -0.5*(lmap*v**2).sum()
      self.VV[3] = (lmap).sum()
      self.pars=np.dot(linalg.inv(self.S),self.VV)
      self.inv=linalg.inv(self.S)
      self.det=linalg.det(self.S)
      self.scaled_data=lmap
      self.u=u
      self.v=v
      self.bf_model=-0.5*(self.pars[0]*self.u**2+2*self.pars[1]*self.u*self.v+self.pars[2]*self.v**2)+self.pars[3]
      self.res = np.exp(self.bf_model)-np.exp(self.scaled_data)
      self.ksq = (self.res**2).sum()
      self.A=np.zeros([2,2])
      self.A[0][0]=self.pars[0]/self.uscal**2
      self.A[0][1]=self.pars[1]/self.uscal/self.vscal
      self.A[1][0]=self.pars[1]/self.uscal/self.vscal
      self.A[1][1]=self.pars[2]/self.vscal**2
      #removes the regularizzation
      self.Pars={}
      self.Pars['A']=self.pars[0]/self.uscal**2
      self.Pars['B']=self.pars[1]/self.uscal/self.vscal
      self.Pars['C']=self.pars[2]/self.vscal**2
      self.Pars['D']=np.nan
      self.Pars['E']=np.nan
      self.Pars['F']=self.pars[3]
      #
      self.R0=np.zeros(2)
      self.heighen_val,hv=linalg.eigh(self.A)
      self.semiaxis_fwhm=2.*np.sqrt(2.*(np.log(2.)-self.pars[3])/self.heighen_val)*180./np.pi
      self.fwhm_min=self.semiaxis_fwhm.min()
      self.fwhm_max=self.semiaxis_fwhm.max()
      self.fwhm=(self.semiaxis_fwhm.prod())**0.5
      self.ellipticity=self.fwhm_max/self.fwhm_min
      self.rot=np.transpose(hv/linalg.det(hv))
      self.psi_ell=np.arctan2(self.rot[0][1],self.rot[0][0])*180./np.pi
      self.zero=self.pars[3]*1
      self.gauss_peak=self.peak*np.exp(self.zero)
      self.DataTh=self.YTh
   def mdl(self,U,V) :
      acc = self.pars[0]*(U/self.uscal)**2
      acc += 2*self.pars[1]*self.pars[2]*(U/self.uscal)*(V/self.vscal)
      acc += self.pars[2]*(V/self.vscal)**2
      return -0.5*acc+self.zero
   def __str__(self) :
      l=[]
      l.append("N              : "+str(self.n))
      #l.append("allowed_radius : "+str(self.allowed_radius))
      #l.append("xscal          : "+str(self.xscal))
      #l.append("yscal          : "+str(self.yscal))
      #l.append("               : ")
      l.append("peak           : "+str(self.gauss_peak))
      l.append("fwhm           : "+str(self.fwhm))
      #l.append("fwhm_min       : "+str(self.fwhm_min))
      #l.append("fwhm_max       : "+str(self.fwhm_max))
      l.append("ellipticity    :"+str(self.ellipticity))
      l.append("psi_ell        :"+str(self.psi_ell))
      return "\n".join(l)

class NoBackground_Deprecated :
   def __init__(self,U,V,Y,Yth=None) :
      import numpy as np
      import numpy.linalg as linalg
      if Yth == None :
         self.YTh = 10**(-0.3)
      else :
         self.YTh = Yth*1
      self.YTh
      self.peak=Y.max()
      lmap=Y/self.peak
      lmap.shape=lmap.size
      idx = np.where(lmap>=self.YTh)[0]
      lmap = np.log(lmap[idx])
      u=U*1
      v=V*1
      u.shape=u.size
      v.shape=v.size
      self.uscal=u.max()-u.min()
      self.vscal=v.max()-v.min()
      u=u[idx]/self.uscal
      v=v[idx]/self.uscal
      self.n=len(idx)
      self.S=np.zeros([6,6]) 
      self.VV=np.zeros(6) 
      self.S[0,0]= 0.25*(u**4).sum()
      self.S[0,1]= 0.5*((u**3)*v).sum()
      self.S[0,2]= 0.25*(u**2*v**2).sum()
      self.S[0,3]= -0.5*(u**2).sum()
      self.S[0,4]= -0.5*(u**3).sum()
      self.S[0,5]= -0.5*(v*u**2).sum()
      self.S[1,1]=(u**2*v**2).sum()
      self.S[1,2]=0.5*(u*v**3).sum()
      self.S[1,3]=-(u*v).sum()
      self.S[1,4]= -(v*u**2).sum()
      self.S[1,5]= -(u*v**2).sum()
      self.S[2,2]=0.25*(v**4).sum()
      self.S[2,3]=-0.5*(v**2).sum()
      self.S[2,4]= -0.5*(u*v**2).sum()
      self.S[2,5]= -0.5*(v**4).sum()
      self.S[3,3]=float(len(idx))
      self.S[3,4]= -0.5*(u).sum()
      self.S[3,5]= -0.5*(v).sum()
      self.S[4,4]= -0.5*(u**2).sum()
      self.S[4,5]= -0.5*(u*v).sum()
      self.S[5,5]= -0.5*(v**2).sum()
      for r in range(len(self.VV)) :
         for c in range(len(self.VV)) :
            if r > c :
               self.S[r,c]=self.S[c,r]*1
      self.VV[0] = -0.5*(lmap*u**2).sum()
      self.VV[1] = -(lmap*v*u).sum()
      self.VV[2] = -0.5*(lmap*v**2).sum()
      self.VV[3] = (lmap).sum()
      self.VV[4] = -0.5*(lmap*u).sum()
      self.VV[5] = -0.5*(lmap*v).sum()
      self.pars=np.dot(linalg.inv(self.S),self.VV)
      self.inv=linalg.inv(self.S)
      self.det=linalg.det(self.S)
      self.y=lmap
      self.u=u
      self.v=v
      self.res = (self.pars[0]*self.u**2+2*self.pars[1]*self.u*self.v+self.pars[2]*self.v**2+self.pars[3])
      self.ksq = self.res**2
      self.A=np.zeros([2,2])
      self.A[0][0]=self.pars[0]/self.uscal**2
      self.A[0][1]=self.pars[1]/self.uscal/self.vscal
      self.A[1][0]=self.pars[1]/self.uscal/self.vscal
      self.A[1][1]=self.pars[2]/self.vscal**2
      self.Vde=np.zeros(2)
      self.Vde[0]=self.pars[4]/self.uscal
      self.Vde[1]=self.pars[5]/self.vscal
      self.R0=np.arcsin(np.dot(linalg.inv(self.A),self.Vde))*180./np.pi*3600.
      self.heighen_val,hv=linalg.eigh(self.A)
      self.semiaxis_fwhm=2.*np.sqrt(2.*(np.log(2.)-self.pars[3])/self.heighen_val)*180./np.pi
      self.fwhm=(self.semiaxis_fwhm[0]*self.semiaxis_fwhm[1])**0.5
      self.ellipticity=self.semiaxis_fwhm.max()/self.semiaxis_fwhm.min()
      self.rot=np.transpose(hv/linalg.det(hv))
      self.psi_ell=np.arctan2(self.rot[0][1],self.rot[0][0])*180./np.pi
      a=np.dot(linalg.inv(self.A),self.Vde)
      self.zero=self.pars[3]+0.5*(self.A[0][0]*a[0]*a[0]+2.*a[0]*a[1]*self.A[0][1]+a[1]*a[1]*self.A[1][1])
      self.gauss_peak=self.peak*np.exp(self.zero)
   def mdl(self,U,V) :
      acc = self.pars[0]*(U/self.uscal)**2
      acc += 2*self.pars[1]*self.pars[2]*(U/self.uscal)*(V/self.vscal)
      acc += self.pars[2]*(V/self.vscal)**2
      return -0.5*acc+self.pars[3]
 

class Model :
   def __init__(self,xmin,xmax,nx,ymin,ymax,ny) :
      import numpy as np
      a=np.linspace(xmin,xmax,nx)
      self.dX=a[1]-a[0]
      self.X=np.tile(np.linspace(xmin,xmax,nx),(ny,1))
      self.Y=np.transpose(np.tile(np.linspace(ymin,ymax,ny),(nx,1)))
      a=np.linspace(ymin,ymax,ny)
      self.dY=a[1]-a[0]
      self.R=None
      self.D=None
      self.fwhm = None
      self.fwhm_min = None
      self.fwhm_max = None
      self.gauss_peak = None
      self.ellipticity = None
      self.psi_ell = None
      self.peak = None
      self.R = None
   def __str__(self) :
      if self.D==None : 
         return ''
      l=[]
      l.append("gauss_peak     : "+str(self.gauss_peak))
      l.append("fwhm           : "+str(self.fwhm))
      l.append("fwhm_min       : "+str(self.fwhm_min))
      l.append("fwhm_max       : "+str(self.fwhm_max))
      l.append("ellipticity    :"+str(self.ellipticity))
      l.append("psi_ell        :"+str(self.psi_ell))
      l.append("X0             :"+str(self.R0[0]))
      l.append("Y0             :"+str(self.R0[1]))
      return "\n".join(l)
   def __call__(self,*arg,**karg) :
      """call(NoBackground_Base)
         call(peak,x0,y0,psi_ell,fwhm,ellipticity,MinMax=False)
         MinMax = False : fwhm=p1, ellipticity=p2, fwhm_min and fwhm_max are derived
         MinMax = True  : fwhm_min=p1, fwhm_max=p2, fwhm and ellipticity are derived
      """
      import numpy as np
      if len(arg) == 0 :
         return
      elif len(arg) == 1 :
         try :
            self.gauss_peak=arg[0].gauss_peak
            self.R0=arg[0].R0
            self.psi_ell=arg[0].psi_ell
            self.fwhm=arg[0].fwhm
            self.fwhm_min=arg[0].fwhm_min
            self.fwhm_max=arg[0].fwhm_max
            self.ellipticity=arg[0].ellipticity
         except :
            return
      else :
         MinMax=False
         try :
            MinMax=karg['MinMax']==True
         except :
            MinMax=False
         self.gauss_peak=float(arg[0])
         self.R0=np.zeros(2)
         self.R0[0]=float(arg[1])
         self.R0[1]=float(arg[2])
         self.psi_ell=float(arg[3])
         if MinMax :
            self.fwhm_min=float(arg[4])
            self.fwhm_max=float(arg[5])
            self.fwhm = (self.fwhm_min*self.fwhm_max)**0.5
            self.ellipticity = self.fwhm_max/self.fwhm_min
         else :
            self.fwhm=float(arg[4])
            self.ellipticity=float(arg[5])
            self.fwhm_min=self.fwhm/self.ellipticity**0.5
            self.fwhm_max=self.fwhm*self.ellipticity**0.5
      self.mdl()
   def mdl(self) :
      import numpy as np
      x=self.X-self.R0[0]
      y=self.Y-self.R0[1]
      self.R=(x**2+y**2)**0.5
      cp=np.cos(self.psi_ell/180.*np.pi)
      sp=np.sin(self.psi_ell/180.*np.pi)
      u=(cp*x-sp*y)**2/self.fwhm_max**2
      u+=(sp*x+cp*y)**2/self.fwhm_min**2
      u*=-8.*np.log(2.)/2.
      self.D=self.gauss_peak*np.exp(u)
   def imshow(self) :
      try :
         from matplotlib import pyplot as plt
      except :
         return
      import numpy as np
      plt.imshow(self.D,origin='lower')
      plt.colorbar()
      
class NoBackground_Base :
   def __init__(self,*arg,**karg) :
      "NoBackground_Base(X,Y,D,DataTh=-np.inf,AllowedRadius=np.inf,Weight=None)"
      import numpy as np
      import numpy.linalg as linalg
      if len(arg) < 3 :
         return
      try :
         doNotScaleAxis=float(karg['doNotScaleAxis'])
      except :
         doNotScaleAxis=True
      try :
         self.DataTh=float(karg['DataTh'])
      except :
         self.DataTh=-np.inf
      try :
         self.AllowedRadius=float(karg['AllowedRadius'])
      except :
         self.AllowedRadius = np.inf
      self.peak=arg[2].max()
      # data are regularized
      lmap=arg[2]/self.peak
      lmap.shape=lmap.size
      x=arg[0]*1
      y=arg[1]*1
      radius=(x**2+y**2)**0.5
      radius.shape=radius.size
      idx = np.where((lmap>=self.DataTh)*(radius<=self.AllowedRadius))[0]
      lmap = np.log(lmap[idx])
      try :
         self.Weight=karg['Weight']*1
         self.Weight.shape=arg[2].size
         self.Weight=self.Weight[idx]
         self.Weight*=1/self.Weight.sum()
      except :
         self.Weight=1
      self.in_shape=arg[2].shape
      self.in_size=arg[2].size
      self.logdata_min=lmap.min()
      self.tagged=np.zeros(arg[2].shape,dtype='int')
      self.tagged.shape=arg[2].size
      self.tagged[idx]=1
      self.tagged.shape=arg[2].shape
      x.shape=x.size
      y.shape=y.size
      if doNotScaleAxis :
         self.xscal=1.
         self.yscal=1.
      else :
         self.xscal=x.max()-x.min()
         self.yscal=y.max()-y.min()
      x=x[idx]/self.xscal
      y=y[idx]/self.yscal
      self.N=len(idx)
      self.S=np.zeros([6,6]) 
      self.VV=np.zeros(6) 
      #unknown are A,B,C,D,E,F
      #
      self.S[0,0]= 0.25*(self.Weight*x**4).sum()
      self.S[0,1]= 0.5*((self.Weight*x**3)*y).sum()
      self.S[0,2]= 0.25*(self.Weight*x**2*y**2).sum()
      self.S[0,3]= 0.25*(self.Weight*x**3).sum()
      self.S[0,4]= 0.25*(self.Weight*x**2*y).sum()
      self.S[0,5]= -0.5*(self.Weight*x**2).sum()
      #
      self.S[1,1]= (self.Weight*x**2*y**2).sum()
      self.S[1,2]= 0.5*(self.Weight*x*y**3).sum()
      self.S[1,3]= 0.5*(self.Weight*x**2*y).sum()
      self.S[1,4]= 0.5*(self.Weight*x*y**2).sum()
      self.S[1,5]= -(self.Weight*x*y).sum()
      #
      self.S[2,2]= 0.25*(self.Weight*y**4).sum()
      self.S[2,3]= 0.25*(self.Weight*x*y**2).sum()
      self.S[2,4]= 0.25*(self.Weight*y**3).sum()
      self.S[2,5]= -0.5*(self.Weight*y**2).sum()
      #
      self.S[3,3]= 0.25*(self.Weight*x**2).sum()
      self.S[3,4]= 0.25*(self.Weight*x*y).sum()
      self.S[3,5]= -0.5*((self.Weight*x).sum()).min()
      #
      self.S[4,4]= 0.25*(self.Weight*y**2).sum()
      self.S[4,5]= -0.5*(self.Weight*y.sum()).min()
      #
      self.S[5,5]= float(len(idx))
      for r in range(len(self.VV)) :
         for c in range(len(self.VV)) :
            if r > c :
               self.S[r,c]=self.S[c,r]*1
      self.VV[0] = -0.5*(self.Weight*lmap*x**2).sum()
      self.VV[1] = -(self.Weight*lmap*x*y).sum()
      self.VV[2] = -0.5*(self.Weight*lmap*y**2).sum()
      self.VV[3] = -0.5*(self.Weight*lmap*x).sum()
      self.VV[4] = -0.5*(self.Weight*lmap*y).sum()
      self.VV[5] = ((self.Weight*lmap).sum()).min()
      #
      self.inv=linalg.inv(self.S)
      self.det=linalg.det(self.S)
      self.pars=np.dot(linalg.inv(self.S),self.VV)
      self.ld=lmap
      self.x=x
      self.y=y
      self.res = self.mdl(x,y)-lmap
      self.ksq_log = (self.res**2).sum()
      #removes the regularizzation
      self.Pars={}
      self.Pars['A']=self.pars[0]/self.xscal**2
      self.Pars['B']=self.pars[1]/self.xscal/self.yscal
      self.Pars['C']=self.pars[2]/self.yscal**2
      self.Pars['D']=self.pars[3]/self.xscal
      self.Pars['E']=self.pars[4]/self.yscal
      self.Pars['F']=self.pars[5]+np.log(self.peak)
      # find the invC matrix
      self.MinvC=np.zeros([2,2])
      self.MinvC[0][0]=self.Pars['A']*1.
      self.MinvC[0][1]=self.Pars['B']*1.
      self.MinvC[1][0]=self.Pars['B']*1.
      self.MinvC[1][1]=self.Pars['C']*1.
      # find the V0 vector 
      self.V0=np.zeros(2)
      self.V0[0]=self.Pars['D']*1.
      self.V0[1]=self.Pars['E']*1.
      # find the center
      self.MC = np.zeros([2,2])
      self.MC[0][0]=self.Pars['C']*1.
      self.MC[0][1]=-self.Pars['B']*1.
      self.MC[1][0]=-self.Pars['B']*1.
      self.MC[1][1]=self.Pars['A']*1.
      self.MC=self.MC/(self.Pars['A']*self.Pars['C']-self.Pars['B']**2)
      self.R0=np.zeros(2)
      self.R0[0]=self.Pars['C']*self.Pars['D']-self.Pars['B']*self.Pars['E']
      self.R0[1]=-self.Pars['B']*self.Pars['D']+self.Pars['A']*self.Pars['E']
      self.R0 = -0.5*self.R0/(self.Pars['A']*self.Pars['C']-self.Pars['B']**2)
      # find the allowed radius
      self.allowed_radius=(((self.x*self.xscal-self.R0[0])**2+(self.y*self.yscal-self.R0[1])**2)**0.5).max()
      # find the eigenvalues and eighenvectors
      self.heighen_val,self.heighen_vec=linalg.eigh(self.MinvC)
      semiaxis_fwhm=2.*np.sqrt(2.*(np.log(2.))/self.heighen_val)
      self.rot=np.transpose(self.heighen_vec/linalg.det(self.heighen_vec))
      for i in range(2) : self.rot[i]*=-1 if self.rot[i][i] < 0 else 1
      # extract the gaussian parameters
      hv=self.heighen_vec
      for i in range(2) : hv[i]*=-1 if hv[i][i] < 0 else 1
      self.psi_ell=np.arctan2(hv[1][0],hv[0][0])*180./np.pi
      self.fwhm_min=semiaxis_fwhm.min()
      self.fwhm_max=semiaxis_fwhm.max()
      self.fwhm=(self.fwhm_max*self.fwhm_min)**0.5
      self.ellipticity=self.fwhm_max/self.fwhm_min
      #self.zero=self.Pars['F']-0.5/4.*self.Pars['A']*self.Pars['D']**2-1./4.*self.Pars['B']*self.Pars['E']*self.Pars['D']-0.5/4.*self.Pars['C']*self.Pars['E']**2
      self.zero=self.Pars['F']+0.5*(self.Pars['A']*self.R0[0]**2+2*self.Pars['B']*self.R0[0]*self.R0[1]+self.Pars['C']*self.R0[1]**2)
      self.gauss_at_center=np.exp(self.zero)
      self.gauss_peak=np.exp(self.zero)
      self.gauss_ksq=((np.exp(self.res)-self.peak*np.exp(lmap))**2).sum()
   def mdl(self,x,y) :
      acc = self.pars[0]*x**2
      acc += 2.*self.pars[1]*x*y
      acc += self.pars[2]*y**2
      acc += self.pars[3]*x
      acc += self.pars[4]*y
      acc *= -0.5
      acc += self.pars[5]
      return acc
   def test_map(self,X,Y,X0,Y0,fwhm_min,fwhm_max,psi_ell,peak) :
      import numpy as np
      cp=np.cos(psi_ell/180.*np.pi)
      sp=np.sin(psi_ell/180.*np.pi)
      u=(X-X0)*cp+(Y-Y0)*sp
      v=-(X-X0)*sp+(Y-Y0)*cp
      smin=fwhm_min/(2.*np.sqrt(2.*np.log(2.)))
      smax=fwhm_max/(2.*np.sqrt(2.*np.log(2.)))
      return peak*np.exp(-0.5*( (u/smax)**2 + (v/smin)**2))
   def __str__(self) :
      l=[]
      l.append("in_shape       : "+str(self.in_shape))
      l.append("in_size        : "+str(self.in_size))
      l.append("DataTh         : "+str(self.DataTh))
      l.append("AllowedRadius  : "+str(self.AllowedRadius))
      l.append("N              : "+str(self.N))
      l.append("allowed_radius : "+str(self.allowed_radius))
      l.append("xscal          : "+str(self.xscal))
      l.append("yscal          : "+str(self.yscal))
      l.append("               : ")
      l.append("peak           : "+str(self.gauss_peak))
      l.append("fwhm           : "+str(self.fwhm))
      l.append("fwhm_min       : "+str(self.fwhm_min))
      l.append("fwhm_max       : "+str(self.fwhm_max))
      l.append("ellipticity    :"+str(self.ellipticity))
      l.append("psi_ell        :"+str(self.psi_ell))
      l.append("X0             :"+str(self.R0[0]))
      l.append("Y0             :"+str(self.R0[1]))
      return "\n".join(l)

class NoBackground(NoBackground_Base) :
   def __init__(self,X,Y,D,DataTh=None,AllowedRadius=None,Weight=None,doNotScaleAxis=True) :
      import numpy as np
      NoBackground_Base.__init__(self,X,Y,D,DataTh=DataTh,AllowedRadius=AllowedRadius,Weight=Weight,doNotScaleAxis=doNotScaleAxis)
      self.R0 = np.arcsin(self.R0)*180./np.pi
      self.fwhm = self.fwhm*180./np.pi
      self.fwhm_min = self.fwhm_min*180./np.pi
      self.fwhm_max = self.fwhm_max*180./np.pi
      self.xscal = np.arcsin(self.xscal)*180./np.pi
      self.yscal = np.arcsin(self.yscal)*180./np.pi
      self.allowed_radius= self.allowed_radius

class gaussCanonicalForm_NoCent :
   """used to convert gauss from Closed Form without Center:
      D=0, E=0
   """
   def __init__(self,GaussClosedForm) :
      import numpy as np
      #find the background
      self.background=GaussClosedForm.b*1
      # find the invC matrix
      self.MinvC=np.zeros([2,2])
      self.MinvC[0][0]=GaussClosedForm.A*1.
      self.MinvC[0][1]=GaussClosedForm.B*1.
      self.MinvC[1][0]=GaussClosedForm.B*1.
      self.MinvC[1][1]=GaussClosedForm.C*1.
     # find the center
      self.MC = np.zeros([2,2])
      self.MC[0][0]=GaussClosedForm.C*1.
      self.MC[0][1]=-GaussClosedForm.B*1.
      self.MC[1][0]=-GaussClosedForm.B*1.
      self.MC[1][1]=GaussClosedForm.A*1.
      self.MC=self.MC/(GaussClosedForm.A*GaussClosedForm.C-GaussClosedForm.B**2)
      self.R0=np.zeros(2)
      # find the allowed radius
      self.allowed_radius=(((self.x*self.xscal-self.R0[0])**2+(self.y*self.yscal-self.R0[1])**2)**0.5).max()
      # find the eigenvalues and eighenvectors
      self.heighen_val,self.heighen_vec=linalg.eigh(self.MinvC)
      semiaxis_fwhm=2.*np.sqrt(2.*(np.log(2.))/self.heighen_val)
      self.rot=np.transpose(self.heighen_vec/linalg.det(self.heighen_vec))
      for i in range(2) : self.rot[i]*=-1 if self.rot[i][i] < 0 else 1
      # extract the gaussian parameters
      hv=self.heighen_vec
      for i in range(2) : hv[i]*=-1 if hv[i][i] < 0 else 1
      self.psi_ell=np.arctan2(hv[1][0],hv[0][0])*180./np.pi
      self.fwhm_min=semiaxis_fwhm.min()
      self.fwhm_max=semiaxis_fwhm.max()
      self.fwhm=(self.fwhm_max*self.fwhm_min)**0.5
      self.ellipticity=self.fwhm_max/self.fwhm_min
      self.zero=GaussClosedForm.F
      self.gauss_peak=np.exp(self.zero)
   def csv(self,header=False,fsept=', ',fmt='%20.18e') :
      "returns a csv table line with the essential information, X0 and Y0 are forced to be 0"
      if header :
         return fsept.join(['peak','X0','Y0','fwhm','ellipticity','psi_ell','background'])
      return fsept.join([fmt%self.gauss_peak,fmt%0,fmt%0,fmt%self.fwhm,fmt%self.ellipticity,fmt%self.psi_ell,fmt%self.background])

class gaussCanonicalForm :
   """used to convert gauss from Closed Form"""
   def __init__(self,GaussClosedForm) :
      import numpy as np
      #find the background
      self.background=GaussClosedForm.b*1
      # find the invC matrix
      self.MinvC=np.zeros([2,2])
      self.MinvC[0][0]=GaussClosedForm.A*1.
      self.MinvC[0][1]=GaussClosedForm.B*1.
      self.MinvC[1][0]=GaussClosedForm.B*1.
      self.MinvC[1][1]=GaussClosedForm.C*1.
      # find the V0 vector 
      self.V0=np.zeros(2)
      self.V0[0]=GaussClosedForm.D*1.
      self.V0[1]=GaussClosedForm.E*1.
      # find the center
      self.MC = np.zeros([2,2])
      self.MC[0][0]=GaussClosedForm.C*1.
      self.MC[0][1]=-GaussClosedForm.B*1.
      self.MC[1][0]=-GaussClosedForm.B*1.
      self.MC[1][1]=GaussClosedForm.A*1.
      self.MC=self.MC/(GaussClosedForm.A*GaussClosedForm.C-GaussClosedForm.B**2)
      self.R0=np.zeros(2)
      self.R0[0]=GaussClosedForm.C*GaussClosedForm.D-GaussClosedForm.B*GaussClosedForm.E
      self.R0[1]=-GaussClosedForm.B*GaussClosedForm.D+GaussClosedForm.A*GaussClosedForm.E
      self.R0 = -0.5*self.R0/(GaussClosedForm.A*GaussClosedForm.C-GaussClosedForm.B**2)
      # find the allowed radius
      self.allowed_radius=(((self.x*self.xscal-self.R0[0])**2+(self.y*self.yscal-self.R0[1])**2)**0.5).max()
      # find the eigenvalues and eighenvectors
      self.heighen_val,self.heighen_vec=linalg.eigh(self.MinvC)
      semiaxis_fwhm=2.*np.sqrt(2.*(np.log(2.))/self.heighen_val)
      self.rot=np.transpose(self.heighen_vec/linalg.det(self.heighen_vec))
      for i in range(2) : self.rot[i]*=-1 if self.rot[i][i] < 0 else 1
      # extract the gaussian parameters
      hv=self.heighen_vec
      for i in range(2) : hv[i]*=-1 if hv[i][i] < 0 else 1
      self.psi_ell=np.arctan2(hv[1][0],hv[0][0])*180./np.pi
      self.fwhm_min=semiaxis_fwhm.min()
      self.fwhm_max=semiaxis_fwhm.max()
      self.fwhm=(self.fwhm_max*self.fwhm_min)**0.5
      self.ellipticity=self.fwhm_max/self.fwhm_min
      self.zero=GaussClosedForm.F+0.5*(GaussClosedForm.A*self.R0[0]**2+2*GaussClosedForm.B*self.R0[0]*self.R0[1]+GaussClosedForm.C*self.R0[1]**2)
      self.gauss_peak=np.exp(self.zero)
   def csv(self,header=False,fsept=', ',fmt='%20.18e') :
      "returns a csv table line with the essential information"
      if header :
         return fsept.join(['peak','X0','Y0','fwhm','ellipticity','psi_ell','background'])
      return fsept.join([fmt%self.gauss_peak,fmt%self.R0[0],fmt%self.R0[1],fmt%self.fwhm,fmt%self.ellipticity,fmt%self.psi_ell,fmt%self.background])
      

class gaussClosedForm :
   """class to handle a gaussian curve in closed form
   """
   def __init__(self,A,B,C,D,E,F,b) :
      "defines a closed form gaussian for A,B,C,D,E,F,b"
      self.A=A
      self.B=B
      self.C=C
      self.D=D
      self.E=E
      self.F=F
      self.b=b
   def calc(self,X,Y):
      "computes for X and Y"
      import numpy as np
      acc=self.A*X**2
      acc+=self.B*X*Y
      acc+=self.C*Y**2
      acc+=self.D*X
      acc+=self.E*Y
      return np.exp(-0.5*acc+self.F)+self.b
   def canonization(self) :
      "convert closed form parameters to canonical form"
      return gaussCanonicalForm(self)
   def csv(self,header=False,fsept=', ',fmt='%20.18e') :
      "returns a csv table line with the essential information"
      if header :
         return fsept.join(['A','B','C','D','E','F','b'])
      return fsept.join([fmt%self.A,fmt%self.B,fmt%self.C,fmt%self.D,fmt%self.E,fmt%self.F,fmt%self.b])
   def __call__(self,XY,A,B,C,D,E,F,b) :
      """call to perform fit with curve_fit
      XY = array of X and Y"""
      self.A=A
      self.B=B
      self.C=C
      self.D=D
      self.E=E
      self.F=F
      self.b=b
      return self.calc(XY[0],XY[1])
      
class efficient_gaussClosedForm_for_fit :
   """class to compute 'efficiently' a gaussian distribution in closed form"""
   def __init__(self,X,Y) :
      "defines a closed form gaussian for A,B,C,D,E,F,b"
      import copy
      self.X=copy.deepcopy(X)
      self.Y=copy.deepcopy(Y)
   def __call__(self,A,B,C,D,E,F,b) :
      "computes for X and Y"
      import numpy as np
      acc=A*self.X**2
      acc+=B*self.X*self.Y
      acc+=C*self.Y**2
      acc+=D*self.X
      acc+=E*self.Y
      return np.exp(-0.5*acc+F)+b
      
class efficient_chisq_gaussClosedForm_for_fit :
   """class to compute 'efficiently' a chisq for a given gaussian distribution in closed form"""
   def __init__(self,X,Y,Data) :
      "defines a closed form gaussian for A,B,C,D,E,F,b"
      import copy
      self.X=copy.deepcopy(X)
      self.Y=copy.deepcopy(Y)
      self.Data=copy.deepcopy(Data)
   def gauss(self,A,B,C,D,E,F,b) :
      "computes for X and Y"
      import numpy as np
      acc=A*self.X**2
      acc+=B*self.X*self.Y
      acc+=C*self.Y**2
      acc+=D*self.X
      acc+=E*self.Y
      return np.exp(-0.5*acc+F)+b
   def residual(self,A,B,C,D,E,F,b) :
      return (self.gauss(A,B,C,D,E,F,b)-self.Data)
   def __call__(self,A,B,C,D,E,F,b) :
      return (self.residual(A,B,C,D,E,F,b)**2).sum()
      
#class super_efficient_gaussClosedForm_forfit :
   #def __init__(self,X,Y,V) :
      #"defines a closed form gaussian for A,B,C,D,E,F,b"
      #self.X=X
      #self.Y=Y
      #self.V=V
   #def __call__(self,A,B,C,D,E,F,b) :
      #"computes for X and Y"
      #import numpy as np
      #acc=A*self.X**2
      #acc+=B*self.X*self.Y
      #acc+=C*self.Y**2
      #acc+=D*self.X
      #acc+=E*self.Y
      #return ((np.exp(-0.5*acc+F)+b-V)**2).sum()
      
if __name__=='__main__' :
   def TestOut(title,m1,GF,latex=False) :
      if latex :
         fmt = '{\\bf %11s} & %13e & %13e & %13e\\\\'
         print '\\begin{tabular}{lccc}\n\\hline\\hline\n\\multicolumn{4}{c}{%s}\\\\\n\hline'%title
         print "\\hline\n&\\multicolumn{1}{c}{{\\bf Input}}&\\multicolumn{1}{c}{{\\bf Fit}}&\\multicolumn{1}{c}{{\\bf Residual}}\\\\ \n \hline"
      else :
         fmt = '%11s :  %13e %13e %13e'
         print title
      for k in ['gauss_peak','fwhm','fwhm_min','fwhm_max','ellipticity','psi_ell','R0'] :
         name=k
         if latex :
            name='\\_'.join(name.split('_'))
         if k=='R0' :
            print fmt%('X0',m1.__dict__[k][0],GF.__dict__[k][0],GF.__dict__[k][0]-m1.__dict__[k][0])
            print fmt%('Y0',m1.__dict__[k][1],GF.__dict__[k][1],GF.__dict__[k][1]-m1.__dict__[k][1])
         else :
            print fmt%(name,m1.__dict__[k],GF.__dict__[k],GF.__dict__[k]-m1.__dict__[k])
      if latex :
         print '\\hline\\hline\n &&&\\\\ \n \\end{tabular}'
         print 
      else :
         print 
      
   print "\nA test\n"
  
   latex=True
  
  
   m1=Model(-1.5e-2,1.5e-2,301,-1.5e-2,1.5e-2,301)
   
   pxl=m1.dX

   m1(1.,0.,0.,0.,30.*pxl,1.,MinMax=False)
   GF=NoBackground_Base(m1.X,m1.Y,m1.D)
   TestOut('Simmetric, centered',m1,GF,latex=latex)
   
   m1(1.,0,0.1*pxl,0.,30.*pxl,1.,MinMax=False)
   GF=NoBackground_Base(m1.X,m1.Y,m1.D)
   TestOut('Simmetric, North',m1,GF,latex=latex)
   
   m1(1.,0.,-0.1*pxl,0.,30.*pxl,1.,MinMax=False)
   GF=NoBackground_Base(m1.X,m1.Y,m1.D)
   TestOut('Simmetric, West',m1,GF,latex=latex)
   
   m1(1.,0,-0.1*pxl,0.,30.*pxl,1.,MinMax=False)
   GF=NoBackground_Base(m1.X,m1.Y,m1.D)
   TestOut('Simmetric, South',m1,GF,latex=latex)

   m1(1.,0.1*pxl,0.,0.,30.*pxl,1.,MinMax=False)
   GF=NoBackground_Base(m1.X,m1.Y,m1.D)
   TestOut('Simmetric, East',m1,GF,latex=latex)

   m1(1.,0.,0.,0.,30.*pxl,2.,MinMax=False)
   GF=NoBackground_Base(m1.X,m1.Y,m1.D)
   TestOut('Asimmetric, Center',m1,GF,latex=latex)
   
   m1(1.,0.,0.1*pxl,0.,30.*pxl,2.,MinMax=False)
   GF=NoBackground_Base(m1.X,m1.Y,m1.D)
   TestOut('Asimmetric, North',m1,GF,latex=latex)

   m1(1.,0.,-0.1*pxl,0.,30.*pxl,2.,MinMax=False)
   GF=NoBackground_Base(m1.X,m1.Y,m1.D)
   TestOut('Asimmetric, West',m1,GF,latex=latex)
   
   m1(1.,0,-0.1*pxl,0.,30.*pxl,2.,MinMax=False)
   GF=NoBackground_Base(m1.X,m1.Y,m1.D)
   TestOut('Asimmetric, South',m1,GF,latex=latex)

   m1(1.,0.1*pxl,0,0.,30.*pxl,2.,MinMax=False)
   GF=NoBackground_Base(m1.X,m1.Y,m1.D)
   TestOut('Asimmetric, East',m1,GF,latex=latex)

   m1(1.,0.*pxl,0*pxl,45.,30.*pxl,2.,MinMax=False)
   GF=NoBackground_Base(m1.X,m1.Y,m1.D)
   TestOut('Asimmetric, Center, Rotated 45 deg',m1,GF,latex=latex)

   m1(1.,0.*pxl,0*pxl,90.,30.*pxl,2.,MinMax=False)
   GF=NoBackground_Base(m1.X,m1.Y,m1.D)
   TestOut('Asimmetric, Center, Rotated 90 deg',m1,GF,latex=latex)

   m1(1.,0.*pxl,0*pxl,-45.,30.*pxl,2.,MinMax=False)
   GF=NoBackground_Base(m1.X,m1.Y,m1.D)
   TestOut('Asimmetric, Center, Rotated -45 deg',m1,GF,latex=latex)

   m1(1.,0.*pxl,0*pxl,-89.,30.*pxl,2.,MinMax=False)
   GF=NoBackground_Base(m1.X,m1.Y,m1.D)
   TestOut('Asimmetric, Center, Rotated -89 deg',m1,GF,latex=latex)

   m1(1.,0.1*pxl,-0.1*pxl,-45.,30.*pxl,2.,MinMax=False)
   GF=NoBackground_Base(m1.X,m1.Y,m1.D)
   TestOut('Asimmetric, South East, Rotated -45 deg',m1,GF,latex=latex)

"""
Example of chisq fitting using iminuit

from numpy import *

from iminuit.util import make_func_code, describe

import iminuit

x=zeros([601,601]) ; y=zeros([601,601])

for k in range(601) : x[:,k]=float(k-300)
for k in range(601) : y[k,:]=float(k-300)

x.shape=x.size;y.shape=y.size;GG=GaussFit.efficient_gaussClosedForm_for_fit(x,y);KSQ=GaussFit.efficient_chisq_gaussClosedForm_for_fit(x,y,GG(0.001,0.,0.002,0.,0.,0.,1.)+randn(x.size)*0.1)

mKSQ=iminuit.Minuit(KSQ,D=0,E=0,fix_D=True,fix_E=True,A=0.005,B=0.005,C=0.005,F=0.,b=0.9)

mKSQ.migrad()
"""
