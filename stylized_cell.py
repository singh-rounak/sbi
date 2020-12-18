from neuron import h
import math
import numpy as np
import pandas as pd
from stylized_module.current_injection import Current_injection

h.load_file('stdrun.hoc')

class Stylized_Cell(object):
    def __init__(self,geometry=None,dL=30,vrest=-70.0):
        """
        Initialize cell model
        geometry: pandas dataframe of cell morphology properties
        dL: maximum segment length
        vrest: reversal potential of leak channel for all segments
        """
        self._h = h
        self._dL = dL
        self._vrest = vrest
        self._nsec = 0
        self._nseg = 0
        self.all = []  # list of all sections
        self.segments = []  # list of all segments
        self.sec_id_in_seg = []
        self.seg_coords = {}
        self.injection = []
        self.set_geometry(geometry)
        self.setup_all()
        
    def setup_all(self):
        if self.geometry is not None:
            self.set_morphology()
            self.set_channels()
#             self.calc_seg_coords()
#             self.init_v()
    
    def set_geometry(self,geometry):
        if geometry is None:
            self.geometry = None
        else:
            if not isinstance(geometry,pd.DataFrame):
                raise TypeError("geometry must be a pandas dataframe")
            if geometry.iloc[0]['type']!=1:
                raise ValueError("first row of geometry must be soma")
            self.geometry = geometry.copy()
    
    def set_morphology(self):
        """Create cell morphology"""
        if self.geometry is None:
            print("Warning: geometry is not loaded.")
            return None
        self._nsec = 0
        self.all = []
        sec_index = [0]*len(self.geometry.index)
        for j,sec in self.geometry.iterrows():
            axial = sec['axial']
            L = sec['L']
            R = sec['R']
            ang = sec['ang']
            sec_index[j] = self._nsec
            section = self.add_section(name=sec['name'],diam=2*R)
            if j==0:
                R0 = R
                nseg = 1
                self.soma = section
                pt0 = [0.,-2*R0,0.]
                pt1 = [0.,0.,0.]
            else:
                nseg = math.ceil(L/self._dL)
                pid = sec_index[sec['pid']]
                psec = self.all[pid]
                pt0 = [psec.x3d(1),psec.y3d(1),psec.z3d(1)]
                pt1 = [0.,L*math.sin(ang),0.]
                if not axial:
                    pt1[0] = L*math.cos(ang)
                for i in range(3):
                    pt1[i] += pt0[i]
                section.connect(psec(1),0)
            self.set_location(section,pt0,pt1,nseg)
            if not axial:
                section = self.add_section(name=sec['name'],diam=2*R)
                section.connect(psec(1),0)
                pt1[0] = -pt1[0]
                self.set_location(section,pt0,pt1,nseg)
        self.set_location(self.soma,[0.,-R0,0.],[0.,R0,0.],1)
        self.store_segments()
    
    def add_section(self,name='null_sec',diam=500.0):
        sec = h.Section(name=name)
        sec.diam = diam
        self.all.append(sec)
        self._nsec += 1
        return sec
    
    def set_location(self,sec,pt0,pt1,nseg):
        sec.pt3dclear()
        sec.pt3dadd(*pt0,sec.diam)
        sec.pt3dadd(*pt1,sec.diam)
        sec.nseg = nseg
    
    def store_segments(self):
        self.segments = []
        self.sec_id_in_seg = []
        nseg = 0
        for sec in self.all:
            self.sec_id_in_seg.append(nseg)
            nseg += sec.nseg
            for seg in sec:
                self.segments.append(seg)
        self._nseg = nseg
    
    def calc_seg_coords(self):
        """Calculate segment coordinates for ECP calculation"""
        p0 = np.empty((self._nseg,3))
        p1 = np.empty((self._nseg,3))
        p05 = np.empty((self._nseg,3))
        r = np.empty(self._nseg)
        for isec,sec in enumerate(self.all):
            iseg = self.sec_id_in_seg[isec]
            nseg = sec.nseg
            pt0 = np.array([sec.x3d(0),sec.y3d(0),sec.z3d(0)])
            pt1 = np.array([sec.x3d(1),sec.y3d(1),sec.z3d(1)])
            pts = np.linspace(pt0,pt1,2*nseg+1)
            p0[iseg:iseg+nseg,:] = pts[:-2:2,:]
            p1[iseg:iseg+nseg,:] = pts[2::2,:]
            p05[iseg:iseg+nseg,:] = pts[1:-1:2,:]
            r[iseg:iseg+nseg] = sec.diam/2
        self.seg_coords = {}
        self.seg_coords['dl'] = p1-p0  # length direction vector
        self.seg_coords['pc'] = p05  # center coordinates
        self.seg_coords['r'] = r  # radius
    
    def get_sec_by_id(self,index=None):
        """Get list of section objects by indices in the section list"""
        if not isinstance(index, (list,tuple,np.ndarray)):
            index = [index]
        return [self.all[i] for i in index]
    
    def get_seg_by_id(self,index=None):
        """Get list of segment objects by indices in the segment list"""
        if not isinstance(index, (list,tuple,np.ndarray)):
            index = [index]
        return [self.segments[i] for i in index]
    
    def set_channels(self):
        """Abstract method for setting biophysical properties, inserting channels"""
        pass
    
    def set_all_passive(self,gl=0.0003):
        """A use case of 'set_channels', set all sections passive membrane"""
        for sec in self.all:
            sec.cm = 1.0
            sec.insert('pas')
            sec.g_pas = gl
            sec.e_pas = self._vrest
    
    def set_soma_hh(self,gl_dend=0.0003,**soma_param):
        """A use case of 'set_channels', set all sections passive but soma with HH channels"""
        for sec in self.all:
            sec.cm = 1.0
        self.soma.insert('hh')
        self.soma.el_hh = self._vrest
        for param,value in soma_param.items():
            setattr(self.soma,param+'_hh',value)
        for sec in self.all[1:]:
            sec.insert('pas')
            sec.g_pas = gl_dend
            sec.e_pas = self._vrest
    
    def add_injection(self,sec_index,**kwargs):
        """Add current injection to a section by its index"""
        self.injection.append(Current_injection(self,sec_index,**kwargs))
    
    def set_h(self,**h_param):
        """For setting neuron.h attributes inside this class"""
        for param,value in h_param.items():
            setattr(h,param,value)
    
    def init_v(self):
        """Set all segments initial voltage to vrest"""
        for seg in self.segments:
            seg.v = self.vrest
