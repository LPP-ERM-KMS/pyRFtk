"""
Wave form generator class

there are 3 parts to the waveform :
    the prolog
    the part repeated a preset number of times
    the epilog part

each waveform part is defined by a list of tuples (tn, wn) where the first
tuple's tn = 0.

the start of the next (or repeated) part is at the time of the preceding
part's tn.


F. Durodie 7 Feb 2014

+-------------+----------------------------------------------------------------+
| Date        | Comment                                                        |
+=============+================================================================+
| 2015-Jun-16 | corrected the building up of the waveform to allow for a new   |
|             | version of scipy.interp1d which allows non-monotonic           |
|             | independent variable to be supplied.                           |
+-------------+----------------------------------------------------------------+
| 2015-Jun-02 | Added the WF_Edit() class which implements a GUI to edit a     |
|             | WaveForm and an edit() method which will edit the WaveForm in  |
|             | situ. Another method asdict() has been added superceeding the  |
|             | the method askwargs() (depreciated).                           |
+-------------+----------------------------------------------------------------+

"""
#===============================================================================
#
# import the usual stuff
#

from scipy.interpolate import interp1d

#===============================================================================
#
# Waveform class
#

class WaveForm():

    def __init__(self, **kwargs):
        """
        example :
        
        WaveForm(dt=0.001,
                 Nrepeat=10,
                 prolog = [(0.000,-2.),
                           (0.500,-2.),
                           (1.000, 2.),
                           (1.500, 2.)],
                 repeat = [(0.000, 2.),
                           (0.001,-2.),
                           (0.005,-2.),
                           (0.010,-1.),
                           (0.020, 0.),
                           (0.040, 1.),
                           (0.080, 2.),
                           (0.100, 2.)],
                 epilog = [(0.000, 2.),
                           (0.500,-2.),
                           (1.000,-2.0)])
                           
        dt : defaults to 0.001
        Nrepeat : defaults to 1 if repeat is present else 0
        prolog : defaults to None
        repeat : defaults to None
        epilog : defaults to None
        min_dt  : defaults to 0.001 of dt
        
        one of prolog, repeat or epilog must be non-None
        """
               
        self.time = 0.
        self.dt = kwargs.get('dt', 0.001)
        self.prolog = kwargs.get('prolog', None)
        self.repeat = kwargs.get('repeat', None)
        self.epilog = kwargs.get('epilog', None)
        self.Nrep = kwargs.get('Nrepeat',1 if self.repeat else 0)
        self.min_dt = kwargs.get('min_dt', 0.001 * self.dt)
        self._init2_()
        
    def _init2_(self):
        def AddPart(part):
            if type(part) is list:
                # print('AddPart; part=',part)
                t0 = self.tw[-1] if self.tw else 0.
                try :
                    for tk, wk in part: 
                        # this will correct a feature of a later version of 
                        # scipy.interp1d as we have seen on Mike Kaufman's 
                        # macOS version.
                        self.tw.append(t0 + tk + 
                               (self.min_dt if tk == 0. and t0 > 0. else 0.))
                        self.ww.append(wk)
                except:
                    print(self.prolog)
                    raise ValueError('WaveForm : parts must be list of tuples'
                                     ' or None')
                # print('AddPart; tw=',self.tw)
                # print('AddPart; ww=',self.ww)

            elif part is not None:
                raise ValueError('WaveForm : parts must be list of tuples'
                                 ' or None')
            
        self.tw, self.ww = [], []
        
        AddPart(self.prolog)
        for k in range(self.Nrep):                             # @UnusedVariable
            AddPart(self.repeat)
        AddPart(self.epilog)

        if not self.tw :
            raise ValueError('WaveForm : one of prolog, repeat or epilog'
                             ' must be given')

        self.time = self.tw[0]
        
        #=======================================================================
        #
        # v a l u e
        #
        self.value = interp1d(self.tw,
                              self.ww,
                              bounds_error=False,
                              fill_value=self.ww[-1])
        
    #===========================================================================
    #
    # a s k w a r g s
    #
    def askwargs(self):
        raise NotImplementedError('WaveForm.askwargs() superceeded by '
                                  'WaveForm.asdict()')
        
        d = {}
        d['dt'] = self.dt
        d['Nrepeat'] = self.Nrep
        if self.prolog:
            d['prolog'] = self.prolog
        if self.repeat:
            d['repeat'] = self.repeat
        if self.epilog:
            d['epilog'] = self.epilog
        
        return d
    
    #===========================================================================
    #
    # _ _ s t r _ _
    #
    def __str__(self):
        s  = 'WaveForm(dt = %s,\n' % repr(self.dt)
        s += '         Nrepeat = %s\n' % repr(self.Nrep)
        if self.prolog:
            s += '         prolog = ['
            sp = ''
            for t in self.prolog:
                s += sp+'(%r,%r),\n' % t
                sp = '                   '
            s = s[:-1] + ']\n'
        if self.repeat:
            s += '         repeat = ['
            for t in self.repeat:
                s += sp+'(%r,%r),\n' % t
                sp = '                   '
            s = s[:-1] + ']\n'
        if self.epilog:
            s += '         epilog = ['
            for t in self.epilog:
                s += sp+'(%r,%r),\n' % t
                sp = '                   '
            s = s[:-1] + ']\n'
        s = s[:-1]+')\n' # drop the last \n
        return s
            
    #===========================================================================
    #
    # a s d i c t
    #
    def asdict(self):
        """asdict:
        
           returns the parameters to create the waveform as a dict which
           can be edited by WF_Edit and resupplied to build a new wavform
           to WaveForm(**the_dict).
        """
        
        d = {}
        d['dt'] = self.dt
        d['Nrepeat'] = self.Nrep
        if self.prolog:
            d['prolog'] = self.prolog
        if self.repeat:
            d['repeat'] = self.repeat
        if self.epilog:
            d['epilog'] = self.epilog

        return d
    
    #===========================================================================
    #
    # e d i t
    #
    def edit(self):
        wf = WF_Edit(wf=self.asdict()).result
        self.time = 0.
        self.Nrep = wf['Nrepeat']
        self.dt = wf['dt']
        self.prolog = wf['prolog']
        self.repeat = wf['repeat']
        self.epilog = wf['epilog']
        self._init2_()
        
    #===========================================================================
    #
    # T m a x
    #
    def Tmax(self):
        return self.tw[-1]
    
    #===========================================================================
    #
    # _ _ g e t i t e m _ _
    #
    def __getitem__(self,i):
        return self.tw[i], self.ww[i]
    
    #===========================================================================
    #
    # _ _ i t e r _ _
    #
    def __iter__(self):
        t = self.tw[0]
        while t <= self.tw[-1]:
            yield t, self.value(t)
            t += self.dt
        
    #===========================================================================
    #
    # s t e p
    #
    def step(self):
        t, value = self.time, self.value(self.time)
        self.time += self.dt
        return t, value

    #===========================================================================
    #
    # t r a c e
    #
    def trace(self, **kwargs):
        """
        return a tuple of 2 lists (time and wave)
        
        if dt is supplied it will be used instead of self.dt
        """
        
        dt = self.dt # save the object's supplied dt
        self.dt = kwargs.get('dt', self.dt)
        
        ts, ws = [], []
        for t, w in self:
            ts.append(t)
            ws.append(w)
            
        self.dt = dt # restore the object's supplied dt
        
        return ts,ws
    
#===============================================================================
if False:
    def strdir(obj, cols=4, width=30, skip=True ):
        s = ''
        dk = 0
        for d in dir(obj):
            if skip and d[0] == '_':
                continue
            dk += 1
            s += f'{d:{width}s} '
            if dk == cols:
                s += '\n'
                dk = 0
        if s[-1] == '\n':
            s = s[:-1]
        return s
    
    import tkinter as Tk
    import pylab as pl
    import numpy as np
    from copy import deepcopy
    
    import matplotlib
    print(matplotlib.__version__)
    
    print('\nmatplotlib\n')
    
    print(strdir(matplotlib))
    
    print('\nmatplotlib.image\n')
    print(strdir(matplotlib.image))
    
    print('\nmatplotlib.backend_tools\n')
    print(strdir(matplotlib.backend_tools))

    print('\nmatplotlib.backend_bases\n')
    print(strdir(matplotlib.backend_bases))

    print('\nmatplotlib.backend_managers\n')
    print(strdir(matplotlib.backend_managers))
    
    #
    # these have changed with the matplotlib versions (sigh)
    #
    from matplotlib.backends.backend_agg import FigureCanvas as FigureCanvasTkAgg
    from matplotlib.backends.backend_agg import NavigationToolbar2 as NavigationToolbar2TkAgg
    
    #===============================================================================
    
    class WF_Edit(Tk.Tk):
        def __init__(self, title='WaveForm Editor', parent=None,      
                     wf={'dt':0.001,
                         'prolog':[(0,0)],
                         'repeat':[(0,0)],
                         'Nrepeat':0,
                         'epilog':[(0,0)]}):
            
            Tk.Tk.__init__(self,parent)
            self.wm_title(title)
            self.parent = parent
            self.twf = deepcopy(wf)
            self.result = deepcopy(wf)
            self.phase = 'prolog'
            self.point = 0
            
            self.v0 = Tk.DoubleVar()
            self.v0.set(self.twf['dt'])
            l0 = Tk.Label(text='dt:').grid(row=1,column=1)
            self.e0 = Tk.Entry(self,
                               width=5,
                               textvariable=self.v0
                              )
            self.e0.grid(row=1,column=2)
            
            b_prolog = Tk.Button(self,
                                 text="Prolog",
                                 command=self.prolog
                                ).grid(row=1,column=3)
            
            b_repeat = Tk.Button(self,
                                 text="Repeat",
                                 command=self.repeat
                                ).grid(row=1,column=4)
            
            self.v1 = Tk.IntVar()
            self.v1.set(self.twf['Nrepeat'])
            l1 = Tk.Label(text='N=').grid(row=1,column=5)
            self.e1 = Tk.Entry(self,
                               width=5,
                               textvariable=self.v1
                              )
            self.e1.grid(row=1,column=6)
            
            b_epilog = Tk.Button(self,
                                 text="Epilog",
                                 command=self.epilog
                                ).grid(row=1,column=7)
            
            self.plot_frame = Tk.Frame(self)
            self.plot_frame.grid(row=2,column=1,columnspan=10)
            
            self.fig = pl.figure(figsize=(8,8))
            self.ax = pl.subplot(1,1,1)
            
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
            self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.plot_frame)
            
            self.toolbar.pack(side=Tk.TOP)
            self.canvas.get_tk_widget().pack(side=Tk.TOP)
                    
        
            b1 = Tk.Button(self,
                           text="<",
                           command=self.previous
                          ).grid(row=3,column=1)
                          
            b2 = Tk.Button(self,
                           text=">",
                           command=self.next
                          ).grid(row=3,column=2)
                          
            b3 = Tk.Button(self,
                           text='Remove',
                           command=self.remove
                          ).grid(row=3,column=3)
    
            l3 = Tk.Label(self,text='time:').grid(row=3,column=4)
            self.v3 = Tk.DoubleVar()
            self.v3.set(self.twf[self.phase][self.point][0])
            self.e3 = Tk.Entry(self,
                               width=10,
                               textvariable=self.v3
                              ).grid(row=3,column=5)
                          
            l4 = Tk.Label(self,text='value:').grid(row=3,column=6)
            self.v4 = Tk.DoubleVar()
            self.v4.set(self.twf[self.phase][self.point][1])
            self.e4 = Tk.Entry(self,
                               width=10,
                               textvariable=self.v4
                              ).grid(row=3,column=7)
            
            b4 = Tk.Button(self,
                           text='Add',
                           command=self.add
                          ).grid(row=3,column=8)
                          
            b5 = Tk.Button(self,
                           text='Cancel',
                           command=self.cancel
                          ).grid(row=3,column=9)
                          
            b6 = Tk.Button(self,
                           text='OK',
                           command=self.done
                          )
            b6.grid(row=3,column=10)
            
            self.plot_wave = Tk.Frame(self)
            self.plot_wave.grid(row=4,column=1,columnspan=10)
                        
            self.bind('<Left>', self.previous)
            self.bind('<Right>',self.next)
            # self.bind('<Return>',self.handlereturn)
            self.bind('<Escape>',self.cancel)
            b6.bind('<Return>',self.done)
            self.e1.bind('<KeyRelease>',self.replot)
            
            self.replot()
            # print('entering main loop')
            self.mainloop()
            # print('leaving mainloop')
            
        #===========================================================================
    
        def replot(self,event=None):
            self.twf[self.phase].sort()
            x = [t[0] for t in self.twf[self.phase]]
            y = [t[1] for t in self.twf[self.phase]]
            dx, dy = x[-1]-x[0], np.max(y)-np.min(y)
            dx = dx if dx else 1
            dy = dy if dy else 1
            
            self.fig.clf()
            pl.subplot(2,1,1)
            pl.plot(x,y,'bo-',markersize=6)
            pl.plot(x[self.point],y[self.point],'r',
                    marker='o',
                    markersize=16,
                    markerfacecolor='none',
                    markeredgecolor='r',
                    markeredgewidth=3)
            pl.grid()
            
            pl.title('Waveform Edit : '+self.phase)
            pl.ylabel('value')
            pl.xlim(xmin=x[0]-0.05*dx, xmax=x[-1]+0.05*dx)
            pl.ylim(ymin=np.min(y)-0.05*dy, ymax=np.max(y)+0.05*dy)
            
            pl.subplot(2,1,2)
            self.twf['dt']=float(self.v0.get())
            self.twf['Nrepeat']=int(self.v1.get())
            xi, yi = WaveForm(**self.twf)[:]
            pl.plot(xi, yi, 'r.-', markersize=6)
            pl.grid()
            pl.xlabel('time [s]')
            pl.ylabel('value')
            pl.xlim(xmin=xi[0]-0.05*dx, xmax=xi[-1]+0.05*dx)
            pl.ylim(ymin=np.min(yi)-0.05*dy, ymax=np.max(yi)+0.05*dy)
            
            pl.tight_layout()
            self.fig.canvas.draw()
            
            self.v3.set(self.twf[self.phase][self.point][0])
            self.v4.set(self.twf[self.phase][self.point][1])
                 
        #===========================================================================
    
        def handlereturn(self, event):
            print("return: event.widget is",event.widget)
            print("focus is:",self.focus_get())  
                  
        #===========================================================================
    
        def previous(self,event=None): 
            # print('previous')
            self.point -= 1 if self.point > 0 else 0
            self.replot()
            
        #===========================================================================
    
        def next(self,event=None):
            # print('next')
            self.point += 1 if self.point < len(self.twf[self.phase]) -1 else 0
            self.replot()
                    
        #===========================================================================
    
        def add(self):
            # print('add')
            xn, yn = self.v3.get(),self.v4.get()
            x = [t[0] for t in self.twf[self.phase]]
            try:
                N = x.index(xn)
                self.twf[self.phase][N]= (xn,yn)
                self.point=N
            except ValueError:
                self.twf[self.phase].append((xn,yn))
                self.twf[self.phase].sort()
                self.point = [t[0] for t in self.twf[self.phase]].index(xn)
            
            self.replot()
                    
        #===========================================================================
    
        def remove(self):
            # print('remove')
            self.twf[self.phase].pop(self.point)
            if self.point > len(self.twf[self.phase])-1:
                self.point -= 1
            self.replot()
            
        #===========================================================================
    
        def cancel(self,event=None):
            # print('cancel')
            self.quit()
            
        #===========================================================================
    
        def prolog(self):
            # print('prolog')
            self.phase='prolog'
            self.point=0
            self.replot()
            
        #===========================================================================
    
        def repeat(self):
            # print('repeat')
            self.phase='repeat'
            self.point=0
            self.replot()
    
        #===========================================================================
    
        def epilog(self):
            # print('epilog')
            self.phase='epilog'
            self.point=0
            self.replot()
            
        #===========================================================================
    
        def done(self,event=None):
            # print('done')
            self.twf['dt']=float(self.v0.get())
            self.twf['Nrepeat']=int(self.v1.get())
            self.result = deepcopy(self.twf)
            self.quit()
            
#===============================================================================
#
# _ _ m a i n _ _
#
if __name__ == '__main__' :

    from pprint import pformat
    from matplotlib import pyplot as pl
    
#     wf = WaveForm(dt = 0.005,
#                   Nrepeat = 3,
#                   prolog = [(0.000,-2.),
#                            (0.500,-2.),
#                            (1.000, 2.),
#                            (1.500, 2.)],
#                  repeat = [(0.000, 2.),
#                            (0.001,-2.),
#                            (0.005,-2.),
#                            (0.010,-1.),
#                            (0.020, 0.),
#                            (0.040, 1.),
#                            (0.080, 2.),
#                            (0.100, 2.)],
#                  epilog = [(0.000, 2.),
#                            (0.500,-2.),
#                            (1.000,-2.0)])

    if True:
        wf = WaveForm(dt=0.0001,prolog=[(0.000, 0),(0.500,0),(0.500,255),(30.000,255),(30.001,0)])
        print(wf)
    
        ts, ws = wf.trace(dt=0.001)
        ti, wi = wf[:]
    
        print(ws)
    
        pl.figure()
        pl.plot(ts,ws,'.')
        pl.plot(ti,wi,'ro')
        pl.xlabel('time [s]')
        pl.ylabel('waveform [A.U.]')
        pl.grid()
        pl.show()
        
    if True:
        myWF = WaveForm(dt = 0.001,
                        prolog = [(0, 0), 
                                  (0.3, 0), 
                                  (0.5, 1.0),
                                  (0.6, 2.0)],
                        Nrepeat = 4, 
                        repeat = [(0.0, 2.0),
                                  (0.5, 3.0)],
                        epilog = [(0.0, 1.0), 
                                  (1.0, 0.0)]
                       )
        
        if True:
            myWF.edit()
            
        else:
            myWF = WaveForm(**WF_Edit(wf=myWF.asdict()).result)
                
        print(pformat(myWF.asdict()))
