    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 19:14:08 2019

@author: Egor Kozlov
"""


if __name__ == '__main__':
    
    
    #Clean Memory
    
    try:
        from IPython import get_ipython
        get_ipython().magic('reset -f')
    except:
        pass
    
    #If on server set Display
    from platform import system
    
    if system() != 'Darwin' and system() != 'Windows':   
        import os
        os.environ['QT_QPA_PLATFORM']='offscreen'
 
    import numpy as np
    from model import Model
    from setup import DivorceCosts
    import time



  
    

            
    #Create grids of parameters
    x0 = np.exp(np.array([ -1.8603,-8.1430,-1.57934,0.25130,-0.4991]))
    print(x0)
    sigma_psi_g=np.linspace(x0[1]*0.5,x0[1]*1.5,3)
    sigma_psi_init_g=np.linspace(x0[2]*1.0,x0[2]*1.0,1)
    di_co_g=np.linspace(x0[0]*0.8,x0[0]*1.2,1)
    bila=np.array([False,True])
    
    #Set some initial parameters
    pmeet = min(x0[3],1.0)
    uls = x0[4]
    
    #Initialize the file with parameters

    import xlwt 
    from xlwt import Workbook 
   
    #File that gives some info on the go
    f = open("iterations","a").close()
    # Workbook is created 
    wb = Workbook() 
    
    # add_sheet is used to create sheet. 
    sheet1 = wb.add_sheet('Sheet 1', cell_overwrite_ok=True) 
    sheet1.write(0, 0, 'Unilateral Divorce')
    sheet1.write(0, 1, 'sigma_psi') 
    sheet1.write(0, 2, 'sigma_psi_init') 
    sheet1.write(0, 3, 'u_cost') 
    sheet1.write(0, 4, '% coh before ret')
    sheet1.write(0, 5, '% mar berfore ret')
    sheet1.write(0, 6, 'hazd[0]')
    sheet1.write(0, 7, 'hazm[0]')
    sheet1.write(0, 8, 'hazs[0]')
    sheet1.write(0, 9, 'flsm')
    sheet1.write(0, 10, 'flsc')
    sheet1.write(0, 11, '% coh mean')
    sheet1.write(0, 12, '% mar mean')
    
    
    
    
    row=0
    for i in range(len(sigma_psi_g)):
        for j in range(len(sigma_psi_init_g)):
            for k in range(len(di_co_g)):
                for h in bila:
                
                    
                    
                

                    row=row+1
                    
                    f = open("iterations.txt","w")
                    f.write('{}'.format(row))
                    dc = DivorceCosts(unilateral_divorce=h,assets_kept = 1.0,u_lost_m=di_co_g[k],u_lost_f=di_co_g[k],eq_split=0.0)
                    sc = DivorceCosts(unilateral_divorce=True,assets_kept = 1.0,u_lost_m=0.0,u_lost_f=0.0)
                    mdl = Model(iterator_name='default',
                                divorce_costs=dc,separation_costs=sc,sigma_psi=sigma_psi_g[i],
                                sigma_psi_init=sigma_psi_init_g[j],pmeet=pmeet,uls=uls)

                    mdl.solve_sim(simulate=True)
                    
                    Tret = mdl.setup.pars['Tret']
                    
                    #Write results on spreadsheet
                    sheet1.write(row, 0, '{}'.format(h)) 
                    sheet1.write(row, 1, '{}'.format(sigma_psi_g[i])) 
                    sheet1.write(row, 2, '{}'.format(sigma_psi_init_g[j])) 
                    sheet1.write(row, 3, '{}'.format(di_co_g[k])) 
                    sheet1.write(row, 4, '{}'.format(mdl.moments['share coh'][Tret-1])) 
                    sheet1.write(row, 5, '{}'.format(mdl.moments['share mar'][Tret-1])) 
                    sheet1.write(row, 6, '{}'.format(mdl.moments['hazard div'][0]))
                    sheet1.write(row, 7, '{}'.format(mdl.moments['hazard mar'][0])) 
                    sheet1.write(row, 8, '{}'.format(mdl.moments['hazard sep'][0]))
                    sheet1.write(row, 9, '{}'.format(np.mean(mdl.moments['flsm'][1:-1])))
                    sheet1.write(row, 10, '{}'.format(np.mean(mdl.moments['flsc'][1:-1])))
                    sheet1.write(row, 11, '{}'.format(np.mean(mdl.moments['share coh'][:Tret]))) 
                    sheet1.write(row, 12, '{}'.format(np.mean(mdl.moments['share mar'][:Tret]))) 
                
    timestr = time.strftime("%Y%m%d-%H-%M.xls")
    wb.save(timestr) 
    f.close()           

    
    
  


   
        


    
