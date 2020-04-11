# -*- coding: utf-8 -*- 
""" 
This file creates Graphs based on Policy and Value Functions 
 
@author: Fabio 
""" 
 
import numpy as np 
import dill as pickle 
import matplotlib.pyplot as plt  
from matplotlib.pyplot import plot, draw, show 
import matplotlib.backends.backend_pdf 
#from ren_mar_pareto import v_mar_igrid 
#from ren_mar_pareto import v_ren_new 
#from ren_mar_alt import v_mar_igrid 
#from ren_mar_alt import v_ren_new 
from marriage import v_mar_igrid 
from renegotiation_unilateral import v_ren_uni
from renegotiation_bilateral import v_ren_bil

 
def v_reshape(setup,desc,field,V_by_t,Tmax): 
    # this reshapes V into many-dimensional object 
    if desc == "Couple, C" or desc == "Couple, M": 
         
        v0 = V_by_t[0][desc][field] 
        shp_v0 = v0.shape 
         
        tht_shape = (v0.ndim>2)*(shp_v0[-1],) # can be empty if no theta dimension 
         
        shape = (shp_v0[0],setup.pars['n_zf_t'][0],setup.pars['n_zm_t'][0],setup.pars['n_psi_t'][0]) + tht_shape + (Tmax,) 
        out = np.empty(shape,v0.dtype)         
        for t in range(Tmax): 
            vin = V_by_t[t][desc][field] 
            out[...,t] = vin.reshape(shape[:-1]) 
    else: 
        shape = V_by_t[0][desc][field].shape + (Tmax,) 
        out = np.empty(shape,V_by_t[0][desc][field].dtype) 
        for t in range(Tmax): 
            vin = V_by_t[t][desc][field] 
            out[...,t] = vin 
     
    return out 
             
             
 
 
 
 
def graphs(mdl,ai,zfi,zmi,psii,ti,thi): 
    # Import Value Funcrtion previously saved on File 
    #with open('name_model.pkl', 'rb') as file: 
    #    (Packed,dec) = pickle.load(file) 
         
    Packed = mdl.V 
    setup = mdl.setup 
    dec = mdl.decisions 
     
    ################################################ 
    # Unpack Stuff to Make it easier to Visualize 
    ################################################ 
    T = setup.pars['Tret'] 
    agrid = setup.agrid_c 
    agrids = setup.agrid_s 
    psig = setup.exogrid.psi_t[ti] 
    vtoutf=np.zeros([T,len(agrids),len(psig)]) 
    thetf=np.zeros([T,len(agrids),len(psig)]) 
    thetf_c=np.zeros([T,len(agrids),len(psig)]) 
    vtoutm=np.zeros([T,len(agrids),len(psig)]) 
    vtoutf_c=np.zeros([T,len(agrids),len(psig)]) 
    vtoutm_c=np.zeros([T,len(agrids),len(psig)]) 
    inds=np.zeros(len(psig)) 
     
     
    # TODO: vectorize this part too (I do not understand what exactly it does...) 
   
    # Here I get the renegotiated values 
    for t in range(T): 
         
        npsi = setup.pars['n_psi_t'][t+1] 
        inds = setup.all_indices(t+1,(zfi*np.ones(npsi,dtype=np.int16),zmi*np.ones(npsi,dtype=np.int16),np.arange(npsi,dtype=np.int16)))[0] 
         
        # cohabitation 
         
        ai_a = ai*np.ones_like(setup.agrid_s,dtype=np.int32) # these are assets of potential partner 
         
        resc = v_mar_igrid(setup,t+1,Packed[t],ai_a,inds,female=True,marriage=False) 
        (vf_c,vm_c), nbs_c, decc, tht_c = resc['Values'], resc['NBS'], resc['Decision'], resc['theta'] 
         
        tcv=setup.thetagrid_fine[tht_c] 
        tcv[tht_c==-1]=None 
         
        # marriage 
        resm = v_mar_igrid(setup,t,Packed[t],ai_a,inds,female=True,marriage=True) 
        (vf_m,vm_m), nbs_m, decm, tht_m = resm['Values'], resm['NBS'], resm['Decision'], resm['theta'] 
         
         
        tcm=setup.thetagrid_fine[tht_m] 
        tcm[tht_m==-1]=None 
         
       
        #Cohabitation-Marriage Choice  
        i_mar = (nbs_m>=nbs_c)  
             
        vout_ft = i_mar*vf_m + (1.0-i_mar)*vf_c 
        thet_ft = i_mar*tcm + (1.0-i_mar)*tcv 
        vout_mt = i_mar*vm_m + (1.0-i_mar)*vm_c 
  
         
        vtoutf[t,:,:]=vout_ft 
        vtoutf_c[t,:,:]=vf_c 
        thetf[t,:,:]=thet_ft 
        thetf_c[t,:,:]=tcv 
        vtoutm[t,:,:]=vout_mt 
        vtoutm_c[t,:,:]=vm_c 
         
         
    #Renegotiated Value 
         
    nex = setup.all_indices(max(ti-1,0),(zfi,zmi,psii))[0] 
    inde=setup.theta_orig_on_fine[thi] 
     
    # if ti = 0 it creates an object that was not used for the solutions,  
    # as V in v_ren_new is the next period value function. ti-1 should be here. 
    V_ren_c = v_ren_uni(setup,Packed[ti],False,ti-1,return_extra=True)[1]['Values'][0][np.arange(agrid.size),nex,inde] 
    
    v_ren_mar = v_ren_uni if setup.div_costs.unilateral_divorce else v_ren_bil
    
    V_ren_m = v_ren_mar(setup,Packed[ti],True,ti-1,return_extra=True)[1]['Values'][0][np.arange(agrid.size),nex,inde] 
    
        
    #Divorced Women and Men 
     
    
    # this thing assembles values of divorce / separation
    
    
    vals = [{'Couple, M': 
            v_ren_mar(setup,Packed[t],True,t-1,return_vdiv_only=True), 
            'Couple, C': 
            v_ren_uni(setup,Packed[t],False,t-1,return_vdiv_only=True), 
           } 
            for t in range(T)] 
        
     
    Vf_div = v_reshape(setup,'Couple, M','Value of Divorce, female',vals,T)[...,0,:] 
    Vm_div = v_reshape(setup,'Couple, M','Value of Divorce, male',vals,T)[...,0,:] 
     
     
    # I take index 0 as ipsi does not matter for this 
     
    #Single Women 
             
    Vfs, cfs, sfs = [v_reshape(setup,'Female, single',f,Packed,T) 
                        for f in ['V','c','s']] 
     
     
    Vms, cms, sms = [v_reshape(setup,'Male, single',f,Packed,T) 
                        for f in ['V','c','s']] 
                      
    #Couples: Marriage+Cohabitation 
     
    ithetam_R = v_reshape(setup,'Couple, M','thetas',dec,T-1) 
     
    thetam_R = setup.thetagrid_fine[ithetam_R] 
    thetam_R[ithetam_R==-1] = None 
     
    ithetac_R = v_reshape(setup,'Couple, C','thetas',dec,T-1) 
    thetac_R = setup.thetagrid_fine[ithetac_R] 
    thetac_R[ithetac_R==-1] = None 
     
     
    Vm, Vmm, Vfm, cm, sm, flsm = [v_reshape(setup,'Couple, M',f,Packed,T) 
                                    for f in ['V','VM','VF','c','s','fls']] 
    Vc, Vmc, Vfc, cc, sc, flsc = [v_reshape(setup,'Couple, C',f,Packed,T) 
                                    for f in ['V','VM','VF','c','s','fls']] 
     
     
 
     
    ######################################### 
    # Additional Variables needed for graphs 
    ######################################### 
     
    #Account for Single-Marriage and Single-Cohabit thresholds 
    trem=np.array([100.0]) 
    trec=np.array([100.0]) 
    for i in reversed(range(len(psig))): 
        if(not np.isnan(thetf[ti,ai,i])): 
            trem=psig[i] 
        if(not np.isnan(thetf_c[ti,ai,i])): 
            trec=psig[i] 
             
    tre=min(trem,trec) 
    if tre>50.0: 
        tre=max(psig) 
     
     
    ##################################### 
    ################################ 
    ## Actually Construct graphs 
    ################################ 
    #################################### 
     
    #Save the graphs 
    pdf = matplotlib.backends.backend_pdf.PdfPages("policy_graphs.pdf") 
     
    ########################################## 
    # Value Functions wrt Love 
    ##########################################  
    fig = plt.figure() 
    f1=fig.add_subplot(2,1,1) 
    plt.plot(psig, Vm[ai,zfi,zmi,0:len(psig),thi,ti],'k',markersize=6, label='Couple Marriage') 
    #plt.plot(psig, Vc[ai,zfi,zmi,0:len(psig),thi,ti],'k',markersize=6, label='Couple Cohabitation') 
    plt.plot(psig, Vmm[ai,zfi,zmi,0:len(psig),thi,ti],'bo',markersize=6, label='Man, Marriage') 
    plt.plot(psig, Vmc[ai,zfi,zmi,0:len(psig),thi,ti],'b',linewidth=0.4, label='Man, Cohabitation') 
    plt.plot(psig, Vfc[ai,zfi,zmi,0:len(psig),thi,ti],'r',linewidth=0.4, label='Women, Cohabitation') 
    plt.plot(psig, Vfm[ai,zfi,zmi,0:len(psig),thi,ti],'r*',markersize=6,label='Women, Marriage') 
    plt.axvline(x=tre, color='b', linestyle='--', label='Treshold Single-Couple') 
    #plt.axvline(x=treb, color='b', linestyle='--', label='Tresh Bilateral') 
    plt.xlabel('Love') 
    plt.ylabel('Utility') 
    #plt.title('Utility  Divorce costs: men=0.5, women=0.5') 
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), 
                  fancybox=True, shadow=True, ncol=3, fontsize='x-small') 
     
    ########################################## 
    # FLS wrt Love 
    ##########################################  
    fig = plt.figure() 
    f2=fig.add_subplot(2,1,1) 
    graphc=[None] * len(psig) 
    graphm=[None] * len(psig) 
    for i in range(len(psig)): 
        
        graphc[i]=setup.ls_levels[flsc[ai,zfi,zmi,i,thi,ti]] 
        graphm[i]=setup.ls_levels[flsm[ai,zfi,zmi,i,thi,ti]] 
 
    plt.plot(psig, graphm,'k',markersize=6, label='Marriage') 
    plt.plot(psig, graphc,'r*',markersize=6,label='Cohabitation') 
    plt.axvline(x=tre, color='b', linestyle='--', label='Treshold Single-Couple') 
    #plt.axvline(x=treb, color='b', linestyle='--', label='Tresh Bilateral') 
    plt.xlabel('Love') 
    plt.ylabel('FLS') 
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), 
                  fancybox=True, shadow=True, ncol=3, fontsize='x-small') 
     
    ########################################## 
    # FLS wrt female earnings 
    ##########################################  
    fig = plt.figure() 
    f3=fig.add_subplot(2,1,1) 
    graphc=[None] * setup.pars['n_zf_t'][ti] 
    graphm=[None] * setup.pars['n_zf_t'][ti] 
     
    for i in range(setup.pars['n_zf_t'][ti]): 
        graphc[i]=setup.ls_levels[flsc[ai,i,zmi,psii,thi,ti]] 
        graphm[i]=setup.ls_levels[flsm[ai,i,zmi,psii,thi,ti]] 
     
    plt.plot(range(setup.pars['n_zf_t'][ti]),graphm,'k',markersize=6, label='Marriage') 
    plt.plot(range(setup.pars['n_zf_t'][ti]), graphc,'r*',markersize=6,label='Cohabitation') 
    #plt.axvline(x=treb, color='b', linestyle='--', label='Tresh Bilateral') 
    plt.xlabel('Female Productivity') 
    plt.ylabel('FLS') 
    #plt.title('Utility  Divorce costs: men=0.5, women=0.5') 
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), 
                  fancybox=True, shadow=True, ncol=2, fontsize='x-small') 
     
    ########################################## 
    # FLS wrt theta 
    ##########################################  
    fig = plt.figure() 
    f4=fig.add_subplot(2,1,1) 
    graphc=[None] * len(setup.thetagrid) 
    graphm=[None] * len(setup.thetagrid) 
     
    for i in range(len(setup.thetagrid)): 
        graphc[i]=setup.ls_levels[flsc[ai,zfi,zmi,psii,i,ti]] 
        graphm[i]=setup.ls_levels[flsm[ai,zfi,zmi,psii,i,ti]] 
     
    plt.plot(setup.thetagrid,graphm,'k',markersize=6, label='Marriage') 
    plt.plot(setup.thetagrid, graphc,'r*',markersize=6,label='Cohabitation') 
    #plt.axvline(x=treb, color='b', linestyle='--', label='Tresh Bilateral') 
    plt.xlabel('Theta') 
    plt.ylabel('FLS') 
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), 
                  fancybox=True, shadow=True, ncol=2, fontsize='x-small') 
     
    ########################################## 
    # Surplues of Marriage wrt Cohabitation, Love grid 
    ########################################## 
     
    #Generate Marriage Surplus wrt cohabitation + Value Functions 
    surpM = [None] * len(psig) 
    surpW = [None] * len(psig) 
    
    for i in range(len(psig)): 
        surpM[i]=max(Vmm[ai,zfi,zmi,i,thi,ti]-Vmc[ai,zfi,zmi,i,thi,ti],0.0) 
        surpW[i]=max(Vfm[ai,zfi,zmi,i,thi,ti]-Vfc[ai,zfi,zmi,i,thi,ti],0.0) 
 
     
    #Graph for the Surplus 
    zero = np.array([0.0] * psig) 
    fig = plt.figure() 
    f5 = fig.add_subplot(2,1,1) 
    plt.plot(psig, zero,'k',linewidth=1) 
    plt.plot(psig, surpM,'b',linewidth=1.5, label='Man') 
    plt.plot(psig, surpW,'r',linewidth=1.5, label='Women') 
    plt.axvline(x=tre, color='b', linestyle='--', label='Treshold Single-Couple') 
    plt.ylim(-0.1*max(max(surpM),max(surpW),max(surpM),max(surpW)),1.1*max(max(surpM),max(surpW))) 
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), 
                  fancybox=True, shadow=True, ncol=3, fontsize='x-small') 
    plt.xlabel('Love') 
    plt.ylabel('Marriage Surplus wrt Cohab.') 
   
     
     
    ########################################## 
    # Value Function and Assets 
    ##########################################  
    fig = plt.figure() 
    f6=fig.add_subplot(2,1,1) 
    plt.plot(agrid, Vmm[0:len(agrid),zfi,zmi,psii,thi,ti],'bo',markersize=6,markevery=5, label='Man, Marriage') 
    plt.plot(agrid, Vmc[0:len(agrid),zfi,zmi,psii,thi,ti],'b',linewidth=0.4, label='Man, Cohabitation') 
    plt.plot(agrid, Vfc[0:len(agrid),zfi,zmi,psii,thi,ti],'r',linewidth=0.4, label='Women, Cohabitation') 
    plt.plot(agrid, Vfm[0:len(agrid),zfi,zmi,psii,thi,ti],'r*',markersize=6,markevery=5,label='Women, Marriage') 
    #plt.axvline(x=treb, color='b', linestyle='--', label='Tresh Bilateral') 
    plt.ylabel('Utility') 
    plt.xlabel('Assets') 
    #plt.title('Utility  Divorce costs: men=0.5, women=0.5') 
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), 
                  fancybox=True, shadow=True, ncol=4, fontsize='x-small') 
     
    ########################################## 
    # Surplues of Marriage wrt Cohabitation, Asset 
    ########################################## 
     
    #Generate Marriage Surplus wrt cohabitation + Value Functions 
    surpM = [None] * len(agrid) 
    surpW = [None] * len(agrid) 
    
    for i in range(len(agrid)): 
        surpM[i]=max(Vmm[i,zfi,zmi,psii,thi,ti]-Vmc[i,zfi,zmi,psii,thi,ti],0.0) 
        surpW[i]=max(Vfm[i,zfi,zmi,psii,thi,ti]-Vfc[i,zfi,zmi,psii,thi,ti],0.0) 
 
     
    #Graph for the Surplus 
    zero = np.array([0.0] * agrid) 
    fig = plt.figure() 
    f7 = fig.add_subplot(2,1,1) 
    plt.plot(agrid, zero,'k',linewidth=1) 
    plt.plot(agrid, surpM,'b',linewidth=1.5, label='Man') 
    plt.plot(agrid, surpW,'r',linewidth=1.5, label='Women') 
    #plt.axvline(x=treb, color='b', label='Tresh Bilateral') 
    #plt.axvline(x=treu, color='r', linestyle='--', label='Tresh Unilateral') 
    plt.ylim(-0.1*max(max(surpM),max(surpW),max(surpM),max(surpW))-0.0001,1.1*max(max(surpM),max(surpW))+0.0001) 
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), 
                  fancybox=True, shadow=True, ncol=3, fontsize='x-small') 
    plt.xlabel('Assets') 
    plt.ylabel('Marriage Surplus wrt Cohab.') 
      
      
    ########################################## 
    # Vf and ASSETS 
    ##########################################  
    fig = plt.figure() 
    f8=fig.add_subplot(2,1,1) 
    #plt.plot(agrid, Vm[0:len(agrid),zfi,zmi,psii,thi,ti],'bo',markersize=4, label='Before Ren M') 
    #plt.plot(agrid, Vc[0:len(agrid),zfi,zmi,psii,thi,ti],'r*',markersize=2,label='Before Ren C') 
    plt.plot(agrid, V_ren_c,'y', markersize=4,label='After Ren C') 
    plt.plot(agrid, V_ren_m,'k', linestyle='--',markersize=4, label='After Ren M') 
    #plt.plot(agrid, Vm_div[0:len(agrid),zfi,zmi,ti],'b',markersize=2, label='Male Divorce')  
    plt.plot(agrid, setup.thetagrid[thi]*Vf_div[0:len(agrid),zfi,zmi,ti]+(1-setup.thetagrid[thi])*Vm_div[0:len(agrid),zfi,zmi,ti],'r', linestyle='--',markersize=2,label='Female Divorce')  
    plt.ylabel('Utility') 
    plt.xlabel('Assets') 
    #plt.title('Utility  Divorce costs: men=0.5, women=0.5') 
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), 
                  fancybox=True, shadow=True, ncol=3, fontsize='x-small') 
     
    ########################################## 
    # Consumption and Assets 
    ##########################################  
    fig = plt.figure() 
    f9=fig.add_subplot(2,1,1) 
    plt.plot(agrid, cm[0:len(agrid),zfi,zmi,psii,thi,ti],'k',markevery=1, label='Marriage') 
    plt.plot(agrid, cc[0:len(agrid),zfi,zmi,psii,thi,ti],'r',linestyle='--',markevery=1, label='Cohabitation') 
    #plt.plot(agrid, sc[0:len(agrid),zfi,zmi,psii,thi,ti],'k',linewidth=2.0,linestyle='--', label='Cohabitation') 
    #plt.plot(agrids, cms[0:len(agrids),zmi,ti],'b',linewidth=2.0,label='Men, Single') 
    #plt.plot(agrids, cfs[0:len(agrids),zfi,ti],'r',linewidth=2.0,linestyle='--', label='Women, Single') 
    #plt.axvline(x=treb, color='b', linestyle='--', label='Tresh Bilateral') 
    plt.ylabel('Consumption') 
    plt.xlabel('Assets') 
    #plt.title('Utility  Divorce costs: men=0.5, women=0.5') 
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), 
                  fancybox=True, shadow=True, ncol=2, fontsize='x-small') 
     
     
    ########################################## 
    # Savings and Assets 
    ##########################################  
    fig = plt.figure() 
    f10=fig.add_subplot(2,1,1) 
    plt.plot(agrid, sm[0:len(agrid),zfi,zmi,psii,thi,ti],'ko',markersize=6,markevery=1, label='Marriage') 
    plt.plot(agrid, sc[0:len(agrid),zfi,zmi,psii,thi,ti],'r*',markersize=6,markevery=1, label='Cohabitation') 
    #plt.plot(agrid, agrid,'k',linewidth=1.0,linestyle='--') 
    #plt.plot(agrids, sms[0:len(agrids),zmi,ti],'b',linewidth=2.0,label='Men, Single') 
    #plt.plot(agrids, sfs[0:len(agrids),zfi,ti],'r',linewidth=2.0,linestyle='--', label='Women, Single') 
    #plt.axvline(x=treb, color='b', linestyle='--', label='Tresh Bilateral') 
    plt.ylabel('Savings') 
    plt.xlabel('Assets') 
    #plt.title('Utility  Divorce costs: men=0.5, women=0.5') 
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), 
                  fancybox=True, shadow=True, ncol=2, fontsize='x-small') 
    #print(111,cms[0:len(agrid),zmi,ti]) 
     
    ########################################## 
    # Value Function and Pareto Weights 
    ##########################################  
    fig = plt.figure() 
    f11=fig.add_subplot(2,1,1) 
    plt.plot(setup.thetagrid, Vmm[ai,zfi,zmi,psii,0:len(setup.thetagrid),ti],'bo',markersize=6, label='Man, Marriage') 
    plt.plot(setup.thetagrid, Vmc[ai,zfi,zmi,psii,0:len(setup.thetagrid),ti],'b',linewidth=0.4, label='Man, Cohabitation') 
    plt.plot(setup.thetagrid, Vfc[ai,zfi,zmi,psii,0:len(setup.thetagrid),ti],'r',linewidth=0.4, label='Women, Cohabitation') 
    plt.plot(setup.thetagrid, Vfm[ai,zfi,zmi,psii,0:len(setup.thetagrid),ti],'r*',markersize=6,label='Women, Marriage') 
    #plt.axvline(x=treb, color='b', linestyle='--', label='Tresh Bilateral') 
    plt.ylabel('Utility') 
    plt.xlabel('Pareto Weight-Women') 
    #plt.title('Utility  Divorce costs: men=0.5, women=0.5') 
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), 
                  fancybox=True, shadow=True, ncol=4, fontsize='x-small') 
     
    ########################################### 
    # Value of Marriage-Cohabitation over time 
    ########################################### 
    #Generate Marriage Surplus wrt cohabitation + Value Functions 
    surpM = [None] * T 
    surpW = [None] * T 
    
    for i in range(T): 
        surpM[i]=max(Vmm[ai,zfi,zmi,psii,thi,i]-Vmc[ai,zfi,zmi,psii,thi,i],0.0) 
        surpW[i]=max(Vfm[ai,zfi,zmi,psii,thi,i]-Vfc[ai,zfi,zmi,psii,thi,i],0.0) 
 
     
    #Graph for the Surplus 
    zero = np.array([0.0] * np.array(range(T))) 
    fig = plt.figure() 
    f12 = fig.add_subplot(2,1,1) 
    plt.plot(np.array(range(T)), zero,'k',linewidth=1) 
    plt.plot(np.array(range(T)), surpM,'b',linewidth=1.5, label='Man') 
    plt.plot(np.array(range(T)), surpW,'r',linewidth=1.5, label='Women') 
    #plt.axvline(x=treb, color='b', label='Tresh Bilateral') 
    #plt.axvline(x=treu, color='r', linestyle='--', label='Tresh Unilateral') 
    plt.ylim(-0.1*max(max(surpM),max(surpW),max(surpM)-0.0001,max(surpW)),1.1*max(max(surpM),max(surpW))+0.0001) 
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), 
                  fancybox=True, shadow=True, ncol=3, fontsize='x-small') 
    plt.xlabel('Time') 
    plt.ylabel('Marriage Surplus wrt Cohab.') 
     
    ########################################## 
    # Rebargained Surplus 
    ##########################################  
     
    #Generate Marriage Surplus wrt cohabitation + Value Functions 
    surpM = [None] * len(psig) 
    surpW = [None] * len(psig) 
    
    for i in range(len(psig)): 
        surpM[i]=max(vtoutm[ti,ai,i]-vtoutm_c[ti,ai,i],0.0) 
        surpW[i]=max(vtoutf[ti,ai,i]-vtoutf_c[ti,ai,i],0.0) 
 
     
    #Graph for the Surplus 
    zero = np.array([0.0] * psig) 
    fig = plt.figure() 
    f13 = fig.add_subplot(2,1,1) 
    plt.plot(psig, zero,'k',linewidth=1) 
    plt.plot(psig, surpM,'b',linewidth=1.5, label='Man') 
    plt.plot(psig, surpW,'r',linewidth=1.5, label='Women') 
    plt.axvline(x=tre, color='b', linestyle='--', label='Treshold Single-Couple') 
    #plt.axvline(x=treb, color='b', label='Tresh Bilateral') 
    #plt.axvline(x=treu, color='r', linestyle='--', label='Tresh Unilateral') 
    plt.ylim(-0.1*max(max(surpM),max(surpW),max(surpM),max(surpW)),1.1*max(max(surpM),max(surpW))) 
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), 
                  fancybox=True, shadow=True, ncol=3, fontsize='x-small') 
    plt.xlabel('Love') 
    plt.ylabel('Marriage Surplus wrt Cohab.') 
    
     
    ########################################## 
    # Initial thetas 
    ##########################################  
     
    #Generate Marriage Surplus wrt cohabitation + Value Functions 
    surpM = [None] * len(psig) 
    surpW = [None] * len(psig) 
    
    for i in range(len(psig)): 
        surpM[i]=max(vtoutm[ti,ai,i]-vtoutm_c[ti,ai,i],0.0) 
        surpW[i]=max(vtoutf[ti,ai,i]-vtoutf_c[ti,ai,i],0.0) 
 
     
    #Graph for the Surplus 
    zero = np.array([0.0] * psig) 
    fig = plt.figure() 
    f14 = fig.add_subplot(2,1,1) 
    plt.plot(psig, zero,'k',linewidth=1) 
    plt.plot(psig,  thetf[ti,ai,0:len(psig)],'b',linewidth=1.5, label='Theta Marriage') 
    plt.plot(psig,  thetf_c[ti,ai,0:len(psig)],'r', linestyle='--',linewidth=1.5, label='Theta Cohabitation') 
    plt.axvline(x=tre, color='k', linestyle='--', label='Treshold Single-Couple') 
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), 
                  fancybox=True, shadow=True, ncol=3, fontsize='x-small') 
    plt.xlabel('Love') 
    plt.ylabel('Theta') 
    
     
    ########################################## 
    # Renegotiated Thetas-Possible Split 
    ##########################################  
    zero = np.array([0.0] * psig) 
    fig = plt.figure() 
    f15 = fig.add_subplot(2,1,1) 
    plt.plot(psig, zero,'k',linewidth=1) 
    plt.plot(psig,  thetam_R[ai,zfi,zmi,0:len(psig),0,ti],'b',linewidth=1.5, label='Theta Marriage') 
    plt.plot(psig,  thetac_R[ai,zfi,zmi,0:len(psig),0,ti],'r', linestyle='--',linewidth=1.5, label='Theta Cohabitation') 
    for j in range(0, len(setup.thetagrid_fine), 10):  
        plt.plot(psig,  thetam_R[ai,zfi,zmi,0:len(psig),j,ti],'b',linewidth=1.5) 
        plt.plot(psig,  thetac_R[ai,zfi,zmi,0:len(psig),j,ti],'r', linestyle='--',linewidth=1.5) 
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), 
                  fancybox=True, shadow=True, ncol=2, fontsize='x-small') 
    plt.xlabel('Love') 
    plt.ylabel('Theta') 
     
    ########################################## 
    # Thetas and Assets 
    ##########################################  
    zero = np.array([0.0] * psig) 
    fig = plt.figure() 
    f16 = fig.add_subplot(2,1,1) 
    #plt.plot(psig, zero,'k',linewidth=1) 
    plt.plot(setup.thetagrid,  sm[ai,zfi,zmi,psii,0:len(setup.thetagrid),ti],'b',linewidth=1.5, label='Savings Marriage') 
    #plt.plot(setup.thetagrid,  sc[ai,zfi,zmi,psii,ti,0:len(setup.thetagrid)],'r', linestyle='--',linewidth=1.5, label='Savings Cohabitation') 
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), 
                  fancybox=True, shadow=True, ncol=1, fontsize='x-small') 
    plt.xlabel('Theta') 
    plt.ylabel('Assets') 
     
     
    ########################################## 
    # Assets and Theta 
    ##########################################  
    zero = np.array([0.0] * psig) 
    fig = plt.figure() 
    f17 = fig.add_subplot(2,1,1) 
    #plt.plot(psig, zero,'k',linewidth=1) 
    plt.plot(agrid,  thetam_R[0:len(agrid),zfi,zmi,psii,0,ti],'b',linewidth=1.5, label='Theta Marriage') 
    plt.plot(agrid,  thetac_R[0:len(agrid),zfi,zmi,psii,0,ti],'r', linestyle='--',linewidth=1.5, label='Theta Cohabitation') 
    for j in range(0, len(setup.thetagrid_fine), 10):  
        plt.plot(agrid,  thetam_R[0:len(agrid),zfi,zmi,psii,j,ti],'b',linewidth=1.5) 
        plt.plot(agrid,  thetac_R[0:len(agrid),zfi,zmi,psii,j,ti],'r', linestyle='--',linewidth=1.5) 
    #plt.plot(setup.thetagrid,  sc[ai,zfi,zmi,psii,ti,0:len(setup.thetagrid)],'r', linestyle='--',linewidth=1.5, label='Savings Cohabitation') 
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), 
                  fancybox=True, shadow=True, ncol=2, fontsize='x-small') 
    plt.xlabel('Assets') 
    plt.ylabel('Thetas') 
     
    ########################################## 
    # Put graphs together 
    ##########################################  
    #show() 
    for fig in range(1, plt.gcf().number + 1): ## will open an empty extra figure :( 
        pdf.savefig( fig ) 
        
    pdf.close() 
    matplotlib.pyplot.close("all") 
   