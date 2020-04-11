# -*- coding: utf-8 -*-
"""
Decomposition of Welfare Effects

@author: Fabio
"""
import numpy as np
from gridvec import VecOnGrid
import matplotlib.backends.backend_pdf   


 
#For nice graphs with matplotlib do the following 
matplotlib.use("pgf") 
matplotlib.rcParams.update({ 
    "pgf.texsystem": "pdflatex", 
    'font.family': 'serif', 
    'font.size' : 11, 
    'text.usetex': True, 
    'pgf.rcfonts': False, 
}) 

def welf_dec(mdl,agents):
    
    ################################################
    #ANALYSIS OF OVERALL WELFARE BY DIVORCE REGIME
    ################################################
 
    shocks=agents.iexo[:,0]
    isfemale=np.array(agents.is_female[:,0],dtype=bool)
    ismale=~isfemale
    Vf_bil=mdl[0].V[0]['Female, single']['V'][0,shocks[isfemale]]
    Vm_bil=mdl[0].V[0]['Male, single']['V'][0,shocks[ismale]]
    Vf_uni=mdl[1].V[0]['Female, single']['V'][0,shocks[isfemale]]
    Vm_uni=mdl[1].V[0]['Male, single']['V'][0,shocks[ismale]]
 
    
    
    
    #Compute additional wealth to make them indifferent-Women
    for i in range(len(mdl[0].setup.agrid_c)):
        Vf_uni_comp=np.mean(mdl[1].V[0]['Female, single']['V'][i,shocks[isfemale]])
        
        if(Vf_uni_comp>np.mean(Vf_bil)):
            wf=(-abs(np.mean(Vf_bil))+abs(np.mean(mdl[1].V[0]['Female, single']['V'][i-1,shocks[isfemale]])))/abs(np.mean(mdl[1].V[0]['Female, single']['V'][i-1,shocks[isfemale]]))
            acomf=((1-wf)*mdl[0].setup.agrid_c[i-1]+(wf)*mdl[0].setup.agrid_c[i])*28331.93
            break
        
    #Compute additional wealth to make them indifferent-Women
    for i in range(len(mdl[0].setup.agrid_c)):
        Vm_uni_comp=np.mean(mdl[1].V[0]['Male, single']['V'][i,shocks[ismale]])
        
        if(Vm_uni_comp>np.mean(Vm_bil)):
            wf=(-abs(np.mean(Vm_bil))+abs(np.mean(mdl[1].V[0]['Male, single']['V'][i-1,shocks[ismale]])))/abs(np.mean(mdl[1].V[0]['Male, single']['V'][i-1,shocks[ismale]]))
            acomm=((1-wf)*mdl[0].setup.agrid_c[i-1]+(wf)*mdl[0].setup.agrid_c[i])*28331.93
            break
        
    #Get worse Value and set it to standard
    standard=max(acomm,acomf)
    acomma=acomm/standard
    acomfa=acomf/standard
    
    content = r'''\begin{tabular}{cccc}
    \hline\midrule
    \multicolumn{2}{c}{\textbf{Female}}& \multicolumn{2}{c}{\textbf{Male}}\\
    \cmidrule(l){1-2}\cmidrule(l){3-4}
     Mutual Consent & Unilateral Divorce & Mutual Consent & Unilateral Divorce\\
     \cmidrule(l){1-4}
    \multicolumn{4}{c}{\textit{Life-Time utilities in $t=0$}}\\[3ex]
     '''+str(round(np.mean(Vf_bil),2))+''' &'''+str(round(np.mean(Vf_uni),2))+''' &'''+str(round(np.mean(Vm_bil),2))+''' &'''+str(round(np.mean(Vm_uni),2))+''' \\\\
    \cmidrule(l){1-4}
    \multicolumn{4}{c}{\\textit{Welfare Losses with Unilateral Divorce}}\\\\[3ex]
    \multicolumn{2}{c}{\Chartgirls{'''+str(acomfa)+'''}}& \multicolumn{2}{c}{\Chartguys{'''+str(acomma)+'''}}\\\\[-0.15ex]
    \multicolumn{2}{c}{'''+str(round(acomf,2))+''' \$}& \multicolumn{2}{c}{'''+str(round(acomm,2))+''' \$}\\\\
    \hline\hline
    \end{tabular}
    '''
 
    #Save the File here
    f=open('welfare.tex','w+')
    f.write(content)
    f.close()
    
    
    
    ###############################################
    #WELFARE AND POLITICAL ECONOMY
    ##############################################
    changep=agents.policy_ind  
    
    #Get value function for Women and men
    expnd0 = lambda x : mdl[0].setup.v_thetagrid_fine.apply(x,axis=2)
    expnd1 = lambda x : mdl[1].setup.v_thetagrid_fine.apply(x,axis=2)
    #Women
    Vfc_bil=np.ones(changep.shape)*-10000000
    Vfc_uni=np.ones(changep.shape)*-10000000
    for t in range(1,mdl[0].setup.pars['T']):
        
        #Prepare Value Functions
        VFbilm=expnd0(mdl[0].V[t]['Couple, M']['VF'])       
        VFunim=expnd0(mdl[1].V[t]['Couple, M']['VF'])       
        VFbilc=expnd0(mdl[0].V[t]['Couple, C']['VF'])       
        VFunic=expnd0(mdl[1].V[t]['Couple, C']['VF'])
        
        change=(changep[:,t]==1) & (changep[:,t-1]==0) & (agents.is_female[:,t]==1)
        single=(change) & (agents.state[:,t]==0)
        marry=(change) & (agents.state[:,t]==2)
        coh=(change) & (agents.state[:,t]==3) 
        Vfc_bil[marry,t]=VFbilm[agents.iassets[marry,t],agents.iexo[marry,t],agents.itheta[marry,t]]
        Vfc_bil[coh,t]=VFbilc[agents.iassets[coh,t],agents.iexo[coh,t],agents.itheta[coh,t]]
        Vfc_bil[single,t]=mdl[0].V[t]['Female, single']['V'][agents.iassets[single,t],mdl[0].setup.all_indices(t,agents.iexo[single,t])[1]]
        Vfc_uni[marry,t]=VFunim[agents.iassets[marry,t],agents.iexo[marry,t],agents.itheta[marry,t]]
        Vfc_uni[coh,t]=VFunic[agents.iassets[coh,t],agents.iexo[coh,t],agents.itheta[coh,t]]
        Vfc_uni[single,t]=mdl[1].V[t]['Female, single']['V'][agents.iassets[single,t],mdl[0].setup.all_indices(t,agents.iexo[single,t])[1]]
    
    wh=np.where(Vfc_bil>-1000000)
    Vfcb=Vfc_bil[wh[0],wh[1]]
    Vfcu=Vfc_uni[wh[0],wh[1]]
    unipf=(Vfcu>=Vfcb)
    
    #Men
    Vmc_bil=np.ones(changep.shape)*-10000000
    Vmc_uni=np.ones(changep.shape)*-10000000
    for t in range(1,mdl[0].setup.pars['T']):
        
        VMbilm=expnd0(mdl[0].V[t]['Couple, M']['VM'])
        VMunim=expnd0(mdl[1].V[t]['Couple, M']['VM'])
        VMbilc=expnd0(mdl[0].V[t]['Couple, C']['VM'])
        VMunic=expnd0(mdl[1].V[t]['Couple, C']['VM'])
        
        change=(changep[:,t]==1) & (changep[:,t-1]==0) & (agents.is_female[:,t]==0)
        single=(change) & (agents.state[:,t]==1)
        marry=(change) & (agents.state[:,t]==2)
        coh=(change) & (agents.state[:,t]==3) 
        Vmc_bil[marry,t]=VMbilm[agents.iassets[marry,t],agents.iexo[marry,t],agents.itheta[marry,t]]
        Vmc_bil[coh,t]=VMbilc[agents.iassets[coh,t],agents.iexo[coh,t],agents.itheta[coh,t]]
        Vmc_bil[single,t]=mdl[0].V[t]['Male, single']['V'][agents.iassets[single,t],mdl[0].setup.all_indices(t,agents.iexo[single,t])[2]]
        Vmc_uni[marry,t]=VMunim[agents.iassets[marry,t],agents.iexo[marry,t],agents.itheta[marry,t]]
        Vmc_uni[coh,t]=VMunic[agents.iassets[coh,t],agents.iexo[coh,t],agents.itheta[coh,t]]
        Vmc_uni[single,t]=mdl[1].V[t]['Male, single']['V'][agents.iassets[single,t],mdl[0].setup.all_indices(t,agents.iexo[single,t])[2]]
    
    wh=np.where(Vmc_bil>-1000000)
    Vmcb=Vmc_bil[wh[0],wh[1]]
    Vmcu=Vmc_uni[wh[0],wh[1]]
    unipm=(Vmcu>=Vmcb)
    
    if len(unipm)>0:
        print("The % of men voting for unilateral is {:0.2f}, the % of women if {:0.2f}".format(np.mean(unipm*100),np.mean(unipf)*100))
    ###############################################
    #WELFARE DECOMPOSITION HERE
    ##############################################

    #Get some stuff from agents
    assets_t=mdl[0].setup.agrid_c[agents.iassets] # FIXME   
    iexo=agents.iexo   
    iexos=agents.iexos   
    state=agents.state   
    theta_t=mdl[0].setup.thetagrid_fine[agents.itheta]   
    setup = mdl[0].setup  
    female=agents.is_female
    cons=agents.c
    consx=agents.x
    labor=agents.ils_i
    shks = agents.shocks_single_iexo 
    psi_check=np.zeros(state.shape)
    shift_check=np.array((state==2),dtype=np.float32)
    single=np.array((state==0),dtype=bool)
    betag=mdl[0].setup.pars['beta_t'][0]**(np.linspace(1,len(state[0,:]),len(state[0,:]))-1)
    betam=np.reshape(np.repeat(betag,len(state[:,0])),(len(state[:,0]),len(betag)),order='F')
    
    #Fill psi and ushift here
    for i in range(len(state[0,:])):
        psi_check[:,i]=((setup.exogrid.psi_t[i][(setup.all_indices(i,iexo[:,i]))[3]])) 
    
    
    #For welfare
    cop_f=mdl[0].setup.u_part(cons,consx,labor,theta_t,psi_check,shift_check*mdl[0].setup.pars['u_shift_mar'])*betam
    s_f=mdl[0].setup.u_single_pub(cons,consx,labor)*betam
    combf=cop_f[0]
    combf[(state==0)]=s_f[(state==0)]
    sommaf=np.sum(combf[(female[:,0]==1),:],axis=1)
    EF=np.mean(sommaf)
    
    s_m=mdl[0].setup.u_single_pub(cons,consx,labor)*betam
    combm=cop_f[1]
    combm[(state==1)]=s_m[(state==1)]
    sommam=np.sum(combm[(female[:,0]==0),:],axis=1)
    EM=np.mean(sommam)
    
    
    #For comparison substituting with other regime
    statew=state.copy()
    statem=state.copy()
    combf1=combf.copy()
    combm1=combm.copy()
    womd=np.ones(combf.shape)*-1000
    mend=np.ones(combm.shape)*-1000
    womd1=np.ones(combf.shape)*-1000
    mend1=np.ones(combm.shape)*-1000
    costs = mdl[0].setup.div_costs
    
    #Get value function for Women and men
    expnd = lambda x : mdl[0].setup.v_thetagrid_fine.apply(x,axis=2)
    
    for t in range(1,mdl[0].setup.pars['Tren']-1):
        
        VF=expnd(mdl[1].V[t]['Couple, M']['VF'])
        VM=expnd(mdl[1].V[t]['Couple, M']['VM'])
       
        ########################
        #WOMEN HERE
        ########################
        
        #Get divorce states
        #
      
        if mdl[1].decisions[t-1]['Couple, M']['Decision'].ndim>2:

            divo=(mdl[1].decisions[t-1]['Couple, M']['Decision'][agents.iassetss[:,t],iexos[:,t],agents.itheta[:,t-1]]==False) & (statew[:,t-1]==2)
        else:

            divo=(mdl[1].decisions[t-1]['Couple, M']['Decision'][agents.iassetss[:,t],iexos[:,t]]==False) & (statew[:,t-1]==2)
            
        #divo=(statew[:,t]==0) & (statew[:,t-1]==2)
        divo1=(divo) & (agents.is_female[:,t]==1) 
        
        if np.any(divo1):

            subs=mdl[0].decisions[t-1]['Couple, M']['Divorce'][0][agents.iassetss[divo1,t],iexos[divo1,t]]#mdl[0].V[t]['Female, single']['V'][assets,iexo[divo1,t]]
            statew[divo1,t:]=-1
            combf1[divo1,t:]=0
            combf1[divo1,t]=betam[divo1,t]*subs
            
            #Get who divorce

            womd[divo1,t]=(mdl[1].decisions[t-1]['Couple, M']['Divorce'][0][agents.iassetss[divo1,t],iexos[divo1,t],mdl[1].setup.igridcoarse[agents.itheta[divo1,t-1]]]>VF[agents.iassets[divo1,t],iexos[divo1,t],agents.itheta[divo1,t-1]])
            mend[divo1,t]=(mdl[1].decisions[t-1]['Couple, M']['Divorce'][1][agents.iassetss[divo1,t],iexos[divo1,t],mdl[1].setup.igridcoarse[agents.itheta[divo1,t-1]]]>VM[agents.iassets[divo1,t],iexos[divo1,t],agents.itheta[divo1,t-1]])
            
        #    womd[divo1,t]=(mdl[1].decisions[t-1]['Couple, M']['Divorce'][0][agents.iassetss[divo1,t],iexos[divo1,t]]>VF[agents.iassets[divo1,t],iexos[divo1,t],agents.itheta[divo1,t-1]])
         #   mend[divo1,t]=(mdl[1].decisions[t-1]['Couple, M']['Divorce'][1][agents.iassetss[divo1,t],iexos[divo1,t]]>VM[agents.iassets[divo1,t],iexos[divo1,t],agents.itheta[divo1,t-1]])
            
        ########################
        #MEN HERE
        ########################
        
        #Get divorce states
        
        if mdl[1].decisions[t-1]['Couple, M']['Decision'].ndim>2:
            divom=(mdl[1].decisions[t-1]['Couple, M']['Decision'][agents.iassetss[:,t],iexos[:,t],agents.itheta[:,t-1]]==False) & (statem[:,t-1]==2)
        else:
            divom=(mdl[1].decisions[t-1]['Couple, M']['Decision'][agents.iassetss[:,t],iexos[:,t]]==False) & (statem[:,t-1]==2)
            
        #divom=(statem[:,t]==1) & (statem[:,t-1]==2)
        divom1=(divom) & (agents.is_female[:,t]==0) 
        
        if np.any(divom1):

            subm=mdl[0].decisions[t-1]['Couple, M']['Divorce'][1][agents.iassetss[divom1,t],iexos[divom1,t]]#mdl[0].V[t]['Male, single']['V'][assetm,iexo[divom1,t]]
            statem[divom1,t:]=-1
            combm1[divom1,t:]=0
            combm1[divom1,t]=betam[divom1,t]*subm
            
            #Get who divorce
  
            womd1[divom1,t]=(mdl[1].decisions[t-1]['Couple, M']['Divorce'][0][agents.iassetss[divom1,t],iexos[divom1,t],mdl[1].setup.igridcoarse[agents.itheta[divom1,t-1]]]>VF[agents.iassets[divom1,t],iexos[divom1,t],agents.itheta[divom1,t-1]])
            mend1[divom1,t]=(mdl[1].decisions[t-1]['Couple, M']['Divorce'][1][agents.iassetss[divom1,t],iexos[divom1,t],mdl[1].setup.igridcoarse[agents.itheta[divom1,t-1]]]>VM[agents.iassets[divom1,t],iexos[divom1,t],agents.itheta[divom1,t-1]])
            
        #   womd1[divom1,t]=(mdl[1].decisions[t-1]['Couple, M']['Divorce'][0][agents.iassetss[divom1,t],iexos[divom1,t]]>VF[agents.iassets[divom1,t],iexos[divom1,t],agents.itheta[divom1,t-1]])
        #    mend1[divom1,t]=(mdl[1].decisions[t-1]['Couple, M']['Divorce'][1][agents.iassetss[divom1,t],iexos[divom1,t]]>VM[agents.iassets[divom1,t],iexos[divom1,t],agents.itheta[divom1,t-1]])
            
        
 
    #Wrap up Results
    sommaf1=np.sum(combf1[(female[:,0]==1),:],axis=1)
    EF1=np.mean(sommaf1)
    sommam1=np.sum(combm1[(female[:,0]==0),:],axis=1)
    EM1=np.mean(sommam1)
    
    #Divorce, women side
    botha=(np.sum((mend==1) & (womd==1)))/(np.sum((mend>-1) & (womd>-1)))
    mena=(np.sum((mend==1) & (womd==0)))/(np.sum((mend>-1) & (womd>-1)))
    woma=(np.sum((mend==0) & (womd==1)))/(np.sum((mend>-1) & (womd>-1)))
    botha1=(np.sum((mend1==1) & (womd1==1)))/(np.sum((mend1>-1) & (womd1>-1)))
    mena1=(np.sum((mend1==1) & (womd1==0)))/(np.sum((mend1>-1) & (womd1>-1)))
    woma1=(np.sum((mend1==0) & (womd1==1)))/(np.sum((mend1>-1) & (womd1>-1)))
    
    print("The Welfare of Women with Bil is {:0.2f}, when only at divorce we apply Unid is {:0.2f}".format(EF,EF1))
    print("The Welfare of Men with Bil is {:0.2f}, when only at divorce we apply Unid is {:0.2f}".format(EM,EM1))
    print("Share divorces (fem. measured) where both agreed is {:0.2f}, men only agreed {:0.2f}, women only agreed {:0.2f}".format(botha,mena,woma))
    print("Share divorces (men measured) where both agreed is {:0.2f}, men only agreed {:0.2f}, women only agreed {:0.2f}".format(botha1,mena1,woma1))
    