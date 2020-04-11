# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 10:41:24 2020

@author: Fabio
"""

import numpy as np   
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

def welfare(mdl,agents):
    
    #Welafare of Agents in t=0 under the two different regimes  
    shocks=agents.iexo[:,0]
    isfemale=np.array(agents.is_female[:,0],dtype=bool)
    ismale=~isfemale
    Vf_bil=mdl[0].V[0]['Female, single']['V'][0,shocks[isfemale]]
    Vm_bil=mdl[0].V[0]['Male, single']['V'][0,shocks[ismale]]
    Vf_uni=mdl[1].V[0]['Female, single']['V'][0,shocks[isfemale]]
    Vm_uni=mdl[1].V[0]['Male, single']['V'][0,shocks[ismale]]
    changef=(Vf_bil-Vf_uni)/np.abs(Vf_bil)
    changem=(Vm_bil-Vm_uni)/np.abs(Vm_bil) 
    
    
    #Get utility from realizations
    #mdl[0].setup.u_part(1,1,1,0.5,0.0,the)
    
    

    
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
    