#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aCreated on Tue Sep 17 19:14:08 2019
 
@author: Egor Kozlov
"""
 
 
 
 
 
if __name__ == '__main__':
     
    try:
        from IPython import get_ipython
        get_ipython().magic('reset -f')
    except:
        pass
 
 
from platform import system
     
import os
if system() != 'Darwin' and system() != 'Windows':      
    os.environ['QT_QPA_PLATFORM']='offscreen'
    
if system() == 'Darwin':
    os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin/'
   

import numpy as np
from residuals import mdl_resid
from data_moments3 import dat_moments
 
print('Hi!')
 
 
 
 
if __name__ == '__main__':
     
     
    #import warnings
    #warnings.filterwarnings("error")
    #For graphs later
    graphs=True
    #Build  data moments and pickle them
    #dat_moments(period=1,sampling_number=100,weighting=True,covariances=True,transform=1)
    
         
    #Initialize the file with parameters
    
    

    #Second
    #x0 = np.array([0.0,   0.06565744,  1.5,  0.2904853,   0.7371481,  0.018159483 - 0.6, -0.091977, 0.805955,0.1])
    #x0 = np.array([1.10511688 , 0.10725931,  3.6224206,   0.44856022 , 0.0472732 ,  0.02879032, -0.09039855,  1.23986084 , 0.10953983])
    x0 = np.array([0.3 , 0.04725931*2.996194651745017, 10/2.996194651745017,   0.25, 1.1 ,  0.0075-0.0, -0.09039855,  1.13986084 , 0.30953983*2.996194651745017])

    #1 1
    x0 = np.array([0.321094,0.167578,3.69922,0.214648,1.14922,0.00989453,-0.0951172,1.11156,0.959375])
    
    #1 2
    x0 = np.array([0.251807,0.228589,4.37695,0.183325,1.10332,0.00648584,-0.0843213,1.12707,0.395508])
    
    #Wisdom
    x0 = np.array([0.287148,0.43265,4.43904,0.197515,1.1088,0.0067326,-0.084137,1.1219,1.532568])
    x0 = np.array([0.44474121,  0.32749023,  7.0759082,   0.2302002,   1.09989648,  0.00966553,-0.06185889,  1.1963916,   1.37988281])
    x0 = np.array([0.44474121,  0.42749023*1.2,  27.0759082,   0.2302002,   0.003989648,  0.0,-0.06185889,  1.1963916,   0.77988281*1.2])
    x0 = np.array([0.44474121,  0.42749023*1.2,  0.0759082,   0.2302002,   0.000001,  0.0,-0.06185889,  1.1963916,   0.77988281*1.2])   
    x0 = np.array([0.8296875,   0.238125,    0.79570312,  0.59226563,  0.36492187,  0.01349219, -0.08658594,  1.03046875,  0.2703125])   
    x0 = np.array([0.23673339843750002, 0.19666015625,0.21979980468750004,0.4838623046875,0.1846240234375, -0.12382861328125,1.0734130859375,0.40615234375000003])
   
    #New way
    x0 = np.array([0.75822509765625,0.030224609375, 0.40762451171875,0.48373779296875, 0.45178955078125,-0.077394775390625,1.30259521484375, 0.32880859375000004])
    x0 = np.array([0.75822509765625,0.030224609375, 0.40762451171875,0.48373779296875, 0.15178955078125,-0.1177394775390625,1.0259521484375, 0.32880859375000004])
    x0 = np.array([0.40019165039062504,0.29312988281249996,0.9469348144531251, 0.889136962890625,0.08512316894531251,-0.00464412841796875, -0.0414232177734375,1.034232177734375, 0.5877197265625])   
    
    
    

    #Envolop
    #x0 = np.array([0.505842,0.328506,0.378367,0.477761,0.056823,-0.0074071,-0.0451931,0.943567,0.417432])

    #x0 = np.array([0.788213,0.537617,0.508721,0.193252,0.14578,-0.0258829,-0.048083,1.07827,0.607617])    

    #x0 = np.array([0.594072,0.61668,0.283018,0.213174,0.141116,-0.0163056,-0.109966,1.05679,0.95918])
    
    x0 = np.array([0.341689,0.325586,0.328994,0.60165,0.0190732,-0.00450908,-0.071376,0.883936,0.455273])


    #Local mimimizer
    #Ãƒâ€šÃ¢â‚¬Â¢x0 = np.array([0.62137078,  0.28477307,  0.95136571,  0.43995757,  0.12810155, -0.01046587, -0.06743287,  1.00750209,  0.46041764])
    
    #x0 = np.array([ 0.35757217,0.27315025,0.99360676,0.39012968,0.14677543,-0.004208126,-0.04573717,1.0229298,0.42787423])
    
    #x0 = np.array([ 0.61923624,0.27795393,0.80388534,0.4729621,0.098108704,-0.009852369,-0.12677989,1.0054418,0.55442649])
    
    x0 = np.array([    0.51368963,0.30499866,0.73090734,0.50559526,0.00000000000000026139676,-0.0072579687,-0.068374362,0.91598437,0.44187723])
    
     #x0 = np.array([0.62137078,  0.28477307,  0.95136571,  0.43995757,  0.12810155, -0.01046587, -0.06743287,  1.00750209,  0.46041764])

    x0 = np.array([0.7607666015625, 0.30711669921875, 1.0903320312499998, 0.48647705078125, 0.269822021484375, -0.0242556396484375,-0.14070556640624998, 1.06548583984375, 0.5080810546875])
    
    x0 = np.array([0.7607666015625, 0.30711669921875*1.3, 0.11503320312499998,  0.139822021484375, -0.01942556396484375,-0.09070556640624998, 1.043548583984375, 0.9080810546875*1.3])

    ########################
    #A Good one from envelop
    ##########################
    x0 = np.array([0.134814453125,0.3867431640625*1.1, 0.1560107421875, 0.3401025390625,0.0,-0.0328173828125,0.9808105468749999,0.688671875*1.1])
    #x0 = np.array([0.634814453125,0.2867431640625*1.5, 0.0960107421875, 0.09401025390625,-0.0012556982421875,-0.1328173828125,0.9808105468749999,0.688671875*1.5])

    #x0 = np.array([ 0.58955078,  0.26713867,  0.1909082 ,  0.08440527, -0.00552256, -0.14674805 , 0.99384766,  0.52421875])
    x0 = np.array([0.40532226562499996, 0.060185546875*1.4,0.147890625,  1.1058984375,-0.0277397460937499998,-0.218591796875000002, 2.7839257812500002, 0.9919921875*1.4])
    x0 = np.array([0.95532226562499996, 0.020185546875*1.4,2.147890625,  0.0001058984375,-3.577397460937499998,-0.218591796875000002, 2.7839257812500002, 0.9919921875*1.4])
    x0 = np.array([0.92532226562499996, 0.020185546875*1.1,2.147890625,  0.0001058984375,-3.577397460937499998,-0.218591796875000002, 2.7839257812500002, 0.9919921875*1.1])
    x0 = np.array([0.84532226562499996, 0.020185546875*1.1,0.547890625,  1.8058984375,-0.307397460937499998,-0.0218591796875000002, 2.7839257812500002, 0.9919921875*1.1])
    x0 = np.array([0.54532226562499996, 0.020185546875*1.1,1.547890625,  0.3058984375,-0.407397460937499998,-0.218591796875000002, 2.7839257812500002, 0.9919921875*1.1])
    #x0 = np.array([0.94532226562499996, 0.080185546875*1.1,0.8547890625,  0.1158984375,-2.47397460937499998,-0.118591796875000002, 3.5, 0.9919921875*1.1])
    
    x0 = np.array([0.054532226562499996, 0.0,0.30547890625,  0.0418984375,-0.2,-0.15, 0.75,0.9])
    x0 = np.array([0.054532226562499996, 0.0,0.30547890625,  150.0,-0.55,-0.25, 1.8,1.2])
    x0 = np.array([0.054532226562499996, 0.0,0.30547890625,  0.0418984375,-0.4,-0.25, 0.75,1.2])
    x0 = np.array([0.154532226562499996, 0.0,0.30547890625,  0.0118984375,-0.2,-0.25, 0.75,1.0])
    x0 = np.array([0.454532226562499996, 0.0,0.30547890625,  0.0118984375,0.0,-0.25, 0.75,1.0])
    x0 = np.array([0.4054532226562499996, 0.0,0.30547890625,  0.1,0.0,-0.25, 1.5,1.3])
    x0 = np.array([0.2054532226562499996, 0.0,0.2550547890625,  19.1,0.0,-0.15, 1.5,1.4])
    x0 = np.array([0.4054532226562499996, 0.0,0.30547890625,  0.1,0.0,-0.25, 1.5,1.3])
    
    
    ############
    #FLS 0.75
    #################
    
    #Kind of work
    x0 = np.array([0.3054532226562499996, 0.0,0.40547890625,  0.0918984375,0.0,-0.15, 0.75,1.3])
    #x0 = np.array([0.3054532226562499996, 0.0,0.40547890625,  0.0918984375,0.0,-0.15, 0.75,1.3])
    
    
    x0 = np.array([0.1054532226562499996, 0.0,0.170547890625,  0.02718984375,0.0,-0.1, 0.75,1.3])
    #0.26 -okish many d
    #x0 = np.array([0.2205, 0.0,0.5547890625,  0.0918984375,0.0,-0.15, 0.75,1.9])
   
    #Not really nice
    x0 = np.array([0.2054532226562499996, 0.0,0.2550547890625,   19.1,0.0,-0.15, 1.5,1.4])
    x0 = np.array([0.3054532226562499996, 0.0,0.1550547890625,  10.1,0.0, -0.15, 1.5,1.8])
    x0 = np.array([0.8054532226562499996, 0.0,0.285250547890625,  18.1,-0.0, -0.05, 1.5,1.8])
    x0 = np.array([0.254532226562499996, 0.0,0.205250547890625,  12.1,-0.0, -0.15, 1.5,1.8])
    x0 = np.array([0.254532226562499996, 0.0,0.135250547890625,  7.1,-0.0, -0.1, 1.5,1.8])
    x0 = np.array([0.454532226562499996, 0.0,0.22250547890625,  7.1,-0.0, -0.1, 1.5,1.8])
    x0 = np.array([0.454532226562499996,0.22250547890625,7.1,1.8,0.2,0.8])
    
    
    
    
    
    #VARTHETA NEGOTIATION
    #x0 = np.array([0.1054532226562499996,0.15250547890625,7.1,0.8,0.4,1.0])
    #x0 = np.array([0.1054532226562499996,0.15250547890625,7.1,0.8,0.4,1.0])
    #x0 = np.array([0.154532226562499996,0.15250547890625,36670.1,0.8,0.4,1.0])
   
  
    #x0 = np.array([0.354532226562499996, 0.5,0.20250547890625,  28.1,0.0, -0.15, 1.5,0.0])
   
    
    #Ish...0.35
    #x0 = np.array([0.384532226562499996, 0.0,0.4550547890625,  34.1,0.0,-0.15, 1.5,1.4])
    
    #ok with 26
    #x0 = np.array([0.284532226562499996, 0.0,0.550547890625,  34.1,0.0,-0.15, 1.5,1.4])
    #x0 = np.array([0.384532226562499996, 0.0,0.450547890625,  34.1,0.0,-0.15, 1.5,1.4])
     
    #Cost 0.28 0.12
    #x0 = np.array([0.0, 0.0,0.4050547890625,  29.1,0.0,-0.15, 1.5,1.4])
  

    #No good    
    #x0 = np.array([0.304532226562499996, 0.0,0.2547890625,  16060250.0,-1.6,-0.15, 3.2,1.6])
   
   
    
    ##########ÃƒÆ’Ã‚Â 
    #FLS 1.0  
    #############
    
    #No very good
    #x0 = np.array([0.3054532226562499996, 0.0,0.305547890625,  0.0138984375,0.0,-0.25, 0.75,1.3])
    
    #Good bu work better with more nodes for income
    #x0 = np.array([0.4054532226562499996, 0.0,0.30547890625,  0.1,-0.0,-0.05, 1.5,1.3])
    #x0 = np.array([0.154532226562499996, 0.0,0.15547890625,  0.15,-0.0,-0.15, 1.5,1.3])
    #x0 = np.array([0.2054532226562499996, 0.0,0.2547890625,  0.18,-0.0,-0.15, 1.5,1.1])
    #x0 = np.array([0.6054532226562499996, 0.0,0.1547890625,  0.09,-0.0,-0.1, 1.5,1.8])
    
    #Kind of ok intercept 0.1
    #x0 = np.array([0.384532226562499996, 0.0,0.4040547890625,  0.15,0.0,-0.15, 1.5,1.3])
    
    #Below cost oÃƒÆ’Ã‚Â¬for edu 0.6- 0.2
    #x0 = np.array([0.0094054532226562499996, 0.0,0.399547890625,  0.1,0.0,-0.15, 1.5,1.3])
    
    
    #Server couple
    x0 = np.array([0.23115234375000002,0.2294921875,11.009765625,1.01328125, 0.13515625,0.5277343750000001])
    #x0 = np.array([0.283251953125,0.25732421875, 9.976367187500001,0.841015625, 0.280078125,0.7333984375])
    #x0 = np.array([0.290771484375,0.17822265625, 9.3837890625,1.133984375,0.349609375, 0.7251953125])
    
    #GOOD ONE: npsi 21, ntheta 11-7. dump 0.3 sig 1.15 alpha 0.53 drift 0.2
    x0 = np.array([0.22155838,  0.19626592 ,11.67366755,  1.35678889 , 0.19755867 , 0.71320478])

    x0 = np.array([0.22155838,  0.0449626592 ,11.67366755,  1.35678889 , 0.19755867 , 0.71320478])
    #x0 = np.array([0.22155838,  0.0949626592  ,11.67366755, 1.35678889 , 0.19755867 , 0.71320478])
    #x0 = np.array([ 0.45136718750000004,  0.05181640625, 10.157031250000001,1.19453125,0.20468750000000002,0.58515625])
    
    #Server New
    x0 = np.array([ 0.46476562499999996, 0.05021484375, 10.40234375,1.49609375,0.2259765625, 0.978515625])
    x0 = np.array([ 0.26476562499999996, 0.05021484375, 8.40234375,1.49609375,0.5259765625, 0.978515625])
    x0 = np.array([ 0.29476562499999996, 0.04021484375, 8.40234375,1.49609375,0.3259765625, 0.678515625])
    x0 = np.array([0.20014,0.0371797,6.7806,1.77938,0.252993,1])
    #x0 = np.array([0.19588,0.0386523,7.27884,1.65462,0.250358,0.946155])
    #x0 = np.array([0.210931,0.03417,7.023,1.81258,0.25,0.999809])
    x0 = np.array([0.35,0.0719456,11.2908,1.68981,0.250931,0.776382,-0.279974])
    #x0 = np.array([0.35,0.0719456,11.2908,1.68981,0.250931,0.776382,-0.2079974])
    
    x0 = np.array([0.18013916015625, 0.0557486572265625, 9.98309326171875,2.24427490234375,0.31585693359375,0.611737060546875, -0.20347900390625])   
    x0 = np.array([0.15781,0.0566984,10.0718,2.24088,0.306568,0.616984,-0.257216])
    x0 = np.array([0.15781,0.0566984,10.0718,2.24088,0.306568,0.616984,-0.257216])

    #More precition in difference in ever a relationship
    x0 = np.array([ 0.20566037607857718,0.05018690002348321,10.839082595363138,2.1648392222044324,0.20701390142645873,0.7826766550337498,-0.2442608336190132])
    #x0 = np.array([ 0.2435546875, 0.049931640625, 9.263671875, 2.35751953125, 0.20820312500000002, 0.59150390625,-0.26417968750000004])
    
    #More points variances only
    x0 = np.array([ 0.1996361485453203, 0.046518845619417735,11.653993209071075, 2.217967680807276, 0.20220524730055478, 0.7580296069266805, -0.2540148141508227])
    x0 = np.array([ 0.5400390625,0.051484375000000006,11.814453125,2.5647460937499997,0.19023437499999998,0.8525390625000001,-0.23953125])
    x0 = np.array([0.35035644531250004,0.05177490234375,13.4688720703125, 2.37093505859375, 0.200048828125, 0.891181884765625, -0.2835205078125])
 
    
    #Very best (3674
    #x0 = np.array([0.421814,    0.05033825, 11.71562626,  2.47427583,  0.18584936,  0.95, -0.23   ])
    
    #3574
    x0 = np.array([0.5178592893786579,0.052671250047870134, 13.239007798920266,  2.5253214623192,0.1935526148086114, 0.9934985581192299, -0.24524532498412388])
   
    #new onepk with correlation 0.57/-0.3 male/0.3 correlation
    x0 = np.array([0.3492804 ,  0.0531327 , 13.06629932 , 2.34502811,  0.19540266,  0.67443746, -0.26172938])
    x0 = np.array([ 0.3909996903412387, 0.06234279279670172, 13.781302221156782, 2.529437037986855, 0.17752406606704368,0.6121270497077264,-0.2876416092457871])
    x0 = np.array([ 0.37812003634042446, 0.06372179812978007, 14.396298058668338, 2.5572039874945087,0.16657396981365394,0.6114995678119386,-0.29709816007123])
    
    
    #Correlation 0.48
    x0 = np.array([ 0.37817931107096325,0.06336142355783655, 14.383871238838053, 2.556550207721383,  0.16580520802117782, 0.618701719199388, -0.2963403055660106])
  
    
    #7 duration
    x0 = np.array([0.30455489807344427,0.05127662652286009,13.042147691874233, 2.410330025995942,  0.17969556109980153, 0.7772418449286228, -0.2370347198715283])
    #Covariance+more precision
    #x0 = np.array([0.27066406,  0.05116504 , 6.19384766,  2.32397461 , 0.15463867 , 0.58085937, -0.23092773])
    #x0 = np.array([ 0.5090240676904383, 0.04375413615891409, 6.494294838498856, 2.3952100775031933,  0.2001491772844866,  0.5613156431585399, -0.1954645239544436])
    #x0 = np.array([0.39531250000000007,0.0465625,9.140625,1.94296875, 0.32484375,0.8484375000000001, -0.2221875])
    #Server Individual
    #x0 = np.array([0.19361328125,0.11587890625, 9.3548828125,0.8384765624999999,0.385546875,0.8681640625])
    #x0 = np.array([0.041523437499999996,0.1895703125, 8.841796875,0.683203125,0.43359375, 0.984765625])
    #x0 = np.array([0.17558593749999998,0.1242578125,10 (8914*11.04796+5562*7.708792)/(8914+5562)
   
    #Variance only
   
    #Name and location of files
    if system() == 'Windows':   
        path='D:/blasutto/store_model'
    else:
        path='D:/blasutto/store_model'
    
    out, mdl, agents, res = mdl_resid(x0,return_format=['distance','models','agents','scaled residuals'],
                                      #load_from=['mdl_save_bil_le.pkl'],#'mdl_save_uni.pkl'],
                                      solve_transition=False,                                    
                                      #save_to=['mdl_save_bil_le.pkl'],#'mdl_save_uni.pkl'],
                                      store_path=path,
                                       verbose=True,calibration_report=False,draw=graphs,graphs=graphs,
                                      welf=False) #Switch to true for decomposition of welfare analysis
                         
    print('Done. Residual in point x0 is {}'.format(out))
     
    #assert False
    
    #Indexes for the graphs
    if graphs:
        ai=0
        zfi=0
        zmi=2
        psii=0
        ti=15
        thi=9
        dd=0
        edu=['e','e']
         
        #Actual Graphs
        mdl[0].graph(ai,zfi,zmi,psii,ti,thi,dd,edu)
        #get_ipython().magic('reset -f')
        #If you plan to use graphs only once, deselect below to save space on disk
        #os.remove('name_model.pkl')
     
     
  
    
    
        
