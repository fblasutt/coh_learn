**Transition to Marriage from Cohabitation
clear all
use C:\Users\Fabio\Dropbox\coh_learn\animals.dta 
replace _end=0 if _end==1
replace endd=0
replace endd=1 if _end==2
stset duration ,   failure(endd)
gen homo=0
replace homo=1 if edu==1 & edup==1
replace homo=1 if edu==0 & edup==0
stcrreg  edu , compete(_end==0)


gen marry=0 if _end==0
replace marry=1 if _end==2

sum marry if edu==0
sum marry if edu==1
sum marry if edu==0 & edu==0
sum marry if edu==1 & edup==1

**Transition to Marriage from singleness
clear all
use C:\Users\Fabio\Dropbox\coh_learn\compete_single.dta 
replace _end=0 if _end==1


gen age2=age^2
gen age3=age^3

*Risk of cohabitation
gen endd=0
replace endd=1 if _end==3
stset duration ,   failure(endd)
stcrreg  edu , compete(_end== 2)

gen marry=0 if _end>1
replace marry=1 if _end==2

sum marry if edu==0
sum marry if edu==1

clear all
use C:\Users\Fabio\Dropbox\coh_learn\compete_single.dta 
replace _end=0 if _end==1


gen age2=age^2
gen age3=age^3

*risk of marriage
gen endd=0
replace endd=1 if _end==2
stset duration ,   failure(endd)
stcrreg  edu, compete(_end== 3)
