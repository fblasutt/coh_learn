**Transition to Marriage from Cohabitation
clear all
use C:\Users\Fabio\Dropbox\coh_learn\animals.dta 

replace endd=0
replace endd=1 if _end==2
stset duration ,   failure(endd)
stcrreg  edu, compete(_end== 1)


**Transition to Marriage from singleness
clear all
use C:\Users\Fabio\Dropbox\coh_learn\compete_single.dta 

gen age2=age^2
gen age3=age^3

*Risk of cohabitation
gen endd=0
replace endd=1 if _end==1
stset duration ,   failure(endd)
stcrreg  edu, compete(_end== 2)


*risk of marriage
replace endd=0
replace endd=1 if _end==2
stset duration ,   failure(endd)
stcrreg  edu, compete(_end== 1)
