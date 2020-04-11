************************************
*Clean SIPP data before using them
***********************************

cd "D:/blasutto/Data/SIPP raw"
clear all

************************************
*2008 SIPP sample
************************************

foreach wave in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 {
clear
import delimited pu2008`wave'.csv
**Data is 2008 SIPP--download all waves

**gen uniqueid
egen str id=concat(ssuid epppnum),format(%25.0g)
destring id, generate(unique_id)
format unique_id %25.0f

sort ssuid epppnum swave srefmon
**drop non reporting months (and therefore duplicates within a wave)


keep if srefmon==1
*******************************************
*** Add in Wave Specific Marriage Flags ***
*******************************************
*get month and year
gen rhcalmn`wave'=rhcalmn
gen rhcalyr`wave'=rhcalyr
 

*cohabitation
egen cohaba=anymatch(errp),values(10)
bysort ssuid:egen cohab_`wave'=sum(cohaba)
replace cohab_`wave'=0 if errp!=1 & errp!=2


**create marriage flag
gen married_`wave'=0
di "married flag updating, `wave'"
replace married_`wave'=1 if (ems==1  | ems==2)

**spouse number
gen epnspous`wave'=epnspous

gen age`wave'=tage

**Currently married-version 2
egen mara=anymatch(errp),values(3)
bysort ssuid:egen mara_`wave'=sum(mara)
replace mara_`wave'=0 if errp!=1  & errp!=2

******keep only unique ID and marital status for waves 02-16
gen errp`wave'=errp
keep unique_id ssuid epppnum age`wave' married_`wave' errp`wave' cohab_`wave' mara_`wave' epnspous`wave' rhcalmn`wave' rhcalyr`wave'
sort unique_id



save 08_`wave'_1, replace
}






**Clean wave 1 data
clear
import delimited pu20081.csv


**gen uniqueid
egen str id=concat(ssuid epppnum),format(%25.0g)
destring id, generate(unique_id)
format unique_id %25.0f
sort ssuid epppnum swave srefmon

*get month and year
gen rhcalmn1=rhcalmn
gen rhcalyr1=rhcalyr

sort unique_id

*Keep only if 1st wave of the year(why?refer to Voena Pistaferri et al)
keep if srefmon==1

**flag for cohabitation
egen cohaba=anymatch(errp),values(10)
bysort ssuid:egen cohab_1=sum(cohaba)
replace cohab_1=0 if errp!=1 & errp!=2



**Currently Married
gen married_1=0
replace married_1=1 if (ems==1  | ems==2)

**Currently married-version 2
egen mara=anymatch(errp),values(3)
bysort ssuid:egen mara_1=sum(mara)
replace mara_1=0 if errp!=1 & errp!=2


**spouse number
gen epnspous1=epnspous
gen errp1=errp


***Keep only some people
keep if tbyear>=1945 & tbyear<1955
gen age1=tage


******merge in married flags on unique ID 

foreach wave in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 {
merge m:m ssuid epppnum using 08_`wave'_1
tab _merge
***DROP ANY ENTRIES FOR INDIVIDUALS WHO WERE NOT IN WAVE 1
drop if _merge==2


drop _merge
sort unique_id
}

*keep if cohab_2==1 | cohab_3==1 | cohab_4==1 | cohab_5==1 | cohab_6==1 | cohab_7==1 | cohab_8==1


**********now replace the married flags to reflect if the respondent was
	*ever married during the waves (ie, divorce/death/etc will not be picked up)
foreach wave in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 {
replace cohab_`wave'=1 if cohab_`wave'>1 & cohab_`wave'!=.
replace mara_`wave'=1 if mara_`wave'>1 & mara_`wave'!=.
}


**reshape and collapse
gen wave=swave
*Need: race, college, income, gender
keep tbyear rhcalyr tpearn tfipsst wpfinwgt ssuid age1-age16 epppnum unique_id-married_16 mara_1-mara_16 cohab_1-cohab_16 rhcalyr1-rhcalyr16 rhcalmn1-rhcalmn16

*Get the data in every period
foreach wave in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 {

gen date`wave'=rhcalyr`wave'+rhcalmn`wave'/12

}

gen  sy=2008
saveold "D:\blasutto\Data\SIPP raw\sipp08t.dta",replace


************************************
*2004 SIPP sample
************************************

foreach wave in 2 3 4 5 6 7 8 9 10 11 12 {
clear
import delimited pu2004`wave'.csv
**Data is 2004 SIPP--download all waves

**gen uniqueid
egen str id=concat(ssuid epppnum),format(%25.0g)
destring id, generate(unique_id)
format unique_id %25.0f

sort ssuid epppnum swave srefmon
**drop non reporting months (and therefore duplicates within a wave)


keep if srefmon==1
*******************************************
*** Add in Wave Specific Marriage Flags ***
*******************************************
*get month and year
gen rhcalmn`wave'=rhcalmn
gen rhcalyr`wave'=rhcalyr
 

*cohabitation
egen cohaba=anymatch(errp),values(10)
bysort ssuid:egen cohab_`wave'=sum(cohaba)
replace cohab_`wave'=0 if errp!=1 & errp!=2


**create marriage flag
gen married_`wave'=0
di "married flag updating, `wave'"
replace married_`wave'=1 if (ems==1  | ems==2)

**spouse number
gen epnspous`wave'=epnspous


gen age`wave'=tage

**Currently married-version 2
egen mara=anymatch(errp),values(3)
bysort ssuid:egen mara_`wave'=sum(mara)
replace mara_`wave'=0 if errp!=1  & errp!=2

******keep only unique ID and marital status for waves 02-16
gen errp`wave'=errp
keep unique_id ssuid epppnum married_`wave' errp`wave' cohab_`wave' mara_`wave' epnspous`wave' rhcalmn`wave' rhcalyr`wave' age`wave'
sort unique_id


save 04_`wave'_1, replace
}






**Clean wave 1 data
clear
import delimited pu20041.csv


**gen uniqueid
egen str id=concat(ssuid epppnum),format(%25.0g)
destring id, generate(unique_id)
format unique_id %25.0f
sort ssuid epppnum swave srefmon

*get month and year
gen rhcalmn1=rhcalmn
gen rhcalyr1=rhcalyr

sort unique_id

*Keep only if 1st wave of the year(why?refer to Voena Pistaferri et al)
keep if srefmon==1

**flag for cohabitation
egen cohaba=anymatch(errp),values(10)
bysort ssuid:egen cohab_1=sum(cohaba)
replace cohab_1=0 if errp!=1 & errp!=2



**Currently Married
gen married_1=0
replace married_1=1 if (ems==1  | ems==2)

**Currently married-version 2
egen mara=anymatch(errp),values(3)
bysort ssuid:egen mara_1=sum(mara)
replace mara_1=0 if errp!=1 & errp!=2


**spouse number
gen epnspous1=epnspous
gen errp1=errp


***Keep only some people
keep if tbyear>=1945 & tbyear<1955
gen age1=tage


******merge in married flags on unique ID 

foreach wave in 2 3 4 5 6 7 8 9 10 11 12  {
merge m:m ssuid epppnum using 04_`wave'_1
tab _merge
***DROP ANY ENTRIES FOR INDIVIDUALS WHO WERE NOT IN WAVE 1
drop if _merge==2


drop _merge
sort unique_id
}

*keep if cohab_2==1 | cohab_3==1 | cohab_4==1 | cohab_5==1 | cohab_6==1 | cohab_7==1 | cohab_8==1


**********now replace the married flags to reflect if the respondent was
	*ever married during the waves (ie, divorce/death/etc will not be picked up)
foreach wave in 1 2 3 4 5 6 7 8 9 10 11 12  {
replace cohab_`wave'=1 if cohab_`wave'>1 & cohab_`wave'!=.
replace mara_`wave'=1 if mara_`wave'>1 & mara_`wave'!=.
}


**reshape and collapse
gen wave=swave
*Need: race, college, income, gender
keep tbyear rhcalyr tpearn tfipsst wpfinwgt ssuid epppnum age1-age12 unique_id-married_12 mara_1-mara_12 cohab_1-cohab_12 rhcalyr1-rhcalyr12 rhcalmn1-rhcalmn12

*Get the data in every period
foreach wave in 1 2 3 4 5 6 7 8 9 10 11 12  {

gen date`wave'=rhcalyr`wave'+rhcalmn`wave'/12

}

gen  sy=2004
gen date13=.
gen date14=.
gen date15=.
gen date16=.


gen mara_13=.
gen mara_14=.
gen mara_15=.
gen mara_16=.


gen married_13=.
gen married_14=.
gen married_15=.
gen married_16=.


gen cohab_13=.
gen cohab_14=.
gen cohab_15=.
gen cohab_16=.


gen age13=.
gen age14=.
gen age15=.
gen age16=.


saveold "D:\blasutto\Data\SIPP raw\sipp04t.dta",replace




************************************
*2001 SIPP sample
************************************

foreach wave in 2 3 4 5 6 7 8 9 {
clear
import delimited pu2001`wave'.csv
**Data is 2001 SIPP--download all waves

**gen uniqueid
egen str id=concat(ssuid epppnum),format(%25.0g)
destring id, generate(unique_id)
format unique_id %25.0f

sort ssuid epppnum swave srefmon
**drop non reporting months (and therefore duplicates within a wave)


keep if srefmon==1
*******************************************
*** Add in Wave Specific Marriage Flags ***
*******************************************
*get month and year
gen rhcalmn`wave'=rhcalmn
gen rhcalyr`wave'=rhcalyr
 

*cohabitation
egen cohaba=anymatch(errp),values(10)
bysort ssuid:egen cohab_`wave'=sum(cohaba)
replace cohab_`wave'=0 if errp!=1 & errp!=2


**create marriage flag
gen married_`wave'=0
di "married flag updating, `wave'"
replace married_`wave'=1 if (ems==1  | ems==2)

**spouse number
gen epnspous`wave'=epnspous

gen age`wave'=tage

**Currently married-version 2
egen mara=anymatch(errp),values(3)
bysort ssuid:egen mara_`wave'=sum(mara)
replace mara_`wave'=0 if errp!=1  & errp!=2

******keep only unique ID and marital status for waves 02-16
gen errp`wave'=errp
keep unique_id ssuid epppnum married_`wave' errp`wave' cohab_`wave' mara_`wave' epnspous`wave' rhcalmn`wave' rhcalyr`wave' age`wave'
sort unique_id



save 01_`wave'_1, replace
}






**Clean wave 1 data
clear
import delimited pu20011.csv


**gen uniqueid
egen str id=concat(ssuid epppnum),format(%25.0g)
destring id, generate(unique_id)
format unique_id %25.0f
sort ssuid epppnum swave srefmon

*get month and year
gen rhcalmn1=rhcalmn
gen rhcalyr1=rhcalyr

sort unique_id

*Keep only if 1st wave of the year(why?refer to Voena Pistaferri et al)
keep if srefmon==1

**flag for cohabitation
egen cohaba=anymatch(errp),values(10)
bysort ssuid:egen cohab_1=sum(cohaba)
replace cohab_1=0 if errp!=1 & errp!=2



**Currently Married
gen married_1=0
replace married_1=1 if (ems==1  | ems==2)

**Currently married-version 2
egen mara=anymatch(errp),values(3)
bysort ssuid:egen mara_1=sum(mara)
replace mara_1=0 if errp!=1 & errp!=2


**spouse number
gen epnspous1=epnspous
gen errp1=errp


***Keep only some people
keep if tbyear>=1945 & tbyear<1955
gen age1=tage


******merge in married flags on unique ID 

foreach wave in 2 3 4 5 6 7 8 9   {
merge m:m ssuid epppnum using 01_`wave'_1
tab _merge
***DROP ANY ENTRIES FOR INDIVIDUALS WHO WERE NOT IN WAVE 1
drop if _merge==2


drop _merge
sort unique_id
}

*keep if cohab_2==1 | cohab_3==1 | cohab_4==1 | cohab_5==1 | cohab_6==1 | cohab_7==1 | cohab_8==1


**********now replace the married flags to reflect if the respondent was
	*ever married during the waves (ie, divorce/death/etc will not be picked up)
foreach wave in 1 2 3 4 5 6 7 8 9{
replace cohab_`wave'=1 if cohab_`wave'>1 & cohab_`wave'!=.
replace mara_`wave'=1 if mara_`wave'>1 & mara_`wave'!=.
}


**reshape and collapse
gen wave=swave
*Need: race, college, income, gender
keep tbyear rhcalyr tpearn tfipsst wpfinwgt ssuid epppnum age1-age9 unique_id-married_9 mara_1-mara_9 cohab_1-cohab_9 rhcalyr1-rhcalyr9 rhcalmn1-rhcalmn9

*Get the data in every period
foreach wave in 1 2 3 4 5 6 7 8 9 {

gen date`wave'=rhcalyr`wave'+rhcalmn`wave'/12

}

gen  sy=2001
gen date10=.
gen date11=.
gen date12=.
gen date13=.
gen date14=.
gen date15=.
gen date16=.

gen mara_10=.
gen mara_11=.
gen mara_12=.
gen mara_13=.
gen mara_14=.
gen mara_15=.
gen mara_16=.

gen married_10=.
gen married_11=.
gen married_12=.
gen married_13=.
gen married_14=.
gen married_15=.
gen married_16=.

gen cohab_10=.
gen cohab_11=.
gen cohab_12=.
gen cohab_13=.
gen cohab_14=.
gen cohab_15=.
gen cohab_16=.

gen age10=.
gen age11=.
gen age12=.
gen age13=.
gen age14=.
gen age15=.
gen age16=.
saveold "D:\blasutto\Data\SIPP raw\sipp01t.dta",replace


************************************
*1996 SIPP sample
************************************

foreach wave in 2 3 4 5 6 7 8 9 {
clear
cd "D:/blasutto/Data/SIPP raw"
import delimited pu1996`wave'.csv
**Data is 1996 SIPP--download all waves

**gen uniqueid
egen str id=concat(ssuid epppnum),format(%25.0g)
destring id, generate(unique_id)
format unique_id %25.0f

sort ssuid epppnum swave srefmon
**drop non reporting months (and therefore duplicates within a wave)


keep if srefmon==1
*******************************************
*** Add in Wave Specific Marriage Flags ***
*******************************************
*get month and year
gen rhcalmn`wave'=rhcalmn
gen rhcalyr`wave'=rhcalyr
 

*cohabitation
egen cohaba=anymatch(errp),values(10)
bysort ssuid:egen cohab_`wave'=sum(cohaba)
replace cohab_`wave'=0 if errp!=1 & errp!=2


**create marriage flag
gen married_`wave'=0
replace married_`wave'=1 if (ems==1  | ems==2)

**spouse number
gen epnspous`wave'=epnspous

gen age`wave'=tage


tab ems
**Currently married-version 2
egen mara=anymatch(errp),values(3)
bysort ssuid:egen mara_`wave'=sum(mara)
replace mara_`wave'=0 if errp!=1  & errp!=2 & errp!=3

******keep only unique ID and marital status for waves 02-16
gen errp`wave'=errp
keep unique_id ssuid age`wave' epppnum married_`wave' errp`wave' cohab_`wave' mara_`wave' epnspous`wave' rhcalmn`wave' rhcalyr`wave' 
sort unique_id


saveold 96`wave'_1, replace
}






**Clean wave 1 data

clear
cd "D:/blasutto/Data/SIPP raw"
import delimited pu19961.csv


**gen uniqueid
egen str id=concat(ssuid epppnum),format(%25.0g)
destring id, generate(unique_id)
format unique_id %25.0f
sort ssuid epppnum swave srefmon

*get month and year
gen rhcalmn1=rhcalmn
gen rhcalyr1=rhcalyr

sort unique_id

*Keep only if 1st wave of the year(why?refer to Voena Pistaferri et al)
keep if srefmon==1

**flag for cohabitation
egen cohaba=anymatch(errp),values(10)
bysort ssuid:egen cohab_1=sum(cohaba)
replace cohab_1=0 if errp!=1 & errp!=2



**Currently Married
gen married_1=0
replace married_1=1 if (ems==1  | ems==2)

**Currently married-version 2
egen mara=anymatch(errp),values(3)
bysort ssuid:egen mara_1=sum(mara)
replace mara_1=0 if errp!=1 & errp!=2  & errp!=3


**spouse number
gen epnspous1=epnspous
gen errp1=errp

tab ems
***Keep only some people
keep if tbyear>=1945 & tbyear<1955
gen age1=tage


******merge in married flags on unique ID 

foreach wave in 2 3 4 5 6 7 8 9   {
merge m:m ssuid epppnum using 96`wave'_1
tab _merge
***DROP ANY ENTRIES FOR INDIVIDUALS WHO WERE NOT IN WAVE 1
drop if _merge==2
*keep if _merge==3


drop _merge
sort unique_id
}

*keep if cohab_2==1 | cohab_3==1 | cohab_4==1 | cohab_5==1 | cohab_6==1 | cohab_7==1 | cohab_8==1


**********now replace the married flags to reflect if the respondent was
	*ever married during the waves (ie, divorce/death/etc will not be picked up)
foreach wave in 1 2 3 4 5 6 7 8 9{
replace cohab_`wave'=1 if cohab_`wave'>1 & cohab_`wave'!=.
replace mara_`wave'=1 if mara_`wave'>1 & mara_`wave'!=.
}


**reshape and collapse
gen wave=swave
*Need: race, college, income, gender
keep tbyear rhcalyr tpearn tfipsst wpfinwgt ssuid epppnum age1-age9 unique_id-married_9 mara_1-mara_9 cohab_1-cohab_9 rhcalyr1-rhcalyr9 rhcalmn1-rhcalmn9

*Get the data in every period
foreach wave in 1 2 3 4 5 6 7 8 9 {

gen date`wave'=rhcalyr`wave'+rhcalmn`wave'/12

}

gen  sy=1996
gen date10=.
gen date11=.
gen date12=.
gen date13=.
gen date14=.
gen date15=.
gen date16=.

gen mara_10=.
gen mara_11=.
gen mara_12=.
gen mara_13=.
gen mara_14=.
gen mara_15=.
gen mara_16=.

gen married_10=.
gen married_11=.
gen married_12=.
gen married_13=.
gen married_14=.
gen married_15=.
gen married_16=.

gen cohab_10=.
gen cohab_11=.
gen cohab_12=.
gen cohab_13=.
gen cohab_14=.
gen cohab_15=.
gen cohab_16=.

gen age10=.
gen age11=.
gen age12=.
gen age13=.
gen age14=.
gen age15=.
gen age16=.

*keep if cohab_9!=.
saveold "D:\blasutto\Data\SIPP raw\sipp96t.dta",replace

