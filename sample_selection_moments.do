*Sample selection for moments creation
clear all

use "C:\Users\Fabio\Dropbox\JMP\empirical analysis\NSFH\Wave 3\ICPSR_00171\DS0001\00171-0001-Data.dta"
drop CASE
gen CASE=CASENUM
keep if TYPE=="R"
merge 1:1  CASE using "C:\Users\Fabio\Dropbox\JMP\empirical analysis\NSFH\Wave 3\ICPSR_00171\DS0005\00171-0005-Data.dta",force

keep if _merge==3

keep CASE-DUPFLG DOBM DOBY IDATMM IDATYY CASEID 

label drop DOBM
label drop DOBY

*Now take result in wave one
gen MCASEID=CASE
merge 1:1  MCASEID using "C:\Users\Fabio\Dropbox\JMP\empirical analysis\NSFH\Wave 1\ICPSR_06041\DS0001\06041-0001-Data.dta" ,force

keep if _merge==3

gen coh_t=0
replace coh_t=1 if (NUMCOHAB>=1 & NUMCOHAB<99)

gen mar_t=0
replace mar_t=1 if (M95>=1 & M95<99)

gen coll=0
replace coll=1 if EDUCAT>=16
drop if EDUCAT==99

keep CASE-DUPFLG coll coh_t mar_t M485M  M499A DOBM DOBY IDATMM IDATYY CASEID M583A M583B M583C SAMWT  M2DP01 M528 M584 M535 M536 M537 M538 E1341 E904 E1006 CURMARCO CHKPTM BKMK2 M2CP01 MARCOHAB

gen weeks=.
replace weeks=M583A/52 if M583A<53 & M583B>53
replace weeks=(M583A+M583B)/52 if M583B<53 & M583A<53
replace weeks=1-M583C/52 if M583B<53 & M583A>53 
*replace weeks=99 if weeks==.

gen hours=.
replace hours=M538 if M538<=93
replace hours=M535 if M538>93
replace hours=0 if M528==2
replace hours=M535 if M528==1
replace hours=M538 if M528==1 & M536==1
replace hours=M538 if M528==1 & M537==2

gen work=hours*week
replace work=99 if weeks==.

gen married=0
replace married=1 if MARCOHAB<=2

gen cohab=0
replace cohab=1 if MARCOHAB>2 & MARCOHAB<=6



******************************************************
*Create Date at introduction of unilateral divorce
*****************************************************
decode M499A,gen(state)

*For policy variables

gen unil=0

replace unil=1971 if state=="Alabama" 
replace unil=1927 if state=="Alaska" 
replace unil=1973 if state=="Arizona" 
replace unil=0 if state=="Arkansas" 
replace unil=1970 if state=="California" 
replace unil=1972 if state=="Colorado" 
replace unil=1973 if state=="Connecticut" 
replace unil=1968 if state=="Delaware" 
replace unil=0 if state=="District of Columbia" 
replace unil=1971 if state=="Florida" 
replace unil=1973 if state=="Georgia" 
replace unil=1972 if state=="Hawaii" 
replace unil=1961 if state=="Idaho" 
replace unil=0 if state=="Illinois" 
replace unil=1973 if state=="Indiana" 
replace unil=1970 if state=="Iowa" 
replace unil=1969 if state=="Kansas" 
replace unil=1972 if state=="Kentucky" 
replace unil=0 if state=="Louisiana" 
replace unil=1973 if state=="Maine" 
replace unil=0 if state=="Maryland" 
replace unil=1975 if state=="Massachusetts" 
replace unil=1972 if state=="Michigan" 
replace unil=1974 if state=="Minnesota" 
replace unil=0 if state=="Mississippi" 
replace unil=0 if state=="Missouri" 
replace unil=1973 if state=="Montana" 
replace unil=1972 if state=="Nebraska" 
replace unil=1967 if state=="Nevada" 
replace unil=1971 if state=="New Hampshire" 
replace unil=0 if state=="New Jersey" 
replace unil=1927 if state=="New Mexico" 
replace unil=0 if state=="New York" 
replace unil=0 if state=="North Carolina" 
replace unil=1971 if state=="North Dakota" 
replace unil=1992 if state=="Ohio" 
replace unil=1927 if state=="Oklahoma" 
replace unil=1971 if state=="Oregon" 
replace unil=0 if state=="Pennsylvania" 
replace unil=1975 if state=="Rhode Island" 
replace unil=0 if state=="South Carolina" 
replace unil=1985 if state=="South Dakota" 
replace unil=0 if state=="Tennessee" 
replace unil=1970 if state=="Texas" 
replace unil=1987 if state=="Utah" 
replace unil=0 if state=="Vermont" 
replace unil=0 if state=="Virginia" 
replace state="Washington state" if state=="Washington"
replace unil=1973 if state=="Washington state" 
replace unil=1984 if state=="West Virginia" 
replace unil=1978 if state=="Wisconsin" 
replace unil=1977 if state=="Wyoming" 


gen eq=0

replace eq=1984 if state=="Alabama" 
replace eq=1927 if state=="Alaska" 
replace eq=0 if state=="Arizona" 
replace eq=1977 if state=="Arkansas" 
replace eq=0 if state=="California" 
replace eq=1972 if state=="Colorado" 
replace eq=1973 if state=="Connecticut" 
replace eq=1927 if state=="Delaware" 
replace eq=1977 if state=="District of Columbia" 
replace eq=1980 if state=="Florida" 
replace eq=1984 if state=="Georgia" 
replace eq=1927 if state=="Hawaii" 
replace eq=0 if state=="Idaho" 
replace eq=1977 if state=="Illinois" 
replace eq=1927 if state=="Indiana" 
replace eq=1927 if state=="Iowa" 
replace eq=1927 if state=="Kansas" 
replace eq=1976 if state=="Kentucky" 
replace eq=0 if state=="Louisiana" 
replace eq=1972 if state=="Maine" 
replace eq=1978 if state=="Maryland" 
replace eq=1974 if state=="Massachusetts" 
replace eq=1927 if state=="Michigan" 
replace eq=1927 if state=="Minnesota" 
replace eq=1989 if state=="Mississippi" 
replace eq=1977 if state=="Missouri" 
replace eq=1976 if state=="Montana" 
replace eq=1972 if state=="Nebraska" 
replace eq=0 if state=="Nevada" 
replace eq=1977 if state=="New Hampshire" 
replace eq=1974 if state=="New Jersey" 
replace eq=1967 if state=="New Mexico" 
replace eq=1980 if state=="New York" 
replace eq=1981 if state=="North Carolina" 
replace eq=1927 if state=="North Dakota" 
replace eq=1981 if state=="Ohio" 
replace eq=1975 if state=="Oklahoma" 
replace eq=1971 if state=="Oregon" 
replace eq=1980 if state=="Pennsylvania" 
replace eq=1981 if state=="Rhode Island" 
replace eq=1985 if state=="South Carolina" 
replace eq=1927 if state=="South Dakota" 
replace eq=1927 if state=="Tennessee" 
replace eq=0 if state=="Texas" 
replace eq=1927 if state=="Utah" 
replace eq=1927 if state=="Vermont" 
replace eq=1982 if state=="Virginia" 
replace eq=0 if state=="Washington state" 
replace eq=1985 if state=="West Virginia" 
replace eq=0 if state=="Wisconsin" 
replace eq=1967 if state=="Wyoming" 



*****************************************************

gen birth = DOBY+1900
*round(M485M/12+1900)
rename M485M birth_month

gen age_eq=.
replace age_eq=eq-birth if eq!=0

gen age_uni=.
replace age_uni=unil-birth if unil!=0



*Save
export delimited using "C:\Users\Fabio\Dropbox\coh_learn\histo_nsfh.csv",replace
