#############################
#FILE FOR GETTING SIPP DATA
##############################


library(readr)
library(SAScii)
library(data.table)



########################################
#DOWNLOAD DATA
########################################


##2008##########
## pull, unzip and save the core files
save<-function(i){
  url<-paste("https://thedataweb.rm.census.gov/pub/sipp/2008/l08puw",i,".zip",sep = '')
  path<-tempfile(fileext = '.zip')
  download.file(url,path,mode = 'wb')
  unzip(zipfile = path,exdir='D:/blasutto/Data')
}

for(i in 1:16) save(i)


##2004##########
save<-function(i){
  url<-paste("https://thedataweb.rm.census.gov/pub/sipp/2004/l04puw",i,".zip",sep = '')
  path<-tempfile(fileext = '.zip')
  download.file(url,path,mode = 'wb')
  unzip(zipfile = path,exdir='D:/blasutto/Data')
}
for(i in 1:12) save(i)

##2001##########
save<-function(i){
  url<-paste("https://thedataweb.rm.census.gov/pub/sipp/2001/l01puw",i,".zip",sep = '')
  path<-tempfile(fileext = '.zip')
  download.file(url,path,mode = 'wb')
  unzip(zipfile = path,exdir='D:/blasutto/Data')
}
for(i in 1:9) save(i)

##1996##########
save<-function(i){
  url<-paste("https://thedataweb.rm.census.gov/pub/sipp/1996/l96puw",i,".zip",sep = '')
  path<-tempfile(fileext = '.zip')
  download.file(url,path,mode = 'wb')
  unzip(zipfile = path,exdir='D:/blasutto/Data')
}
for(i in 1:12) save(i)


##################################
#Define the Variables to extract
################################

## select the variable you need in your research
## my current research is about education attainment, focus on high quality certificate.
## here, I selected "age, gender, education attainment, income" and obviously the "weight, panel, wave, month".
#Assets
#					Own		             Sole
#Stocks				east3b	       Assumed
#Checking account	east2a	   eckoast
#Gov't security		east3d	   egvoast
#CD					east2d	         ecdoast
#Money market		east2c	     emdoast
#Mortgage			east3e	       emrtown
#Municip/corp bonds	east3c	 ebdoast
#Rental property	east4a	   eownrnt
#IRA				east1b	         Assumed
#Business			eincpb1	       Assumed
#Mutual funds		east3a	     Assumed
#Savings			east2b	       esvoast
#Government bonds	east1a	   Assumed


var<-c('SPANEL',
       'SWAVE',
       'SREFMON',
       'MONTHCODE',
       'WPFINWGT',
       'TAGE','ESEX',
       'TBYEAR', #year of birth
       'TFIPSST', #state of residence
       'EEDUCATE','TPTOTINC',
       'ERRP','EMS',
       'EMS', #marital status
       'SSUID','EPPPNUM',
       'EAST3B','EAST2A','EAST3D','EAST2C','EAST3E','EAST3C',
       'EAST4A','EAST1B','EINCPB1','EAST3A','EAST2B','EAST1A',
       'ECKOAST','EGVOAST','ECDOAST','EMDOAST','EMRTOWN','EBDOAST','EOWNRNT','ESVOAST',
       'ERACE',
       'TPEARN', #earned month income
       'INCP',
       'EORIGIN',
       'EBORNUS',
       'EPNSPOUS','EFSPOUSE',#person number of spouse
       'EHREFPER',#number of reference person
       'TJBOCC1','EOCCTIM1', # occupation
       'RHCALYR','RHCALMN'
)





## process the SAS input statement from Census, thanks the SAScii package from Anthony Damico
sasinput2008.url<-'https://thedataweb.rm.census.gov/pub/sipp/2008/l08puw1.sas'
sasinput2008<-parse.SAScii(sasinput2008.url , beginline = 5 )
sasinput2008$end<-cumsum(sasinput2008$width)
sasinput2008$start<-sasinput2008$end-sasinput2008$width+1
sasinput2008<-sasinput2008[sasinput2008$varname %in% var,]

sasinput2004.url<-'https://thedataweb.rm.census.gov/pub/sipp/2004/l04puw1.sas'
sasinput2004<-parse.SAScii(sasinput2004.url , beginline = 5 )
sasinput2004$end<-cumsum(sasinput2004$width)
sasinput2004$start<-sasinput2004$end-sasinput2004$width+1
sasinput2004<-sasinput2004[sasinput2004$varname %in% var,]

sasinput2001.url<-'https://thedataweb.rm.census.gov/pub/sipp/2001/p01puw1.sas'
sasinput2001<-parse.SAScii(sasinput2001.url , beginline = 5 )
sasinput2001$end<-cumsum(sasinput2001$width)
sasinput2001$start<-sasinput2001$end-sasinput2001$width+1
sasinput2001<-sasinput2001[sasinput2001$varname %in% var,]

sasinput1996.url<-'https://thedataweb.rm.census.gov/pub/sipp/1996/sip96lgt.sas'
sasinput1996<-parse.SAScii(sasinput1996.url , beginline = 5 )
sasinput1996$end<-cumsum(sasinput1996$width)
sasinput1996$start<-sasinput1996$end-sasinput1996$width+1
sasinput1996<-sasinput1996[sasinput1996$varname %in% var,]

#######EXTRACT THE DATA##################

## 2008
i=1
df=data.frame()
while(i<=16){
  location<-paste0('D:/blasutto/Data/l08puw',i,'.dat')
  wave<-read_fwf(location,
                 fwf_positions(c(sasinput2008$start),c(sasinput2008$end),c(sasinput2008$varname)))
  #wave<-wave[wave$SREFMON==4,]
  df=rbind(df,wave)
  write.csv(wave,file = paste('D:/blasutto/Data/pu2008',i,'.csv',sep = ''),row.names = FALSE)
  i=i+1
}

## 2004
i=1
df=data.frame()
while(i<=12){
  location<-paste0('D:/blasutto/Data/l04puw',i,'.dat')
  wave<-read_fwf(location,
                 fwf_positions(c(sasinput2004$start),c(sasinput2004$end),c(sasinput2004$varname)))
  #wave<-wave[wave$SREFMON==4,]
  df=rbind(df,wave)
  write.csv(wave,file = paste('D:/blasutto/Data/pu2004',i,'.csv',sep = ''),row.names = FALSE)
  i=i+1
}

## 2001
i=1
df=data.frame()
while(i<=19){
  location<-paste0('D:/blasutto/Data/l01puw',i,'.dat')
  wave<-read_fwf(location,
                 fwf_positions(c(sasinput2001$start),c(sasinput2001$end),c(sasinput2001$varname)))
  #wave<-wave[wave$SREFMON==4,]
  df=rbind(df,wave)
  write.csv(wave,file = paste('D:/blasutto/Data/pu2001',i,'.csv',sep = ''),row.names = FALSE)
  i=i+1
}

## 1996
i=1
df=data.frame()
while(i<=12){
  location<-paste0('D:/blasutto/Data/l96puw',i,'.dat')
  wave<-read_fwf(location,
                 fwf_positions(c(sasinput1996$start),c(sasinput1996$end),c(sasinput1996$varname)))
  #wave<-wave[wave$SREFMON==4,]
  df=rbind(df,wave)
  write.csv(wave,file = paste('D:/blasutto/Data/pu1996',i,'.csv',sep = ''),row.names = FALSE)
  i=i+1
}

#df<-apply(df,2,as.numeric)
#write.csv(df,file = 'D:/blasutto/Data/pu2008.csv',row.names = FALSE)