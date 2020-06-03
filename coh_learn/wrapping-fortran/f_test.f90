! HA 1, Problem 1
! Value and policy function
include "f_modules.f90"

! This is version without Monte-Carlo
! Stationary distribution is obtained via Markov Chain iteration

subroutine VFI(sigma_in,r_result)
!program VFI
	
	use globals
	use utilityfunction
	use nextstatefunction
	
	implicit none
	

	real*8, intent(in) :: sigma_in
	real*8, intent(out) :: r_result
	
	real*8 :: UMAT(gridlength,gridlength,nstates)
	real*8 :: Vopt(gridlength,nstates),  Vold(gridlength,nstates), Vh(gridlength,nstates), Vav(gridlength)
	real*8 :: apopt(gridlength,nstates), probs(nstates)
	
	
	integer :: popt(gridlength,nstates)
	
	integer :: Howard = 1 !set to 1 to turn on Howard improvement
	
	real*8 :: dist(gridlength,nstates), distold(gridlength,nstates), distuncond(gridlength)
	real*8 :: aval(gridlength)
	real*8 :: runiform, avgutility
	
	real*8 :: knew,lnew,rnew,aa

	integer :: i, j, ii, jj, newi
	
	real*8 :: change, tol = 10.0d0**(-5.0d0)
	
	real*8 :: rmax = (1/beta) - 1
    real*8 :: rmin = -delta
	
	! Setting probabilities and states
    
    ! L is states matrix, declared in globals
    ! PP is transition probabilities, see modules
    

    sigma = sigma_in
    !sigma = 0.6d0

    agrid(:) = (/(amin + gridstep*(iii-1),iii=1,gridlength)/)

    L(1) = 0d0
    L(2) = 1d0
    
    PP(1,1) = 0.5d0
    PP(1,2) = 0.5d0
    PP(2,1) = 0.2d0
    PP(2,2) = 0.8d0
	
	write(*,*) 'Hi!'
	write(*,*) 'Gridsize today is ', gridlength
	
	write(*,*) ' '
	!call tic()
	tol = 10.0d0**(-5.0d0)
	
	
	
	r = 0.5*rmax+0.5*rmin ! initial value
    w = (1-alpha)*(alpha/(r+delta))**(alpha/(1-alpha))
    
    Vopt(:,:) = 0.0d0	
    
    distold = 0.0d0
    dist = 0.0d0
    dist(int(dble(gridlength)/2),1) = 1
    
    
    ! Here is main do
    write(*,*) 'Interest rate is:'
    do
    
    
	do ii = 1,nstates
		do i = 1,gridlength
			do j = 1,gridlength
				UMAT(i,j,ii) = utility(agrid(i),agrid(j),L(ii))
			enddo
		enddo
	enddo
	

    
  
	do
	Vold = Vopt
	do ii = 1,nstates
		Vav = 0d0
		do j = 1,nstates
			Vav = Vav + PP(ii,j)*Vold(:,j)
		enddo
		popt(:,ii) = (/(maxloc((UMAT(i,:,ii)+beta*Vav(:)),1),i=1,gridlength)/)
		Vopt(:,ii) = (/(UMAT(i,popt(i,ii),ii)+beta*Vav(popt(i,ii)),i=1,gridlength)/)
	enddo
	if (Howard == 1) then
	! Do Howard to speed up
		do
			Vh = Vopt
			do ii = 1,nstates
				Vav = 0d0
			do j = 1,nstates
				Vav = Vav + PP(ii,j)*Vh(:,j)
			enddo
			Vopt(:,ii) = (/(UMAT(i,popt(i,ii),ii)+beta*Vav(popt(i,ii)),i=1,gridlength)/)
			enddo
		change = maxval(maxval(abs(Vopt-Vh),2),1)
		!write(*,*) 'H', change
		if (change <= tol) exit
		enddo
	end if
	change = maxval(maxval(abs(Vopt-Vold),2),1)
	!write(*,*) 'A', change
	if (change <= tol) exit
    enddo
    
   ! do i = 1,gridlength
   !  write(*,*) popt(i,1), ' ', popt(i,2)
   ! enddo
    

    apopt = 0d0
    do i = 1,nstates
    	do j = 1,gridlength
    		apopt(j,i) = agrid(popt(j,i))
    	enddo
    enddo
    
    
    
    !write(*,*) 'Iterations'
    
    
    
   
    
    ! Iterate distribution
    
    
    
    do
    distold = dist
    dist = 0.0d0
    do i=1,gridlength
    do j=1,nstates
    	aa = distold(i,j) ! Probability in cell
    	newi = popt(i,j) ! New asset position
    	do jj=1,nstates ! New possible state
    		dist(newi,jj) = dist(newi,jj) + PP(j,jj)*aa
    	enddo
	enddo
	enddo
    
    change = maxval(abs(distold-dist))
    !write(*,*) change
    if (change<=tol) exit
    enddo
    
    distuncond = sum(dist,2)
    
    knew = 0
    lnew = 0
   
    
    do i = 1,gridlength
    do j = 1,nstates
    	lnew = lnew + L(j)*dist(i,j)
    	knew = knew + agrid(i)*dist(i,j)
    enddo
    enddo
    
    
    
	
	
    
    rnew = alpha*((knew/lnew)**(alpha-1))-delta
    
    change = abs(r-rnew)
    
    write(*,'(f8.6)') r
    
    if (change <= tol) exit

    r = 0.9d0*r + 0.1d0*rnew
    w = (1-alpha)*(alpha/(r+delta))**(alpha/(1-alpha))

	
    enddo
    

    
    r_result = r

    !call toc()
    
    avgutility = 0.0d0
    do i=1,gridlength
    do j=1,nstates
    	avgutility = avgutility + dist(i,j)*UMAT(i,popt(i,j),j)
    enddo
    enddo
    
	
	open(21,file='distribution.csv')
	do i = 1,gridlength
	write(21,'(f7.4,x,f7.4, x, f7.4)') agrid(i), dist(i,1), dist(i,2) 
	enddo
	close(21)
	
	open(22,file='output.txt')
	write(22,'(a, x, f7.6)') 'Interest rate', r
	write(22,'(a, x, f8.6)') 'Wage rate', w
	!write(22,'(a, x, f7.4)') 'Wage rate', w
	write(22,'(a, x, f7.4)') 'Avg utility', avgutility
	close(22)
    
    !call plot_hist(agrid,distuncond)
    !call execplot(title='Asset distribution',xlabel='Assets',ylabel='Share')
    
    !call plot(agrid,Vopt(:,1),legend='Unhealthy')
    !call plot(agrid,Vopt(:,2),legend='Healthy')
    !call execplot(title='Value functions',xlabel='Assets',ylabel='V(a)')
    !adist = (/(agrid(dist(i,1)),i=1,nagents)/)

    !do i = 1,gridlength
    !	write(*,*) dist(i,1), adist(i)
    !	!write(*,*) dist(i,1), ' ', dist(i,2), ' ', distold(i,1), ' ', distold(i,2)
    
    
    
end subroutine
!end program