! HA 1, Problem 1
! Value and policy function

module globals

    implicit none
    real*8, parameter :: beta = 0.97d0, delta = 0.1d0,alpha = 0.3d0
    real*8, parameter :: gridstep = 0.005d0
    real*8, parameter :: amin = gridstep, amax = 10.0d0
    
    real*8 :: sigma
    
    integer, parameter :: gridlength = 2000 ! 1 + nint((amax - amin)/gridstep)
    real*8 :: agrid(gridlength)
    
    integer :: iii
    
    
    integer, parameter :: nstates = 2
    real*8 :: L(nstates)
    
    real*8 :: PP(nstates,nstates) ! transition matrix
    ! Rows are ``from'', columns are ``to'', so pi(3,1) is probability to go from 3 to 1

    
    real*8 :: r, w
    
    
    
end module


module utilityfunction

implicit none

contains

function utility(a, ap, ll)

    use globals
    real*8, intent(in) :: a, ap, ll
    real*8 :: utility
    real*8 :: c

    c = (1+r)*a + w*ll - ap !+ 1e-6  ! o/w 0+0-0 might be negative
    if (c > 0) then
    	utility = (c**(1-sigma))/(1-sigma)
    else
    	utility = -1e10
    endif
    	

end function

end module

module nextstatefunction

implicit none

contains

function nextstate(state)

    use globals
    integer :: nextstate
    integer, intent(in) :: state
	real*8 :: probs(nstates), cumprob, rnum
	integer :: u
	
	probs = PP(state,:)
	
	u = 1
	cumprob = probs(u)
	call random_number(rnum)
	do
	if (rnum <= cumprob) exit
	u = u+1
	cumprob = cumprob + probs(u)
	enddo
	nextstate = u
end function

end module
