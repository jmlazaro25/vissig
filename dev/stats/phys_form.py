import re
import decimal
import numpy as np
from scipy import  interpolate

# <From Nikita Blinov> ##################################################
# Hadronic R ratio used to compute hadronic width of the A'
try: Rvals=np.loadtxt("r_fixed.dat")
except: Rvals=np.loadtxt("stats/r_fixed.dat")
Rvals_interp = interpolate.interp1d(Rvals[:,0],Rvals[:,1],kind='linear');

def Rfunc(s):
    if np.sqrt(s) >= 0.36:
        return Rvals_interp(np.sqrt(s))
    else:
        return 0.

def gamma_ap_to_ll(mAp,ml,eps):
    if mAp < 2.*ml:
        return 0.

    aEM=1/137.035999679

    beta=1. - (2*ml/mAp)**2

    return (1./3.)*(aEM*eps**2)*mAp*np.sqrt(beta)*(1 + (2*ml**2)/mAp**2)

# Total decay rate of Ap into electrons and muons and hadrons
# Valid for mAp > 2*me
# Width is returned in GeV
def gamma_ap_tot(mAp, epsilon):
    me = 0.51099895/1000.
    mmu = 105.6583745/1000.
    return gamma_ap_to_ll(mAp,me,epsilon) \
            + gamma_ap_to_ll(mAp,mmu,epsilon)*(1. + Rfunc(mAp**2))

# </From Nikita Blinov> ##################################################

##################################################
# Line editing
##################################################

def numsInLine(line_w_nums):

    """ Find numbers in line """

    nums =  re.findall(r' [\d,\.,e,\-,\+]+', line_w_nums) # Seems close enough

    return [ num[1:] for num in nums ]

def rescaleLine(line_w_nums, scale=decimal.Decimal('0.1'), tokens=[0]):

    """ Replace numbers at given tokens (indicies) with scaled versions """

    numbers = numsInLine(line_w_nums)
    numbers = [ numbers[i] for i in tokens ] # Get numbers at desired indicies
    scaled_line = line_w_nums

    for number in numbers:
        scaled_line = re.sub(re.sub(r'\+', r'\\+', number), # looks like - equiv not needed
                            str(decimal.Decimal(number)*scale), scaled_line,
                            count=1)

    return scaled_line

def replaceNums(line_w_nums, tokens, new_vals):

    """ Replace numbers at given tokens (indicies) with specific new_vals """

    numbers = numsInLine(line_w_nums)
    numbers = [ numbers[i] for i in tokens ]# Numbers we care about
    new_line = line_w_nums

    for number, new_val in zip(numbers,new_vals):
        new_line = re.sub(number, str(new_val), new_line, count=1)

    return new_line
