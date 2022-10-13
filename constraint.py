#from scipy.interpolate import InterpolatedUnivariateSpline
#import math
from scipy import integrate
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

step = 5001
mmu = 105.6583755*1e-3
delalpha = 251*1e-11
delalpha_upper_1sig = (251+1*59)*1e-11
delalpha_lower_1sig = (251-1*59)*1e-11
delalpha_upper_2sig = (251+2*59)*1e-11
delalpha_lower_2sig = (251-2*59)*1e-11

def f_int(x,y):
    return integrate.quad(lambda t: (y**2/(4*(np.pi)**2))*(t**2*(1-t))/(t**2+((1-t)/(mmu/x)**2)), 0.0, 1.0)[0]

gm2_upper = np.loadtxt('/Users/shihyentseng/local/muon_gm2/gm2_upper.txt')
gm2_lower = np.loadtxt('/Users/shihyentseng/local/muon_gm2/gm2_lower.txt')

f_gm2_upper = interp1d(gm2_upper[:,0], gm2_upper[:,1], kind='linear', axis=0, fill_value="extrapolate")
f_gm2_lower = interp1d(gm2_lower[:,0], gm2_lower[:,1], kind='linear', axis=0, fill_value="extrapolate")

gm2_upper_new = np.empty([step,2])
gm2_lower_new = np.empty([step,2])

for i in range(step):
    gm2_upper_new[i,0] = i*(50.0-0.003)/(step-1)
    gm2_upper_new[i,1] = f_gm2_upper(i*(50.0-0.003)/(step-1))
    gm2_lower_new[i,0] = i*(50.0-0.003)/(step-1)
    gm2_lower_new[i,1] = f_gm2_lower(i*(50.0-0.003)/(step-1))

BaBar = np.loadtxt('/Users/shihyentseng/local/muon_gm2/BaBar.txt')
BaBar4mu = np.loadtxt('/Users/shihyentseng/local/muon_gm2/BaBar4mu.txt')
BaBar_inv = np.loadtxt('/Users/shihyentseng/local/muon_gm2/BaBar_inv.txt')
CMS4mu = np.loadtxt('/Users/shihyentseng/local/muon_gm2/CMS4mu.txt')
elegm2 = np.loadtxt('/Users/shihyentseng/local/muon_gm2/elegm2.txt')
NA64 = np.loadtxt('/Users/shihyentseng/local/muon_gm2/NA64.txt')
Charm2 = np.loadtxt('/Users/shihyentseng/local/muon_gm2/Charm2.txt')
COHERENT= np.loadtxt('/Users/shihyentseng/local/muon_gm2/COHERENT.txt')
Borexino = np.loadtxt('/Users/shihyentseng/local/muon_gm2/Borexino.txt')
CCFR = np.loadtxt('/Users/shihyentseng/local/muon_gm2/CCFR.txt')
KLOE = np.loadtxt('/Users/shihyentseng/local/muon_gm2/KLOE.txt')
BBN = np.loadtxt('/Users/shihyentseng/local/muon_gm2/BBN.txt')
E137_upper = np.loadtxt('/Users/shihyentseng/local/muon_gm2/E137_upper.txt')
E137_lower = np.loadtxt('/Users/shihyentseng/local/muon_gm2/E137_lower.txt')
White_Dwarfs = np.loadtxt('/Users/shihyentseng/local/muon_gm2/White_Dwarfs.txt')

f_elegm2 = interp1d(elegm2[:,0], elegm2[:,1], kind='linear', axis=0, fill_value="extrapolate")
f_NA64 = interp1d(NA64[:,0], NA64[:,1], kind='linear', axis=0, fill_value="extrapolate")
f_Charm2= interp1d(Charm2[:,0], Charm2[:,1], kind='linear', axis=0, fill_value="extrapolate")
f_COHERENT = interp1d(COHERENT[:,0], COHERENT[:,1], kind='linear', axis=0, fill_value="extrapolate")
f_Borexino = interp1d(Borexino[:,0], Borexino[:,1], kind='linear', axis=0, fill_value="extrapolate")
f_BBN = interp1d(BBN[:,0], BBN[:,1], kind='linear', axis=0, fill_value="extrapolate")
f_E137_upper = interp1d(E137_upper[:,0], E137_upper[:,1], kind='linear', axis=0, fill_value="extrapolate")
f_E137_lower = interp1d(E137_lower[:,0], E137_lower[:,1], kind='linear', axis=0, fill_value="extrapolate")
f_White_Dwarfs = interp1d(White_Dwarfs[:,0], White_Dwarfs[:,1], kind='linear', axis=0, fill_value="extrapolate")

elegm2_new = np.empty([step,2])
NA64_new = np.empty([step,2])
Charm2_new = np.empty([step,2])
COHERENT_new = np.empty([step,2])
Borexino_new = np.empty([step,2])
BBN_new = np.empty([step,2])
E137_upper_new = np.empty([step,2])
E137_lower_new = np.empty([step,2])
White_Dwarfs_new = np.empty([step,2])

for i in range(step):
    elegm2_new[i,0] = i*(0.1-0.003)/(step-1)
    elegm2_new[i,1] = f_elegm2(i*(0.1-0.003)/(step-1))
    NA64_new[i,0] = i*(0.1-0.003)/(step-1)
    NA64_new[i,1] = f_NA64(i*(0.1-0.003)/(step-1))
    Charm2_new[i,0] = i*(8e1-0.0021)/(step-1)
    Charm2_new[i,1] = f_Charm2(i*(8e1-0.0021)/(step-1))
    COHERENT_new[i,0] = i*(8e1-0.0021)/(step-1)
    COHERENT_new[i,1] = f_COHERENT(i*(8e1-0.0021)/(step-1))
    Borexino_new[i,0] = i*(0.219-0.0021)/(step-1)
    Borexino_new[i,1] = f_Borexino(i*(0.219-0.0021)/(step-1))
    BBN_new[i,0] = i*(0.0075-0.0021)/(step-1)
    BBN_new[i,1] = f_BBN(i*(0.0075-0.0021)/(step-1))
    E137_upper_new[i,0] = i*(0.007-0.002)/(step-1)
    E137_upper_new[i,1] = f_E137_upper(i*(0.007-0.002)/(step-1))
    E137_lower_new[i,0] = i*(0.007-0.002)/(step-1)
    E137_lower_new[i,1] = f_E137_lower(i*(0.007-0.002)/(step-1))
    White_Dwarfs_new[i,0] = i*(2-0.002)/(step-1)
    White_Dwarfs_new[i,1] = f_White_Dwarfs(i*(2-0.002)/(step-1))

####################
# Plot Constraints #
####################

xmin, xmax = np.log10(2e-3), np.log10(8e1)
ymin, ymax = np.log10(1e-6), np.log10(9.5e-2)

plt.xlim(10**(xmin), 10**(xmax))
plt.ylim(10**(ymin), 10**(ymax))
plt.xscale('log')
plt.yscale('log')
plt.grid(linestyle='dotted')

f_int_vec = np.vectorize(f_int)
delta = 300
xlist = np.linspace(xmin, xmax, delta)
ylist = np.linspace(ymin, ymax, delta)
X, Y = np.meshgrid(xlist, ylist)
Z = f_int_vec(10**X, 10**Y)

#BBN
plt.fill_between(BBN_new[:,0], BBN_new[:,1], 9.5e-2, where=(BBN_new[:,1] <= 9.5e-2), color='olive', alpha=0.5, interpolate=True, label="BBN")

#CCFR
plt.fill_between(CCFR[:,0], CCFR[:,1], 9.5e-2, where=(CCFR[:,1] <= 9.5e-2), color='purple', alpha=0.5, interpolate=True, label="CCFR")

#White Dwarf
plt.fill_between(White_Dwarfs_new[:,0], White_Dwarfs_new[:,1], 9.5e-2, where=(White_Dwarfs_new[:,1] <= 9.5e-2), color='blue', alpha=0.5, interpolate=True, label="White Dwarf")
plt.plot(White_Dwarfs_new[:,0], White_Dwarfs_new[:,1], c='blue', alpha=1, linestyle='--')

#Borexino
plt.fill_between(Borexino_new[:,0], Borexino_new[:,1], 9.5e-2, where=(Borexino_new[:,1] <= 9.5e-2), color='brown', alpha=0.5, interpolate=True, label="Borexino")

#COHERENT
plt.fill_between(COHERENT_new[:,0], COHERENT_new[:,1], 9.5e-2, where=(COHERENT_new[:,1] <= 9.5e-2), color='wheat', alpha=0.5, interpolate=True, label="COHERENT")

#Charm II
plt.fill_between(Charm2_new[:,0], Charm2_new[:,1], 9.5e-2, where=(Charm2_new[:,1] <= 9.5e-2), color='deeppink', alpha=0.5, interpolate=True, label="Charm II")

#BaBar 4mu
plt.fill_between(BaBar4mu[:,0], BaBar4mu[:,1], 9.5e-2, where=(BaBar4mu[:,1] <= 9.5e-2), color='green', alpha=0.5, interpolate=True, label="BaBar 4$\mu$")

#BaBar inv
plt.fill_between(BaBar_inv[:,0], BaBar_inv[:,1], 9.5e-2, where=(BaBar_inv[:,1] <= 9.5e-2), color='gray', alpha=0.5, interpolate=True, label="BaBar inv")

#NA64
plt.fill_between(NA64_new[:,0], NA64_new[:,1], 9.5e-2, where=(NA64_new[:,1] <= 9.5e-2), color='orange', alpha=0.5, interpolate=True, label="NA64")

#electron g-2
plt.fill_between(elegm2_new[:,0], elegm2_new[:,1], 9.5e-2, where=(elegm2_new[:,1] <= 9.5e-2), color='yellow', alpha=0.5, interpolate=True, label="$(g-2)_{e}$")

#E137
plt.fill_between(E137_lower_new[:,0], E137_lower_new[:,1], E137_upper_new[:,1], where=(E137_lower_new[:,1] <= E137_upper_new[:,1]), color='coral', alpha=0.5, interpolate=True, label="E137")

#KLOE
plt.fill_between(KLOE[:,0], KLOE[:,1], 9.5e-2, where=(KLOE[:,1] <= 9.5e-2), color='lime', alpha=0.5, interpolate=True, label="KLOE")

#BaBar
plt.fill_between(BaBar[:,0], BaBar[:,1], 9.5e-2, where=(BaBar[:,1] <= 9.5e-2), color='lightskyblue', alpha=0.5, interpolate=True, label="BaBar")

#CMS 4mu
#plt.fill_between(CMS4mu[:,0], CMS4mu[:,1], 9.5e-2, where=(CMS4mu[:,1] <= 9.5e-2), color='silver', alpha=1, interpolate=True)

#muon g-2
con_cen = plt.contour(10**X, 10**Y, Z, [delalpha], colors='darkred', alpha=0.5)
con_1sig = plt.contourf(10**X, 10**Y, Z, [delalpha_lower_1sig,delalpha_upper_1sig], colors='red', alpha=0.5)
con_2sig = plt.contourf(10**X, 10**Y, Z, [delalpha_lower_2sig,delalpha_upper_2sig], colors='indianred', alpha=0.5)

plt.legend(loc='lower right')
plt.savefig('/Users/shihyentseng/local/muon_gm2/constraint.pdf')
plt.show()
