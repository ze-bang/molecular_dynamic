import h5py
import numpy as np
from opt_einsum import contract
import matplotlib.pyplot as plt
import os

z = np.array([np.array([1,1,1])/np.sqrt(3), np.array([1,-1,-1])/np.sqrt(3), np.array([-1,1,-1])/np.sqrt(3), np.array([-1,-1,1])/np.sqrt(3)])
x = np.array([[-2,1,1],[-2,-1,-1],[2,1,-1], [2,-1,1]])/np.sqrt(6)
y = np.array([[0,-1,1],[0,1,-1],[0,-1,-1], [0,1,1]])/np.sqrt(2)
localframe = np.array([x,y,z])


def Spin_global_pyrochlore(k,S,P):
    size = int(len(P)/4)
    tS = np.zeros((len(k),3), dtype=np.complex128)
    for i in range(4):
        ffact = np.exp(1j * contract('ik,jk->ij', k, P[i*size:(i+1)*size]))
        tS = tS + contract('js, ij, sp->ip', S[i*size:(i+1)*size], ffact, localframe[:,i,:])/np.sqrt(size)
    return tS

def Spin(k, S, P):
    ffact = np.exp(1j*contract('ik,jk->ij', k, P))
    N = len(S)
    return contract('js, ij->is', S, ffact)/np.sqrt(N)


def SSSF_q(k, S, P, gb=False):
    if gb:
        A = Spin_global_pyrochlore(k, S, P)
    else:
        A = Spin(k, S, P)
    return np.real(contract('ia, ib -> iab', A, np.conj(A)))

def hnhltoK(H, L, K=0):
    A = contract('ij,k->ijk',H, 2*np.array([np.pi,-np.pi,0])) \
        + contract('ij,k->ijk',L, 2*np.array([0,0,np.pi]))
    return A

def hhztoK(H, K):
    return contract('ij,k->ijk',H, 2*np.array([np.pi,0,0])) + contract('ij,k->ijk',K, 2*np.array([0,np.pi,0]))

def hnhkkn2ktoK(H, K, L=0):
    A = contract('ij,k->ijk',H, 2*np.array([np.pi,-np.pi,0])) \
        + contract('ij,k->ijk',K, 2*np.array([np.pi,np.pi,-2*np.pi]))
    return A

def SSSFGraphHnHL(A,B,d1, filename):
    plt.pcolormesh(A,B, d1)
    plt.colorbar()
    plt.ylabel(r'$(0,0,L)$')
    plt.xlabel(r'$(H,-H,0)$')
    plt.savefig(filename+".pdf")
    plt.clf()

def SSSFGraphHK0(A,B,d1, filename):
    plt.pcolormesh(A,B, d1)
    plt.colorbar()
    plt.ylabel(r'$(0,K,0)$')
    plt.xlabel(r'$(H,0,0)$')
    plt.savefig(filename+".pdf")
    plt.clf()

def SSSFGraphHH2K(A, B, d1, filename):
    plt.pcolormesh(A,B, d1)
    plt.colorbar()
    plt.ylabel(r'$(K,K,-2K)$')
    plt.xlabel(r'$(H,-H,0)$')
    plt.savefig(filename + ".pdf")
    plt.clf()

def SSSFHnHL(S, P, nK, filename, gb=False):
    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)
    K = hnhltoK(A, B).reshape((nK*nK,3))
    S = SSSF_q(K, S, P, gb)
    if gb:
        f1 = filename + "Sxx_global"
        f2 = filename + "Syy_global"
        f3 = filename + "Szz_global"
        f4 = filename + "Sxy_global"
        f5 = filename + "Sxz_global"
        f6 = filename + "Syz_global"
    else:
        f1 = filename + "Sxx_local"
        f2 = filename + "Syy_local"
        f3 = filename + "Szz_local"
        f4 = filename + "Sxy_local"
        f5 = filename + "Sxz_local"
        f6 = filename + "Syz_local"
    S = S.reshape((nK, nK, 3, 3))
    np.savetxt(f1 + '.txt', S[:,:,0,0])
    np.savetxt(f2 + '.txt', S[:,:,1,1])
    np.savetxt(f3 + '.txt', S[:,:,2,2])
    np.savetxt(f4 + '.txt', S[:,:,0,1])
    np.savetxt(f5 + '.txt', S[:,:,0,2])
    np.savetxt(f6 + '.txt', S[:,:,1,2])
    SSSFGraphHnHL(A, B, S[:,:,0,0], f1)
    SSSFGraphHnHL(A, B, S[:,:,1,1], f2)
    SSSFGraphHnHL(A, B, S[:,:,2,2], f3)
    SSSFGraphHnHL(A, B, S[:, :, 0, 1], f4)
    SSSFGraphHnHL(A, B, S[:, :, 0, 2], f5)
    SSSFGraphHnHL(A, B, S[:, :, 1, 2], f6)

def SSSFHK0(S, P, nK, filename, gb=False):
    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)
    K = hhztoK(A, B).reshape((nK*nK,3))
    S = SSSF_q(K, S, P, gb)
    if gb:
        f1 = filename + "Sxx_global"
        f2 = filename + "Syy_global"
        f3 = filename + "Szz_global"
        f4 = filename + "Sxy_global"
        f5 = filename + "Sxz_global"
        f6 = filename + "Syz_global"
    else:
        f1 = filename + "Sxx_local"
        f2 = filename + "Syy_local"
        f3 = filename + "Szz_local"
        f4 = filename + "Sxy_local"
        f5 = filename + "Sxz_local"
        f6 = filename + "Syz_local"
    S = S.reshape((nK, nK, 3, 3))
    np.savetxt(f1 + '.txt', S[:,:,0,0])
    np.savetxt(f2 + '.txt', S[:,:,1,1])
    np.savetxt(f3 + '.txt', S[:,:,2,2])
    np.savetxt(f4 + '.txt', S[:,:,0,1])
    np.savetxt(f5 + '.txt', S[:,:,0,2])
    np.savetxt(f6 + '.txt', S[:,:,1,2])
    SSSFGraphHK0(A, B, S[:,:,0,0], f1)
    SSSFGraphHK0(A, B, S[:,:,1,1], f2)
    SSSFGraphHK0(A, B, S[:,:,2,2], f3)
    SSSFGraphHK0(A, B, S[:,:,0,1], f4)
    SSSFGraphHK0(A, B, S[:,:,0,2], f5)
    SSSFGraphHK0(A, B, S[:,:,1,2], f6)


def SSSFHnHKKn2K(S, P, nK, filename, gb=False):
    H = np.linspace(-3, 3, nK)
    L = np.linspace(-3, 3, nK)
    A, B = np.meshgrid(H, L)
    K = hnhkkn2ktoK(A, B).reshape((nK*nK,3))
    S = SSSF_q(K, S, P, gb)
    if gb:
        f1 = filename + "Sxx_global"
        f2 = filename + "Syy_global"
        f3 = filename + "Szz_global"
        f4 = filename + "Sxy_global"
        f5 = filename + "Sxz_global"
        f6 = filename + "Syz_global"
    else:
        f1 = filename + "Sxx_local"
        f2 = filename + "Syy_local"
        f3 = filename + "Szz_local"
        f4 = filename + "Sxy_local"
        f5 = filename + "Sxz_local"
        f6 = filename + "Syz_local"
    S = S.reshape((nK, nK, 3, 3))
    np.savetxt(f1 + '.txt', S[:,:,0,0])
    np.savetxt(f2 + '.txt', S[:,:,1,1])
    np.savetxt(f3 + '.txt', S[:,:,2,2])
    np.savetxt(f4 + '.txt', S[:,:,0,1])
    np.savetxt(f5 + '.txt', S[:,:,0,2])
    np.savetxt(f6 + '.txt', S[:,:,1,2])
    SSSFGraphHH2K(A, B, S[:,:,0,0], f1)
    SSSFGraphHH2K(A, B, S[:,:,1,1], f2)
    SSSFGraphHH2K(A, B, S[:,:,2,2], f3)
    SSSFGraphHH2K(A, B, S[:,:,0,1], f4)
    SSSFGraphHH2K(A, B, S[:,:,0,2], f5)
    SSSFGraphHH2K(A, B, S[:,:,1,2], f6)

def SSSF_collect(S, P, nK, filename, field_dir):
    if not os.path.isdir(filename):
        os.mkdir(filename)

    if field_dir == "110":
        SSSFHnHL(S, P, nK, filename)
    elif field_dir == "001":
        SSSFHK0(S, P, nK, filename)
    else:
        SSSFHnHKKn2K(S, P, nK, filename)

    
def magnetization(S):
    A = np.zeros((4,3))
    size = int(len(S)/4)
    # print(S.shape, len(S),size)
    for i in range (4):
        A[i] = np.mean(S[i*size:(i+1)*size,:], axis=0)
    mag = contract('ax, xas->s', A, localframe)
    return mag


def plot_lattice(P, S, filename):
    ax = plt.axes(projection='3d')
    ax.set_axis_off()
    ax.scatter(P[:,0],P[:,1],P[:,2], color='w', edgecolors='b', s=60,alpha=1)
    ax.quiver(P[:,0], P[:,1], P[:,2],S[:,0], S[:,1], S[:,2], color='red', length=0.3)
    plt.savefig(filename)
    plt.show()
    plt.clf()


def fullread(Jpm_start, Jpm_end, nJpm, H_start, H_end, nH, field_dir, dir):

    JPMS = np.linspace(Jpm_start, Jpm_end, nJpm)
    HS = np.linspace(H_start, H_end, nH)
    phase_diagram = np.zeros((nJpm, nH))

    if field_dir == "110":
        n = np.array([1,1,0])/np.sqrt(2)
    elif field_dir == "111":
        n = np.array([1,1,1])/np.sqrt(3)
    else:
        n = np.array([0,0,1])

    directory = os.fsencode(dir)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".h5"):
            print(filename)
            info = filename.split("_")
            SLURM_ID = int(info[0])
            f = h5py.File(dir + filename, 'r')
            S = f['spins'][:]
            P = f['site_positions'][:]
            Jpm_index = SLURM_ID % nJpm
            H_index = SLURM_ID// nJpm
            mag = magnetization(S)
            phase_diagram[Jpm_index, H_index] = np.dot(mag,n)
            SSSF_collect(S, P, 50, dir + filename + "_SSSF/", field_dir)
            plot_lattice(P, S,  dir + filename + "_real_config.pdf")

    np.savetxt(dir+"_phase.txt", phase_diagram)
    plt.imshow(phase_diagram.T, origin="lower", aspect="auto", extent=[Jpm_start, Jpm_end, H_start, H_end])
    plt.colorbar()
    plt.savefig(dir+"_phase.pdf")



def read_MC(Jpm_start, Jpm_end, nJpm, H_start, H_end, nH, field_dir, dir, filename):

    JPMS = np.linspace(Jpm_start, Jpm_end, nJpm)
    HS = np.linspace(H_start, H_end, nH)
    phase_diagram = np.zeros((nJpm, nH))

    if field_dir == "110":
        n = np.array([1,1,0])/np.sqrt(2)
    elif field_dir == "111":
        n = np.array([1,1,1])/np.sqrt(3)
    else:
        n = np.array([0,0,1])


    info = filename.split("_")
    SLURM_ID = int(info[0])
    f = h5py.File(dir + filename, 'r')
    S = f['spins'][:]
    P = f['site_positions'][:]
    S_global = np.zeros(S.shape)
    size = int(len(P)/4)
    for i in range(4):
        S_global[i*size:(i+1)*size] = contract('js, sp->jp', S[i*size:(i+1)*size], localframe[:,i,:])

    Jpm_index = SLURM_ID % nJpm
    H_index = SLURM_ID// nJpm
    mag = magnetization(S)
    phase_diagram[Jpm_index, H_index] = np.dot(mag,n)
    plot_lattice(P, S_global,  dir + filename + "_real_config.pdf")
    SSSF_collect(S, P, 50, dir + filename + "_SSSF/", field_dir)


read_MC(-0.3, 0.1, 30, 0, 2.0, 20, "111", "/scratch/zhouzb79/Files/MC_XYZ_111/", "45_0.h5")
# fullread(-0.3, 0.1, 30, 0, 2.0, 20, "110", "/scratch/zhouzb79/Files/MC_XYZ_110/")
# fullread(-0.3, 0.1, 30, 0, 2.0, 20, "001", "/scratch/zhouzb79/Files/MC_XYZ_001/")
