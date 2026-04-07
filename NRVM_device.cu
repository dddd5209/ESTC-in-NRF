
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_malloc.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <cuda_runtime.h>
#include "NRVM_structure.cu"

#define radius2 (1.0)   // (interaction range)^2
#define my_min(A, B) ((A)>(B) ? (B):(A))
#define nThreadMax 128
#define two_ppi (6.28318530717958648)
#define ppi     (3.14159265358979324)

void error_output(const char *desc)
{
    printf("%s\n", desc) ; exit(-1) ;
}

// initializing RNG for all threads
__global__ 
void initialize_prng(int n_total, unsigned int seed, curandState *state)
{
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if (tid>=n_total) return;

    curand_init(seed, tid, 0, &state[tid]) ;
}

// initial configuration
__global__ 
void init_config( const int xsize, const int ysize,  const int n_A, const int n_B, 
                Vec2 *positions, double *angles, char *species, curandState *state) 
{
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x ;
    if (tid>=n_A+n_B) return;

    positions[tid].x = (1.-curand_uniform(&state[tid]))*xsize ;
    positions[tid].y = (1.-curand_uniform(&state[tid]))*ysize ;

    if (tid < n_A)  species[tid]= 0;
    else  species[tid]= 1;
    
    angles[tid] =// two_ppi*(0.5-curand_uniform(&state[tid])) ;
                 two_ppi*species[tid]/4 ;

}


__global__ void cal_cossin(int n_total, double *angles, Vec2 *CosSin)
{
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x ;
    if (tid>=n_total) return;

    CosSin[tid]= {cos(angles[tid]),sin(angles[tid])};

}

// spin flip using the Euler algorithm
__global__ void angle_interactions_norm_pre(int n_total, const int xsize, const int ysize, const double J_AA, const double J_AB, const double J_BA, const double J_BB, const double dt,
        Vec2 *positions, double *angles, char* species, int *cellHead, int *cellTail, double *angleTemp, double *torque, Vec2 *CosSin)
{
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x ;
    if (tid>=n_total) return;

    double x = positions[tid].x ;
    double y = positions[tid].y ;
    double dx, dy;
    double del_theta0=0.0 ;
    double del_theta1=0.0 ;
    int k, zz ;
    int nofn0=0;
    int nofn1=0;
    char my_species=species[tid];

    for(int a=-1; a<=1; a++) {
        for(int b=-1; b<=1; b++) {
            // zz : index for neighboring cells
            zz = ((int)x+a+xsize)%xsize + (((int)y+b+ysize)%ysize)*xsize ;
            // loop over all the other particles in the cell zz
            for(k=cellHead[zz]; k<=cellTail[zz]; k++) {
                dx = fabs(x-positions[k].x) ;
                if(dx>xsize/2.) dx = xsize-dx ;
                dy = fabs(y-positions[k].y) ;
                if(dy>ysize/2.) dy = ysize-dy ;
                if(dx*dx+dy*dy < radius2) {
                    if (my_species == 0 && species[k] == 0)      {del_theta0 += J_AA*(CosSin[k].y*CosSin[tid].x-CosSin[k].x*CosSin[tid].y); nofn0++;}
                    else if (my_species == 0 && species[k] == 1) {del_theta1 += J_AB*(CosSin[k].y*CosSin[tid].x-CosSin[k].x*CosSin[tid].y); nofn1++;}
                    else if (my_species == 1 && species[k] == 0) {del_theta0 += J_BA*(CosSin[k].y*CosSin[tid].x-CosSin[k].x*CosSin[tid].y); nofn0++;}
                    else if (my_species == 1 && species[k] == 1) {del_theta1 += J_BB*(CosSin[k].y*CosSin[tid].x-CosSin[k].x*CosSin[tid].y); nofn1++;}
                }
            }
        }
    }
    if (nofn0==0) nofn0=1;
    if (nofn1==0) nofn1=1;
    // spin flip 
    torque[tid] = del_theta0/(double)nofn0+del_theta1/(double)nofn1;
    angleTemp[tid] = angles[tid]+torque[tid]*dt;
}

__global__ void angle_noise(int n_total, const int xsize, const int ysize, const double noise_amplitude_A, const double noise_amplitude_B,
                            char* species, curandState *state, double *angleTemp)
{
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x ;
    if (tid>=n_total) return;

    double rand=curand_normal_double(&state[tid]);

    if (species[tid] == 0)  angleTemp[tid] += noise_amplitude_A*rand;
    else angleTemp[tid] += noise_amplitude_B*rand;

    angleTemp[tid] = angleTemp[tid] - two_ppi * floor((angleTemp[tid] + ppi) / two_ppi);

}

__global__ 
void position_update(int n_total, const int xsize, const int ysize, 
                        Vec2 *positions, double *angles, char *species, const double vA, const double vB, const double dt)
{
    // particle index
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x ;
    if (tid>=n_total) return;

    // update
    if (species[tid] == 0)  positions[tid] += vA*AngleToVec2{}(angles[tid])*dt;
    else positions[tid] += vB*AngleToVec2{}(angles[tid])*dt;

    positions[tid] = apply_periodic_boundary(positions[tid], xsize, ysize);

}

__global__ void angle_update(int n_total, double *anglesTemp, double *angles)
{
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x ;
    if (tid>=n_total) return;

    angles[tid] = anglesTemp[tid];

}

// make a table "cell[i]" for the cell index for a particle i
__global__ 
void find_address(int n_total,const int xsize, const int ysize, int *cell, Vec2 *positions)
{
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x ;
    if (tid>=n_total) return;

    cell[tid] = (int)positions[tid].x%xsize + xsize*((int)positions[tid].y%ysize);
}

// make tables "cellHead[c]" and "cellTail[c]" for the index 
// of the first and the last praticle in a cell c
// empty cells are not updated
__global__ 
void cell_head_tail(int n_total, int *cell, int *cellHead, int *cellTail)
{
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x ;
    if (tid>=n_total) return;

    if(tid==0) cellHead[cell[tid]] = tid ;
    else {
        if(cell[tid]!=cell[tid-1]) cellHead[cell[tid]] = tid ;
    }
    if(tid==n_total-1) cellTail[cell[tid]] = tid ;
    else {
        if(cell[tid]!=cell[tid+1]) cellTail[cell[tid]] = tid ;
    }
}

void __global__ cell_m_chi(const int xsize,const  int ysize, const char q, const Vec2 *positions, const double *angles, const double *torque, 
                        const char* species, const  int *cellHead, const  int *cellTail,const  int box_len, int* NinBox, Vec2 *VinBox, double *CinBox, double dt)
{    
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x ;
    if (tid>=xsize*ysize/box_len/box_len) return;
    
    for (int q_sub=0;q_sub<q;q_sub++){
        NinBox[tid*q+q_sub]=0;
        VinBox[tid*q+q_sub]=Vec2({0.,0.});
        CinBox[tid*q+q_sub]=0.;
    }

    int Nb1=xsize/box_len;

    int index1=(tid/Nb1)*box_len*xsize+(tid%Nb1)*box_len;

    for (int i=0;i<box_len*box_len;i++){
        int index2=(i/box_len)*xsize+i%box_len;

        for(int k=cellHead[index1+index2]; k<=cellTail[index1+index2]; k++) 
        {
            NinBox[tid*q+species[k]] ++;
            VinBox[tid*q+species[k]] += AngleToVec2{}(angles[k]);
            CinBox[tid*q+species[k]] += torque[k];
        }
    }

}
