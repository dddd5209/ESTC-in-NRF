# ESTC-in-NRF
codes for supporting "Extensive Spatio-Temporal Chaos in Non-reciprocal Flocking"

## Overview

This code is a CUDA implementation of a two-species Vicsek-type particle system with non-reciprocal alignment in a two-dimensional periodic domain. Each particle moves at constant speed, while its heading angle is updated by alignment interactions with neighbors inside the interaction range `R = 1` and by Gaussian angular noise. The implemented microscopic dynamics follows the microscopic model described in the attached paper, including species-wise normalization of the alignment terms and chirality measured from the deterministic angular velocity.

The main executable source file is `NRVM_chi.cu`. The remaining source files are separated into data structures, CUDA kernels, and simulator class implementation.

## File Structure

`NRVM_structure.cu` defines the `Vec2` structure, basic vector operations, and helper functions for periodic boundary handling.

`NRVM_device.cu` contains the CUDA kernels. This includes initial particle placement, random number initialization, `cos/sin` evaluation, neighbor search and alignment torque calculation, noise injection, angle and position updates, cell indexing, construction of cell head/tail arrays, and coarse-grained box observables.

`NRVM_class.cu` defines the simulator class `NRCTVM`. It handles memory allocation, initialization, particle sorting by cell index, one integration step, multi-step time evolution, global order parameter calculation, and box-level observable extraction.

`NRVM_chi.cu` contains the `main` program. It reads command-line arguments, runs the simulation, and writes the global order parameters and chirality to file. When `boxsize > 0`, it also writes coarse-grained box data.

## Physical Model

Particle `i` has position `r_i = (x_i, y_i)` and heading angle `theta_i`. The position evolves with constant speed, and the angle is updated by same-species and cross-species alignment terms plus Gaussian noise. Following the paper, the implementation normalizes the contribution from each species by the number of neighbors of that species before adding the two contributions together. The deterministic angular velocity of a particle therefore has the schematic form

```text
omega_i = (J_same / N_same) * sum_same sin(theta_j - theta_i)
        + (J_other / N_other) * sum_other sin(theta_j - theta_i)
```

where neighbors are particles within distance 1 and the boundary condition is periodic. The angular update uses an Euler-Maruyama discretization.

The global polarizations of species A and B are computed as

```text
P_A = (1 / N_A) sum_{i in A} (cos theta_i, sin theta_i)
P_B = (1 / N_B) sum_{i in B} (cos theta_i, sin theta_i)
```

The species-resolved global chirality is recorded as the species average of the deterministic angular velocity. In other words, the output values `Chi1` and `Chi2` are the instantaneous mean angular velocities without the noise term.

The initial condition sets the headings of species A and species B with a relative phase offset of `pi/2`. This is designed to probe the stability of the homogeneous chiral state.

## Numerical Implementation

At each time step, the simulation proceeds as follows.

First, particles are assigned to integer grid cells. Since the interaction range is 1, each particle only needs to check its own cell and the eight neighboring cells. To do this efficiently, particles are sorted by cell index, and the start and end positions of each cell in the sorted arrays are stored in the `Head` and `Tail` arrays.

Next, for each particle, all neighbors within distance 1 are scanned, and the alignment contributions from the same species and the other species are accumulated separately. These accumulated contributions are normalized by the corresponding neighbor counts to obtain the deterministic torque used to predict the new angle.

Gaussian noise is then added, the angle is wrapped back into `(-pi, pi]`, and the particle is moved using the updated angle. Periodic boundary conditions are applied to the position afterward.

`GetOPS()` computes the global polarization vectors and species-resolved chirality. `GetBoxConf_chi()` stores, for each user-defined box of size `boxsize`, the particle count, summed velocity components, and summed chirality.

## Compilation

This code should be compiled with `nvcc`. The recommended command is

```bash
nvcc -o NRVM_chi.exe NRVM_chi.cu -lcurand_static -lculibos
```

The build therefore assumes a CUDA environment in which CURAND and the required CUDA runtime libraries are available.

## Usage

The program takes 17 command-line arguments.

```bash
./NRVM_chi.exe xsize ysize rho_A rho_B v_A v_B T_A T_B J_AA J_AB J_BA J_BB boxsize Delta_t_meas dt total_t seed
```

The arguments are interpreted as follows.

`xsize`, `ysize` are the linear dimensions of the system.

`rho_A`, `rho_B` are the number densities of the two species. The actual particle numbers are obtained as `N_A = xsize * ysize * rho_A` and `N_B = xsize * ysize * rho_B`, converted to integers in the code.

`v_A`, `v_B` are the particle speeds of the two species.

`T_A`, `T_B` are the angular noise strengths of the two species. In the code, the angular noise is Gaussian with standard deviation `sqrt(2 T dt)`.

`J_AA`, `J_AB`, `J_BA`, `J_BB` are the intra-species and inter-species alignment couplings.

`boxsize` is the coarse-graining length used for box output. 
It should divide both `xsize` and `ysize` exactly.

`Delta_t_meas` is the measurement interval.

`dt` is the integration time step.

`total_t` is the total simulation time.

`seed` is the random seed. If `0` is given, the code generates a fresh seed from `std::random_device`.

## Example Run

A typical symmetric-density setup with self-alignment in both species can be run as

```bash
./NRVM_chi.exe 256 256 3.0 3.0 5.0 5.0 0.1 0.1 1.0 0.0 0.0 1.0 16 1.0 0.01 1000.0 12345
```

To introduce non-reciprocity, choose different values for `J_AB` and `J_BA`. For example, a point close to the purely antisymmetric line can be set by choosing `J_AB = -J_BA`.

## Output Files

The global observable file is written with a name of the form

```text
NRVM_Lx{Lx}Ly{Ly}rhoA{...}rhoB{...}vA{...}vB{...}TA{...}TB{...}JAA{...}JAB{...}JBA{...}JBB{...}.dat
```

The first line stores the run parameters as a header. Each following line contains the seven values

```text
t  OP1.x  OP1.y  OP2.x  OP2.y  Chi1  Chi2
```

where `OP1` and `OP2` are the global polarization vectors of species A and B, and `Chi1`, `Chi2` are the corresponding mean deterministic angular velocities.

If `boxsize > 0`, a separate box-output file is also written.

```text
NRVM_box{boxsize}_Lx{Lx}Ly{Ly}rhoA{...}rhoB{...}vA{...}vB{...}TA{...}TB{...}JAA{...}JAB{...}JBA{...}JBB{...}.dat
```

This file is written in binary format. At each measurement time, for each box and each species, the following values are stored in order:

```text
int N_in_box
double Vx_sum
double Vy_sum
double Chi_sum
```

The box index runs with `x` varying fastest, and within each box the data are stored in the order species A followed by species B. The total number of box-species slots per measurement is therefore

```text
2 * (xsize / boxsize) * (ysize / boxsize)
```
