#include "NRVM_device.cu" 

class NRCTVM{
public:
    NRCTVM(int xsize, int ysize, int nA, int nB, double vA, double vB, double TA, double TB, 
            double JAA, double JAB, double JBA, double JBB, int config_box, double dt, unsigned int seed);

    ~NRCTVM();
    void TimeStep();
    void TimeEvolution(double t);
    void GetConf(Vec2* host_r, Vec2* host_v, char* host_species);
    Vec2 GetOP();
    void GetBoxConf(int* ninbox,Vec2* vinbox, int box_len);
    void GetBoxConf_chi(int* ninbox, Vec2* vinbox, double* cinbox, int box_len);
    void GetOPS(Vec2& OP1, Vec2& OP2, double& Chi1, double& Chi2);

protected:
    // System parameters
    int XSize, YSize, NofGrid;
    int N_A, N_B, NofParticles;
    double V_A, V_B;
    double T_A, T_B;
    double NoiseAmplitude_A, NoiseAmplitude_B;
    double J_AA, J_AB, J_BA, J_BB;
    int Config_Box;
    double Dt;
    unsigned int Seed;

    // CUDA execution parameters
    int NofThreads, NofBlocks, NofBlocks_c;

    // Device pointers
    int *Cell, *Head, *Tail;
    Vec2 *Positions,*CosSin;
    double *Angles, *TempAngles, *Torque;
    char *Species;
    curandState *States;
    int* NinBox;
    Vec2 *VinBox;
    double* CinBox;
    // Member functions
    void Initialization();
    void AllocateMemory();
    void FreeMemory();

    void Fill_Cell();
    void Step_Euler();
};

NRCTVM::NRCTVM(
    int xsize, int ysize, int nA, int nB, double vA, double vB, double TA, double TB, 
    double JAA, double JAB, double JBA, double JBB, int config_box, double dt, unsigned int seed)
    : XSize(xsize), YSize(ysize), N_A(nA), N_B(nB), V_A(vA), V_B(vB), T_A(TA), T_B(TB),
      J_AA(JAA), J_AB(JAB), J_BA(JBA), J_BB(JBB), Config_Box(config_box), Dt(dt), Seed(seed)
{
    NofParticles = N_A + N_B;
    NofGrid = XSize * YSize;
    NoiseAmplitude_A=pow(2.*T_A*Dt,0.5);
    NoiseAmplitude_B=pow(2.*T_B*Dt,0.5);
    
    NofThreads = my_min(nThreadMax, NofParticles);
    NofBlocks  = NofParticles / NofThreads;
    if (NofParticles % NofThreads != 0) NofBlocks++;
    NofBlocks_c = XSize * YSize / NofThreads;
    if ((XSize * YSize) % NofThreads != 0) NofBlocks_c++;

    AllocateMemory();
    Initialization();
}

NRCTVM::~NRCTVM()
{
    FreeMemory();
}

void NRCTVM::AllocateMemory() {
    // Allocate Device memory

    cudaMalloc(&Cell,     sizeof(int)*NofParticles) ;
    cudaMalloc(&Head,     sizeof(int)*NofGrid) ;
    cudaMalloc(&Tail,     sizeof(int)*NofGrid) ;

    cudaMalloc(&NinBox,     sizeof(int)*NofGrid*2/SizeofBox/SizeofBox) ;
    cudaMalloc(&VinBox,     sizeof(Vec2)*NofGrid*2/SizeofBox/SizeofBox) ;
    cudaMalloc(&CinBox,     sizeof(double)*NofGrid*2/SizeofBox/SizeofBox) ;

    cudaMalloc(&Positions,  sizeof(Vec2)*NofParticles) ;
    cudaMalloc(&Angles,      sizeof(double)*NofParticles) ;
    cudaMalloc(&Torque, sizeof(double)*NofParticles) ;
    cudaMalloc(&TempAngles,  sizeof(double)*NofParticles) ;
    cudaMalloc(&Species,    sizeof(char)*NofParticles) ;
    cudaMalloc(&CosSin,  sizeof(Vec2)*NofParticles) ;

    cudaMalloc(&States,     sizeof(curandState)*NofParticles) ;

    initialize_prng<<<NofBlocks, NofThreads>>>(NofParticles, Seed, States) ;
}

void NRCTVM::Initialization() {
    init_config<<<NofBlocks,NofThreads>>>(XSize, YSize, N_A, N_B, Positions, Angles, Species, States) ;
}

void NRCTVM::FreeMemory() {
    // Free Device memory
    cudaFree(Cell);
    cudaFree(Head);
    cudaFree(Tail);

    cudaFree(NinBox);
    cudaFree(VinBox);
    cudaFree(CinBox);

    cudaFree(Positions);
    cudaFree(Angles);
    cudaFree(Torque);
    cudaFree(TempAngles);
    cudaFree(Species);
    cudaFree(CosSin);

    cudaFree(States);
}

void NRCTVM::Fill_Cell(){
    // locate each particle in a cell
    find_address<<<NofBlocks, NofThreads>>>(NofParticles, XSize, YSize, Cell, Positions);

    // sort particles in the ascending order of the cell address
    auto zipped_begin = thrust::make_zip_iterator(
        thrust::make_tuple(
            thrust::device_ptr<Vec2>(Positions), 
            thrust::device_ptr<double>(Angles), 
            thrust::device_ptr<char>(Species),
            thrust::device_ptr<double>(Torque)
        ));    

    thrust::sort_by_key(
        thrust::device_ptr<int>(Cell),
        thrust::device_ptr<int>(Cell) + NofParticles,
        zipped_begin
    );

    // setting cell head to a null value -1
    thrust::fill(thrust::device_ptr<int>(Head),
            thrust::device_ptr<int>(Head)+NofGrid, -1);
    // setting cell tail to a null value -2 
    thrust::fill(thrust::device_ptr<int>(Tail),
            thrust::device_ptr<int>(Tail)+NofGrid, -2);
    // particle indices of the first and last particles in each cell
    cell_head_tail<<<NofBlocks, NofThreads>>>(NofParticles, Cell, Head, Tail);
}


void NRCTVM::TimeStep(){
    Fill_Cell();
    cal_cossin<<<NofBlocks, NofThreads>>>(NofParticles, Angles, CosSin);
    angle_interactions_norm_pre<<<NofBlocks, NofThreads>>>(NofParticles, XSize, YSize, J_AA, J_AB, J_BA, J_BB, Dt,
                                        Positions, Angles, Species, Head, Tail, TempAngles, Torque,CosSin);
    angle_noise<<<NofBlocks, NofThreads>>>(NofParticles, XSize, YSize, NoiseAmplitude_A, NoiseAmplitude_B, Species, States, TempAngles);

    angle_update<<<NofBlocks, NofThreads>>>(NofParticles, TempAngles, Angles);
    position_update<<<NofBlocks, NofThreads>>>(NofParticles, XSize, YSize, Positions, Angles, Species, V_A, V_B, Dt);
}

void NRCTVM::TimeEvolution(double t){
    for (double t0=Dt/2.;t0<t;t0+=Dt){
        TimeStep();
    }
}

void NRCTVM::GetOPS(Vec2& OP1, Vec2& OP2, double& Chi1, double& Chi2) {

    auto zipped_begin = thrust::make_zip_iterator(
        thrust::make_tuple(
            thrust::device_ptr<Vec2>(Positions), 
            thrust::device_ptr<double>(Angles), 
            thrust::device_ptr<double>(Torque)
        ));    

    thrust::sort_by_key(
        thrust::device_ptr<char>(Species),
        thrust::device_ptr<char>(Species) + NofParticles,
        zipped_begin
    );

    thrust::device_ptr<const double> beg(Angles);
    thrust::device_ptr<const double> end1beg2 = beg + N_A;
    thrust::device_ptr<const double> end2 = end1beg2 + N_B;

    const Vec2 zero(0.0, 0.0);

    // run entirely on the GPU
    OP1 = thrust::transform_reduce(
        thrust::device,          // or thrust::cuda::par.on(stream)
        beg, end1beg2,                // [Angles, Angles + N)
        AngleToVec2{},           // map θ -> (cosθ, sinθ)
        zero,                    // init
        Vec2Plus{}               // sum Vec2s
    );

    double invN = 1.0 / static_cast<double>(N_A);
    OP1.x *= invN;
    OP1.y *= invN;

    OP2 = thrust::transform_reduce(
        thrust::device,          // or thrust::cuda::par.on(stream)
        end1beg2, end2,                // [Angles, Angles + N)
        AngleToVec2{},           // map θ -> (cosθ, sinθ)
        zero,                    // init
        Vec2Plus{}               // sum Vec2s
    );

    invN = 1.0 / static_cast<double>(N_B);
    OP2.x *= invN;
    OP2.y *= invN;

    thrust::device_ptr<const double> begT(Torque);
    thrust::device_ptr<const double> end1beg2T = begT + N_A;
    thrust::device_ptr<const double> end2T = end1beg2T + N_B;

    const double z = 0.0;

    // run entirely on the GPU
    Chi1 = thrust::reduce(
        thrust::device,       
        begT, end1beg2T,             
        z,
        thrust::plus<double>()
    );
    
    // run entirely on the GPU
    Chi2 = thrust::reduce(
        thrust::device,       
        end1beg2T, end2T,             
        z,
        thrust::plus<double>()
    );

    invN = 1.0 / static_cast<double>(N_A);
    Chi1 *= invN;
    invN = 1.0 / static_cast<double>(N_B);
    Chi2 *= invN;

}

void NRCTVM::GetBoxConf_chi(int* ninbox, Vec2* vinbox, double* cinbox, int SizeofBox)
{
    Fill_Cell();
    cell_m_chi<<<NofBlocks_c, NofThreads>>>(XSize, YSize, 2, Positions, Angles, Torque, Species, Head, Tail, SizeofBox, NinBox, VinBox, CinBox, Dt);

    cudaMemcpy(ninbox, NinBox, sizeof(int)*(2*NofGrid/SizeofBox/SizeofBox), cudaMemcpyDeviceToHost);
    cudaMemcpy(vinbox, VinBox, sizeof(Vec2)*(2*NofGrid/SizeofBox/SizeofBox), cudaMemcpyDeviceToHost);
    cudaMemcpy(cinbox, CinBox, sizeof(double)*(2*NofGrid/SizeofBox/SizeofBox), cudaMemcpyDeviceToHost);

}
