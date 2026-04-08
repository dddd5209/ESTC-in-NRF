
#include "NRVM_class.cu" 
#include <fstream>

int main(int argc, char *argv[])
{
    if(argc!=18) error_output("x y rho_A rho_B v_A v_B T_A T_B J_AA J_AB J_BA J_BB boxsize Detla_t_meas dt tmax seed") ;
    const int    xsize  = atoi(argv[1]);
    const int    ysize  = atoi(argv[2]);
    const double  rho_A  = atof(argv[3]);
    const double  rho_B  = atof(argv[4]);
    const double  v_A      = atof(argv[5]);
    const double  v_B      = atof(argv[6]);
    const double  T_A      = atof(argv[7]);
    const double  T_B      = atof(argv[8]);
    const double  J_AA      = atof(argv[9]);
    const double  J_AB      = atof(argv[10]);
    const double  J_BA      = atof(argv[11]);
    const double  J_BB      = atof(argv[12]);
    const int     config_box = atoi(argv[13]);
    const double  Delta_t_meas = atof(argv[14]);
    const double  dt     = atof(argv[15]);
    unsigned int  total_t   = atoi(argv[16]);
    unsigned int  seed   = atoi(argv[17]);

    if(seed==0) {
        std::random_device rd;
        seed = rd();
    }

    // total number of particles
    const  int   n_A = (int)(xsize*ysize*(rho_A)+0.01) ;
    const  int   n_B = (int)(xsize*ysize*(rho_B)+0.01) ;

    char fname[100] ;

    std::ofstream writeFile,writeFile2;

    NRCTVM NRVM(xsize, ysize, n_A, n_B, v_A, v_B, T_A, T_B, J_AA, J_AB, J_BA, J_BB, config_box, dt, seed);

    sprintf(fname,"NRVM_Lx%dLy%drhoA%.2frhoB%.2fvA%.2fvB%.2fTA%.2fTB%.2fJAA%.2fJAB%.2fJBA%.2fJBB%.2f.dat", xsize, ysize, rho_A, rho_B, v_A, v_B, T_A, T_B, J_AA, J_AB, J_BA, J_BB);
    writeFile.open(fname, std::ios::binary);
    writeFile << "# xsize=" << xsize << " ysize=" << ysize << " rho_A=" << rho_A << " rho_B=" << rho_B 
              << " v_A=" << v_A << " v_B=" << v_B << " T_A=" << T_A << " T_B=" << T_B 
              << " J_AA=" << J_AA << " J_AB=" << J_AB << " J_BA=" << J_BA << " J_BB=" << J_BB 
              << " config_box=" << config_box << " Delta_t_meas=" << Delta_t_meas << " dt=" << dt 
              << " total_t=" << total_t << " seed=" << seed << "\n";

    Vec2 OP1,OP2;
    double Chi1{},Chi2{};
    NRVM.GetOPS(OP1,OP2,Chi1,Chi2); 

    int *NinBox = new int[2*xsize*ysize/config_box/config_box];
    Vec2 *VinBox = new Vec2[2*xsize*ysize/config_box/config_box];
    double *CinBox = new double[2*xsize*ysize/config_box/config_box];

    sprintf(fname,"NRVM_box%d_Lx%dLy%drhoA%.2frhoB%.2fvA%.2fvB%.2fTA%.2fTB%.2fJAA%.2fJAB%.2fJBA%.2fJBB%.2f.dat",config_box,xsize, ysize, rho_A, rho_B, v_A, v_B, T_A, T_B, J_AA, J_AB, J_BA, J_BB);
    writeFile2.open(fname, std::ios::binary);

    writeFile << 0 <<" " <<OP1.x<<" " <<OP1.y<<" " <<OP2.x<<" " <<OP2.y<<" "<< Chi1<<" "<< Chi2 <<" \n";

    NRVM.GetBoxConf(NinBox,VinBox,CinBox);

    for (int i=0;i<2*xsize*ysize/config_box/config_box;i++){
        writeFile2.write(reinterpret_cast<const char*>(&NinBox[i]), sizeof(int));
        writeFile2.write(reinterpret_cast<const char*>(&VinBox[i].x), sizeof(double));
        writeFile2.write(reinterpret_cast<const char*>(&VinBox[i].y), sizeof(double));
        writeFile2.write(reinterpret_cast<const char*>(&CinBox[i]), sizeof(double));
    }

    for(double t=Delta_t_meas; t<=total_t; t+=Delta_t_meas) {

        NRVM.TimeEvolution(Delta_t_meas);        
        NRVM.GetOPS(OP1,OP2,Chi1,Chi2); 

        writeFile << t <<" " <<OP1.x<<" " <<OP1.y<<" " <<OP2.x<<" " <<OP2.y<<" "<< Chi1<<" "<< Chi2 <<" \n";

        if (config_box!=0 && (int)(t/Delta_t_meas+0.01)%1==0)
        {    
            NRVM.GetBoxConf(NinBox,VinBox,CinBox);

            for (int i=0;i<2*xsize*ysize/config_box/config_box;i++){
                writeFile2.write(reinterpret_cast<const char*>(&NinBox[i]), sizeof(int));
                writeFile2.write(reinterpret_cast<const char*>(&VinBox[i].x), sizeof(double));
                writeFile2.write(reinterpret_cast<const char*>(&VinBox[i].y), sizeof(double));
                writeFile2.write(reinterpret_cast<const char*>(&CinBox[i]), sizeof(double));
            }
        }

    }

    writeFile.close();
    writeFile2.close();

}
