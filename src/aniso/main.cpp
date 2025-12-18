#include "aniso/aniso.hpp"
#include "shared/GQTable.hpp"
#include "shared/iofunc.hpp"

#include <iostream>
#include <filesystem>

int main (int argc, char **argv){
    // read model name
    if(argc != 6 &&  argc != 7) {
        printf("Usage: ./surfani modelfile f1 f2 nt phi [KERNEL_TYPE = 0]\n");
        printf("freqs = logspace(log10(f1),log10(f2),nt)\n");
        exit(1);
    }

    // initialize GLL
    GQTable:: initialize();

    // read mesh 
    const char *filename = argv[1];
    specswd::Mesh mesh;
    mesh.read_model(filename);
    mesh.create_model_attributes();
    int nz = mesh.nz_tomo;

    // check if it's love wave
    if(mesh.SWD_TYPE != 2) {
        printf("THis module can only handle fully anisotropic wave!\n");
        exit(1);
    }

    // print info to debug
    mesh.print_model();

    // Period
    int nt;
    float f1,f2;
    sscanf(argv[2],"%g",&f1); sscanf(argv[3],"%g",&f2);
    sscanf(argv[4],"%d",&nt);
    f1 = std::log10(f1); f2 = std::log10(f2);
    if(f1 > f2) std::swap(f1,f2);
    std::vector<double> freq(nt);
    for(int it = 0; it < nt; it ++) {
        double coef = (nt - 1);
        if(coef == 0.) coef = 1.;
        coef = 1. / coef;
        double f = f1 + (f2 - f1) * coef * it;
        freq[it] = std::pow(10,f);
    }
    float phi;
    sscanf(argv[5],"%f",&phi);

    int KERNEL_TYPE = 1;
    if(argc == 7) {
        sscanf(argv[5],"%d",&KERNEL_TYPE);
    }

    // initialize solver
    specswd::SolverAniso sol;

    // create output dir
    if(!std::filesystem::exists("out/"))
        std::filesystem::create_directory("out/");

    // open file to write out meta data
    FILE *fp = fopen("out/swd.txt","w");
    FILE *fio = fopen("out/database.bin","wb");
    for(int it = 0; it < nt; it ++) {
        fprintf(fp,"%g ",1. / freq[it]);
    }
    fprintf(fp,"\n");

    // write meta data int database
    using specswd::write_binary_f;
    int nkers = 23,ncomp = 3;
    write_binary_f(fio,&mesh.SWD_TYPE,1);
    write_binary_f(fio,&mesh.HAS_ATT,1);
    if(mesh.HAS_ATT) {
        nkers = 24 + mesh.nQani;
    }
    write_binary_f(fio,&nz,1);
    write_binary_f(fio,&nkers,1);
    write_binary_f(fio,&ncomp,1);

    // loop every frequency to compute phase velocity
    // compute phase velocity for each frequency
    for(int it = 0; it < nt; it ++) {
        // create database
        mesh.create_database(freq[it],phi);

        // prepare all matrices
        sol.prepare_matrices(mesh);

        // write coordinates
        write_binary_f(fio,mesh.znodes.data(),mesh.znodes.size());

        // constants
        int ng = mesh.nglob_el*3 + mesh.nglob_ac;

        // compute eigenvalues
        using specswd::scmplx;
        std::vector<scmplx> ur,ul,displ;
        if(!mesh.HAS_ATT) {
            std::vector<float> c,frekl;
            sol.compute_egn(mesh,c,ur,ul,true);
            int npts = mesh.ibool.size();

            // save phase/group velocity
            int nc = c.size();
            for(int ic = 0; ic < nc; ic ++) {
                float u,uphi;
                sol.group_vel(mesh,c[ic],&ur[ic*ng],&ul[ic*ng],u,uphi);
                fprintf(fp,"%d %g %g %g %g %d\n",it,c[ic],mesh.phi*180/M_PI,u,uphi,ic);

                // save displacement
                displ.resize(3*npts);
                sol.egn2displ(mesh,c[ic],&ur[ic*ng],displ.data());
                write_binary_f(fio,displ.data(),npts*3);
            }
        }
        else {
            std::vector<scmplx> c,frekl;
            sol.compute_egn_att(mesh,c,ur,ul,true);
            int npts = mesh.ibool.size();
            int nc = c.size();
            for(int ic = 0; ic < nc; ic ++) {
                scmplx ux,uy;
                sol.group_vel_att(mesh,c[ic],&ur[ic*ng],&ul[ic*ng],ux,uy);
                float cp = c[ic].real();
                float qc = 0.5 * c[ic].real() / c[ic].imag();
                fprintf(fp,"%d %g %g %g %g %g %d\n",it,cp,qc,mesh.phi*180/M_PI,ux.real(),uy.real(),ic);

                // save displacement
                displ.resize(3*npts);
                sol.egn2displ_att(mesh,c[ic],&ur[ic*ng],displ.data());
                write_binary_f(fio,displ.data(),npts*3);
            }
        }
    }

    // close file
    fclose(fio);
    fclose(fp);
    
    return 0;
}