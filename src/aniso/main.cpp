#include "aniso/aniso.hpp"
#include "shared/GQTable.hpp"
#include "shared/iofunc.hpp"

#include <iostream>
#include <filesystem>

using specswd::real_t;
using specswd::complex_t;

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
    auto mesh = std::make_unique<specswd::Mesh>();
    mesh->read_model(filename);
    mesh->create_model_attributes();
    int nz = mesh->nz_tomo;

    // check if it's love wave
    if(mesh->SWD_TYPE != 2) {
        printf("THis module can only handle fully anisotropic wave!\n");
        exit(1);
    }

    // print info to debug
    mesh->print_model();

    // Period
    int nt;
    real_t f1,f2;
    sscanf(argv[2],"%lg",&f1); sscanf(argv[3],"%lg",&f2);
    sscanf(argv[4],"%d",&nt);
    f1 = std::log10(f1); f2 = std::log10(f2);
    if(f1 > f2) std::swap(f1,f2);
    std::vector<real_t> freq(nt);
    for(int it = 0; it < nt; it ++) {
        real_t coef = (nt - 1);
        if(coef == 0.) coef = 1.;
        coef = 1. / coef;
        real_t f = f1 + (f2 - f1) * coef * it;
        freq[it] = std::pow(10,f);
    }
    real_t phi;
    sscanf(argv[5],"%lg",&phi);

    int KERNEL_TYPE = 1;
    if(argc == 7) {
        sscanf(argv[6],"%d",&KERNEL_TYPE);
    }

    // initialize solver
    auto sol = std::make_unique<specswd::SolverAniso>();
    sol->build(mesh.get());

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
    int ncomp = 3;
    int nkers = sol->nkers_el + sol->nkers_ac;
    write_binary_f(fio,&mesh->SWD_TYPE,1);
    write_binary_f(fio,&mesh->HAS_ATT,1);
    write_binary_f(fio,&nz,1);
    write_binary_f(fio,&nkers,1);
    write_binary_f(fio,&ncomp,1);

    // loop every frequency to compute phase velocity
    // compute phase velocity for each frequency
    for(int it = 0; it < nt; it ++) {
        // create database
        mesh->create_database(freq[it],phi);

        // prepare all matrices
        sol->prepare_matrices();

        // write coordinates
        write_binary_f(fio,mesh->znodes.data(),mesh->znodes.size());

        // compute eigenvalues
        sol->compute_egn(true);

        // compute group velocity
        sol->compute_group_vel();

        // compute eigenvalues
        std::vector<complex_t> ur,ul,displ;
        if(!mesh->HAS_ATT) {
            std::vector<real_t> c,ux,uy;

            // allocate phase/group velocity
            int nc = sol->c_phase.size();
            c.resize(nc);
            ux.resize(nc);
            uy.resize(nc);

            // get phase/group velocity
            for(int ic = 0; ic < nc; ic ++) {
                real_t temp;
                real_t tempx,tempy,tempx_i,tempy_i;
                sol->get_phase_vel(ic,c[ic],temp);
                sol->get_group_vel(ic,tempx,tempx_i,tempy,tempy_i);
                ux[ic] = tempx;
                uy[ic] = tempy;
 
                // compute phi 
                real_t uphi = std::atan2(tempy,tempx) * 180. / M_PI;
                real_t u = std::sqrt(tempx * tempx + tempy * tempy);
                fprintf(fp,"%d %g %g %g %g %d\n",it,c[ic],mesh->phi*180/M_PI,u,uphi,ic);
                
                // save displacement
                int npts = mesh->ibool.size();
                displ.resize(3*npts);
                sol->egn2displ(ic,displ.data());
                write_binary_f(fio,displ.data(),npts*3);
            }
        }
        else {
            std::vector<complex_t> c,ux,uy;

            // allocate phase/group velocity
            int nc = sol->c_phase.size();
            c.resize(nc);
            ux.resize(nc);
            uy.resize(nc);

            for(int ic = 0; ic < nc; ic ++) {
                real_t tempx,tempy,tempx_i,tempy_i;
                sol->get_phase_vel(ic,tempx,tempx_i);
                c[ic] = complex_t{tempx,tempx_i};
                sol->get_group_vel(ic,tempx,tempx_i,tempy,tempy_i);
                ux[ic] = complex_t{tempx,tempx_i};
                uy[ic] = complex_t{tempy,tempy_i};

                real_t cp = c[ic].real();
                real_t qc = 0.5 * c[ic].real() / c[ic].imag();
                fprintf(fp,"%d %g %g %g %g %g %d\n",it,cp,qc,mesh->phi*180/M_PI,ux[ic].real(),uy[ic].real(),ic);

                // save displacement
                int npts = mesh->ibool.size();
                displ.resize(3*npts);
                sol->egn2displ(ic,displ.data());
                write_binary_f(fio,displ.data(),npts*3);
            }
        }
    }

    // close file
    fclose(fio);
    fclose(fp);
    
    return 0;
}